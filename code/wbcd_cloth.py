import open3d as o3d
import numpy as np
from airo_typing import (
    NumpyIntImageType,
    PointCloud,
    HomogeneousMatrixType,
)
from airo_dataset_tools.data_parsers.pose import Pose

import Difference_Eigenvalues as de
import segment_crop as seg
from pathlib import Path
# import grasp_planning as gp

from cloth_tools.dataset.download import download_latest_observation
from cloth_tools.dataset.bookkeeping import datetime_for_filename
from cloth_tools.dataset.format import load_competition_observation, CompetitionObservation

from dataclasses import dataclass

import matplotlib.pyplot as plt

import json
import os
import cv2
import copy
import time


from cloth_tools.visualization.opencv import draw_pose
from cloth_tools.dataset.upload import upload_grasp
from cloth_tools.annotation.grasp_annotation import grasp_hanging_cloth_pose

from datetime import datetime

MIN = -4
MAX = 4

RED_COLOR = np.array([1, 0, 0])
GREEN_COLOR = np.array([0, 1, 0])
BLUE_COLOR = np.array([0, 0, 1])

@dataclass
class ProcessingData:
    segmented_img: NumpyIntImageType
    cloth_bbox: list
    cropped_point_cloud: o3d.geometry.PointCloud
    edge_point_cloud: o3d.geometry.PointCloud
    pc_wo_table: o3d.geometry.PointCloud

@dataclass
class Sample:
    observation: CompetitionObservation
    processing: ProcessingData


def calculate_grasp_point(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    x = data["position_in_meters"]["x"]
    y = data["position_in_meters"]["y"]
    z = data["position_in_meters"]["z"]

    # Grasp direction
    roll, pitch, yaw = data["rotation_euler_xyz_in_radians"]["roll"], data["rotation_euler_xyz_in_radians"]["pitch"], \
    data["rotation_euler_xyz_in_radians"]["yaw"]

    # Convert Euler angles to rotation matrix
    R = np.array([
        [np.cos(yaw) * np.cos(pitch), np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll),
         np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
        [np.sin(yaw) * np.cos(pitch), np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll),
         np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
        [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]
    ])

    # Grasp direction vector
    grasp_dir_vec = np.dot(R, np.array([0, 0, 1]))
    distance = 0.05

    # Calculate actual grasp point by moving grasp point along grasp direction by given distance
    actual_grasp_point = np.array([x, y, z]) - distance * grasp_dir_vec

    return actual_grasp_point, R


def rotation_matrix_to_rpy(rotation_matrix):
    # Extract roll (x-axis rotation)
    roll = np.arctan2(rotation_matrix[2][1], rotation_matrix[2][2])

    # Extract pitch (y-axis rotation)
    pitch = np.arctan2(-rotation_matrix[2][0], np.sqrt(rotation_matrix[2][1] ** 2 + rotation_matrix[2][2] ** 2))

    # Extract yaw (z-axis rotation)
    yaw = np.arctan2(rotation_matrix[1][0], rotation_matrix[0][0])

    return roll, pitch, yaw


# among sharp points + 아래쪽에 있음 + 양옆으로 제일 멀다
def select_best_points(num_z, sharp_percent, edge_pcd, output_dir, horizontal_axis, vertical_axis, debug=False):
    # step 1: get sharp points (red channel 기준)
    edge_points = np.asarray(edge_pcd.points).copy()
    edge_colors = np.asarray(edge_pcd.colors).copy()

    sharp_cutline = np.min(edge_colors[:, 0]) * sharp_percent + (1 - sharp_percent) * np.max(edge_colors[:, 0])
    sharp_mask = edge_colors[:, 0] > sharp_cutline
    sharp_points = edge_points[sharp_mask]

    if len(sharp_points) == 0:
        raise ValueError("No sharp points found.")

    # step 2: select lowest (bottom) points along vertical axis
    vertical_axis = vertical_axis / np.linalg.norm(vertical_axis)
    z_values = sharp_points @ vertical_axis  # projection onto vertical axis
    lowest_indices = np.argsort(z_values)[:num_z]
    small_z_points = sharp_points[lowest_indices]
    if debug:
        pcd_debug = o3d.geometry.PointCloud()
        pcd_debug.points = o3d.utility.Vector3dVector(small_z_points)
        pcd_debug.paint_uniform_color(RED_COLOR)

        pcd_debug2 = o3d.geometry.PointCloud()
        pcd_debug2.points = o3d.utility.Vector3dVector(sharp_points)
        pcd_debug.paint_uniform_color(BLUE_COLOR)

        o3d.visualization.draw_geometries([pcd_debug], window_name="100 bottom points")

    # step 3: among those, pick 2 most horizontally distant points
    horizontal_axis = horizontal_axis / np.linalg.norm(horizontal_axis)
    x_values = small_z_points @ horizontal_axis  # projection onto horizontal axis
    min_idx = np.argmin(x_values)
    max_idx = np.argmax(x_values)

    best_points = np.array([small_z_points[min_idx], small_z_points[max_idx]])

    # check min x point with blue color
    if debug:
        debug_pcd1 = color_near_specific_point(
            np.asarray(edge_pcd.points).copy(), np.asarray(edge_pcd.colors).copy(),
            [best_points[0]], [BLUE_COLOR], [0.01]
        )
        debug_pcd2 = color_near_specific_point(
            np.asarray(debug_pcd1.points).copy(), np.asarray(debug_pcd1.colors).copy(),
            [best_points[1]], [BLUE_COLOR], [0.01]
        )

        if output_dir:
            output_path = str(output_dir / f"best_point_ftn_3_{datetime.now().strftime('%H_%M_%S')}.ply")
            o3d.io.write_point_cloud(output_path, debug_pcd2)
            print(f"{output_path} saved")

    return best_points


def calculate_normal_vector(pcd, point, debug=False):
    # estimate normal vectors
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    # normal vector 옷 바깥쪽으로 뒤집기
    pcd.orient_normals_consistent_tangent_plane(k=15)

    distances = np.linalg.norm(pcd.points - point, axis=1)
    min_distance_index = np.argmin(distances)
    normal = np.asarray(pcd.normals)[min_distance_index]

    if debug:
        print("Normal vector: ", normal)

    return normal


def calculate_mean_point_near_specific_point(pcd, specific_point, debug, output_dir):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    # [k, idx, _] = pcd_tree.search_radius_vector_3d(best_point, 0.02)
    [k, idx, _] = pcd_tree.search_knn_vector_3d(specific_point, 3000) # 2000

    points = np.asarray(pcd.points).copy()
    colors = np.asarray(pcd.colors).copy()

    colors[idx[1:], :] = GREEN_COLOR

    near_points = points[idx[1:], :].copy()
    mean_point = near_points.mean(axis=0)

    if debug:
        debug_pcd = color_near_specific_point(
            points, colors,
            [mean_point, specific_point],
            [RED_COLOR, BLUE_COLOR],
            [0.01, 0.01]
        )

        if output_dir:
            output_path = str(output_dir / f"grasp_mean_point_{datetime.now().strftime('%H_%M_%S')}.ply")
            o3d.io.write_point_cloud(output_path, debug_pcd)
            print(f"{output_path} saved")

    return mean_point


# points, colors 매개변수 넘길 때 얕은 복사인지, 깊은 복사인지 주의 깊게 고려하고 사용할 것
def color_near_specific_point(points, colors, target_points, target_colors, tolerances):
    assert len(target_points) == len(target_colors) and len(target_colors) == len(tolerances), \
        "표시하려는 점의 개수, 색의 개수, 칠하는 영역의 범위 개수가 같아야 합니다"

    colors[:] = [0.1, 0, 0]

    for i in range(len(target_colors)):
        target_point = target_points[i]
        target_color = target_colors[i]
        tolerance = tolerances[i]

        x_condition = (points[:, 0] >= target_point[0] - tolerance) & (points[:, 0] <= target_point[0] + tolerance)
        y_condition = (points[:, 1] >= target_point[1] - tolerance) & (points[:, 1] <= target_point[1] + tolerance)
        z_condition = (points[:, 2] >= target_point[2] - tolerance) & (points[:, 2] <= target_point[2] + tolerance)
        matching_indices = np.where(x_condition & y_condition & z_condition)[0]
        colors[matching_indices] = target_color

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], window_name="color_near_specific_point")

    return pcd


def convert_to_o3d_pcd(pcd: PointCloud):
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.points)
    o3d_pcd.colors = o3d.utility.Vector3dVector(pcd.colors / 255.0)

    return o3d_pcd


def save_grasp_pose(grasps_dir: Path, grasp_poses_list: list[HomogeneousMatrixType]):
    grasp_pose_name = f"grasp_pose_{datetime_for_filename()}.json"
    grasp_pose_file = grasps_dir / grasp_pose_name

    with open(grasp_pose_file, "w") as f:
        grasp_pose_model = Pose.from_homogeneous_matrix(grasp_poses_list[0])
        json.dump(grasp_pose_model.model_dump(exclude_none=False), f, indent=4)
        grasp_pose_model = Pose.from_homogeneous_matrix(grasp_poses_list[1])
        json.dump(grasp_pose_model.model_dump(exclude_none=False), f, indent=4)

    return grasp_pose_file


def visualize_grasp_pose(sample, grasp_poses, output_path, debug):
    mesh1 = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh1.scale(0.1, center=(0, 0, 0))
    mesh1 = copy.deepcopy(mesh1).transform(grasp_poses[0])

    mesh2 = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh2.scale(0.1, center=(0, 0, 0))
    mesh2 = copy.deepcopy(mesh2).transform(grasp_poses[1])

    if debug:
        o3d.visualization.draw_geometries([sample.processing.cropped_point_cloud, mesh1, mesh2], window_name="Final Grasp Poses")

    X_W_C = sample.observation.camera_pose_in_world
    intrinsics = sample.observation.camera_intrinsics

    image_bgr1 = cv2.cvtColor(sample.observation.image_left, cv2.COLOR_RGB2BGR)
    draw_pose(image_bgr1, grasp_poses[0], intrinsics, X_W_C, 0.1)
    image_rgb1 = cv2.cvtColor(image_bgr1, cv2.COLOR_BGR2RGB)

    image_bgr2 = cv2.cvtColor(image_rgb1, cv2.COLOR_RGB2BGR)
    draw_pose(image_bgr2, grasp_poses[1], intrinsics, X_W_C, 0.1)
    image_rgb2 = cv2.cvtColor(image_bgr2, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))
    plt.imshow(image_rgb2)
    plt.title("Example grasp pose")

    plt.savefig(output_path)
    print(f"{output_path} saved")

    plt.show()


# Method 2: grasp direction fixed (forward direction)
def get_grasp_pose(sample, processing_dir, debug):
    print('method1 is selected.')
    # world frame에 대한 camera frame의 x, y, z 축을 벡터 형태로 구한다.
    cam_axes = seg.get_camera_axes_in_world(sample.observation.camera_pose_in_world)

    # TODO: modify hyperparameters
    grasp_points = select_best_points(
        num_z=300,
        sharp_percent=0.8,  # sharp 한 상위 포인트들
        edge_pcd=sample.processing.edge_point_cloud,
        output_dir=processing_dir,
        horizontal_axis=cam_axes["x"],
        vertical_axis=-1*cam_axes["y"],
        debug=debug,
    )

    inside_grasps = grasp_points.copy()
    # point가 너무 바깥쪽이면 살짝 안으로 집어넣는다
    for i, grasp_point in enumerate(grasp_points):
        mean_point = calculate_mean_point_near_specific_point(
            pcd=sample.processing.cropped_point_cloud,
            specific_point=grasp_point,
            debug=debug,
            output_dir=processing_dir
        )

        tuning_vector = mean_point - grasp_point
        distance = np.linalg.norm(tuning_vector)
        is_inside_point = distance < 0.018

        print(f"distance: {distance}")
        print(f"is inside point: {is_inside_point}")

        if not is_inside_point:
            tuning_vector = tuning_vector / np.linalg.norm(tuning_vector)
            tuning_scale = 0.02

            if debug:
                point_1 = grasp_point
                point_2 = grasp_point + tuning_vector * tuning_scale

                mesh1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=point_1)
                mesh2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=point_2)
                o3d.visualization.draw_geometries([sample.processing.edge_point_cloud, mesh1, mesh2], window_name="Grasp Points")

            inside_grasps[i] = grasp_point + tuning_vector * tuning_scale

    # appraoch direction: up to down
    approach = np.asarray([0, 0, -1])  # TODO: modify hyperparameters
    return [grasp_hanging_cloth_pose(inside_grasps[0], approach, 0.03), grasp_hanging_cloth_pose(inside_grasps[1], approach, 0.03)]

def cloth_bimanual_grasp(sample, sample_dir, rgb_image, depth_image, pointcloud, camera_intrinsic, camera_extrinsic, debug=False):
    '''
    :param sample: data structure for inputs/outputs (refer to line 44-54)
    :param sample_dir: directory to sample
    :param rgb_image: 2D RGB image for cloth segmentation (np.ndarray format)
    :param depth_image: Depth image (np.ndarray format)
    :param pointcloud: Open3D format pointcloud ()
    :param camera_intrinsic: 3x3 camera intrinsics matrix (K = [[fx,s,cx],[0,fy,cy],[0,0,1]])
    :param camera_extrinsic: pose of camera frame w.r.t world frame (4x4 homogeneous matrix)
    :param debug: if True, show intermediate results using Open3D
    :return: 2 grasp poses for each gripper
    '''
    start_time = time.time()
    # cloth segmentation
    processing_dir = sample_dir / "processing"
    mask = seg.segmentation(image=rgb_image, output_dir=processing_dir)
    sample.processing.segmented_img = mask

    if debug:
        plt.imshow(sample.processing.segmented_img)
        plt.title("Segmentation image from left observation rgb-d image")
        plt.show()

    # segmentation 이미지 중 가장 면적이 넓은 부분이 진짜 옷임. 그 bbox 를 찾아서
    cloth_bbox = seg.contour(binary_image=mask, output_dir=processing_dir, debug=debug)
    sample.processing.cloth_bbox = cloth_bbox

    # 그 bbox 를 이용해서 point cloud 를 crop 한다.
    cropped_point_cloud = seg.crop(
        bbox_coordinates=cloth_bbox,
        depth_image=depth_image,
        camera_intrinsics=camera_intrinsic,
        camera_extrinsic=camera_extrinsic,
        point_cloud=pointcloud,
        output_dir=processing_dir,
        debug=debug
    )
    sample.processing.cropped_point_cloud = cropped_point_cloud

    # remove table points
    # TODO: modify hyperparameters
    pc_wo_table = seg.remove_table(pcd = cropped_point_cloud, output_dir = processing_dir, distance_threshold = 0.01, debug = debug)
    sample.processing.pc_wo_table = pc_wo_table

    # edge extraction
    edge_pointcloud = de.extract_edge(
        pcd= sample.processing.pc_wo_table,
        output_dir=processing_dir,
        uniformed=False,
    )  # from Difference_Eigenvalues.py
    sample.processing.edge_point_cloud = edge_pointcloud

    if debug:
        o3d.visualization.draw_geometries([edge_pointcloud], window_name="Edge Points")

    idx = 1
    x_offset = 0

    # calculate grasp pose
    grasp_poses = get_grasp_pose(sample, processing_dir, debug)
    print("Grasp pose is ", grasp_poses)
    visualize_grasp_pose(sample, grasp_poses, processing_dir / f"grasp_pose_success_idx{idx}.png", debug)

    # save
    grasp_pose_file = save_grasp_pose(processing_dir, grasp_poses)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Edge detection execution time: {execution_time:.6f}s")
    return grasp_poses

if __name__ == '__main__':
    # TODO: modify input directory/formats
    sample_dir = Path(f"../datasets/cloth_competition_dataset_0000_0-9/sample_000001")  # choose among 000000-000009
    observation_dir = sample_dir / "observation_start"
    sample = Sample(
        observation=load_competition_observation(observation_dir),
        processing=ProcessingData(
            segmented_img=None,
            cloth_bbox=None,
            cropped_point_cloud=None,
            edge_point_cloud=None,
            pc_wo_table=None,
        )
    )

    grasp_poses = cloth_bimanual_grasp(
        sample=sample,
        sample_dir=sample_dir,
        rgb_image=sample.observation.image_left,
        depth_image=sample.observation.depth_image,
        pointcloud=sample.observation.point_cloud,
        camera_intrinsic=sample.observation.camera_intrinsics,
        camera_extrinsic=sample.observation.camera_pose_in_world,
        debug=True)
