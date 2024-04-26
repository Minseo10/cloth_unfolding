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

MIN = -4
MAX = 4


@dataclass
class ProcessingData:
    segmented_img: NumpyIntImageType
    cloth_bbox: list
    cropped_point_cloud: o3d.geometry.PointCloud
    edge_point_cloud: o3d.geometry.PointCloud

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


# annotated grasp pose given by organizer
def find_best_point_and_normal_vector(root_path, origin_pcd, edge_pcd):
    # best point selection
    # 현재는 bring real grasp point from grasp_pose.json
    grasp_pose_filepath = root_path + "grasp/grasp_pose.json"
    best_point, grasp_pose = calculate_grasp_point(grasp_pose_filepath)

    points_array = np.asarray(origin_pcd.points)
    normals_array = np.asarray(origin_pcd.normals)
    min_dist = 1000000
    best_point_idx = 0

    for i in range(points_array.shape[0]):
        dist = np.linalg.norm(points_array[i] - best_point)
        if dist < min_dist:
            min_dist = dist
            best_point_idx = i

    point = points_array[best_point_idx]
    normal = normals_array[best_point_idx]

    if debug:
        print("Best point: ", point, "\n")
        print("Normal vector: ", normal, "\n")

    return point, normal


# sharp edge 위의 점들 중 x 축으로 가장 톡 튀어 나와있는 점을 best point 라고 찾고, 그 점에서의 normal vector 예측값을 grasp direction 으로 찾는 방식
def find_best_point_and_normal_vector_2(pcd : o3d.geometry.PointCloud, edge_pcd : o3d.geometry.PointCloud,
                                        output_dir: Path = None, debug = False):
    edge_points = np.asarray(edge_pcd.points)
    point_idx = np.argmin(edge_points[:, 0])  # idx in edge points array
    best_point = edge_points[point_idx]

    # check min x point with blue color
    point_colors = np.asarray(edge_pcd.colors)
    point_colors[:] = [1, 0, 0]
    colors = color_near_specific_point(edge_points, point_colors, best_point, [0, 0, 1], 0.01)
    edge_pcd.colors = o3d.utility.Vector3dVector(colors)

    if debug:
        o3d.visualization.draw_geometries([edge_pcd])

    if output_dir:
        o3d.io.write_point_cloud(str(output_dir / "ftn2.ply"), edge_pcd)

    distances = np.linalg.norm(pcd.points - best_point, axis=1)
    min_distance_index = np.argmin(distances)
    normal = np.asarray(pcd.normals)[min_distance_index]

    if debug:
        print("Best point: ", best_point, "\n")
        print("Normal vector: ", normal, "\n")

    return best_point, normal


# 가장 뾰족한 점은 max sigma (red color value) 를 갖고 있고, 상당히 앞에 나와 있는 점들 중 가장 뾰족한 점을 best point 로 선정, grasp direction 은 normal vector
def find_best_point_and_normal_vector_3(pcd : o3d.geometry.PointCloud, edge_pcd : o3d.geometry.PointCloud,
                                        output_dir: Path = None, debug = False):
    edge_points = np.asarray(edge_pcd.points)
    colors = np.asarray(edge_pcd.colors)

    condition1 = edge_points[:, 0] < np.min(edge_points[:, 0]) + 0.05
    condition2 = edge_points[:, 1] < np.max(edge_points[:, 1]) - 0.05
    condition = condition1 & condition2
    colors[~condition] = [0, 0, 0]
    point_idx = np.argmax(colors[:, 0])
    best_point = edge_points[point_idx]

    # check min x point with blue color
    colors[:] = [1, 0, 0]
    colors = color_near_specific_point(edge_points, colors, best_point, [0, 0, 1], 0.01)
    edge_pcd.colors = o3d.utility.Vector3dVector(colors)

    if debug:
        o3d.visualization.draw_geometries([edge_pcd])

    if output_dir:
        o3d.io.write_point_cloud(str(output_dir / "ftn3.ply"), edge_pcd)

    distances = np.linalg.norm(pcd.points - best_point, axis=1)
    min_distance_index = np.argmin(distances)
    normal = np.asarray(pcd.normals)[min_distance_index]

    if debug:
        print("Best point: ", best_point, "\n")
        print("Normal vector: ", normal, "\n")

    return best_point, normal


# 적당히 뾰족한 애들 중 제일 앞에 튀어나와 있는 점 100개를 추리고 그 중 가장 밑에 있는 점 찾기
def find_best_point_and_normal_vector_4(pcd : o3d.geometry.PointCloud, edge_pcd : o3d.geometry.PointCloud,
                                        output_dir: Path = None, debug = False):
    edge_points = np.asarray(edge_pcd.points)
    edge_colors = np.asarray(edge_pcd.colors)

    sharp_percent = 0.8
    sharp_cutline = np.min(edge_colors[:, 0]) * sharp_percent + (1 - sharp_percent) * np.max(edge_colors[:, 0])
    sharp_mask = edge_colors[:, 0] > sharp_cutline

    edge_points = edge_points[sharp_mask]
    sorted_indices = np.argsort(edge_points[:, 0])
    sorted_points = edge_points[sorted_indices]

    # x 값이 가장 작은 포인트 추출
    x_count = 100
    small_x_points = sorted_points[:x_count]

    # 그 중 z 값이 가장 작은 포인트 추출
    min_z_index = np.argmin(small_x_points[:, 2])
    best_point = small_x_points[min_z_index]

    # check min x point with blue color
    point_colors = np.asarray(edge_pcd.colors)
    point_colors[:] = [1, 0, 0]
    colors = color_near_specific_point(edge_points, point_colors, best_point, [0, 0, 1], 0.01)
    edge_pcd.colors = o3d.utility.Vector3dVector(colors)

    if debug:
        o3d.visualization.draw_geometries([edge_pcd])

    if output_dir:
        o3d.io.write_point_cloud(str(output_dir / "ftn4.ply"), edge_pcd)

    distances = np.linalg.norm(pcd.points - best_point, axis=1)
    min_distance_index = np.argmin(distances)
    normal = np.asarray(pcd.normals)[min_distance_index]

    if debug:
        print("Best point: ", best_point, "\n")
        print("Normal vector: ", normal, "\n")

    return best_point, normal


def color_near_specific_point(points, colors, point, color, tolerance):
    x_condition = (points[:, 0] >= point[0] - tolerance) & (points[:, 0] <= point[0] + tolerance)
    y_condition = (points[:, 1] >= point[1] - tolerance) & (points[:, 1] <= point[1] + tolerance)
    z_condition = (points[:, 2] >= point[2] - tolerance) & (points[:, 2] <= point[2] + tolerance)
    matching_indices = np.where(x_condition & y_condition & z_condition)[0]
    colors[matching_indices] = color

    return colors


def convert_to_o3d_pcd(pcd: PointCloud):
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.points)
    o3d_pcd.colors = o3d.utility.Vector3dVector(pcd.colors / 255.0)

    return o3d_pcd


def save_grasp_pose(grasps_dir: str, grasp_pose_fixed: HomogeneousMatrixType) -> str:
    os.makedirs(grasps_dir, exist_ok=True)

    grasp_pose_name = f"grasp_pose_{datetime_for_filename()}.json"
    grasp_pose_file = os.path.join(grasps_dir, grasp_pose_name)

    with open(grasp_pose_file, "w") as f:
        grasp_pose_model = Pose.from_homogeneous_matrix(grasp_pose_fixed)
        json.dump(grasp_pose_model.model_dump(exclude_none=False), f, indent=4)

    return grasp_pose_file


if __name__ == '__main__':
    debug = True
    from_server = False
    to_server = False

    server_url = "https://robotlab.ugent.be"

    # download input files
    if from_server:
        data_dir = Path("../datasets")
        dataset_dir = data_dir / "downloaded_dataset_0000"

        observation_dir, sample_id = download_latest_observation(dataset_dir, server_url)
        sample_dir = Path(observation_dir + "/../")

    else:
        sample_id = f"sample_{'{0:06d}'.format(1)}"
        sample_dir = Path(f"../datasets/cloth_competition_dataset_0001/{sample_id}")
        observation_dir = sample_dir / "observation_start"

    start_time = time.time()

    sample = Sample(
        observation=load_competition_observation(observation_dir),
        processing=ProcessingData(
            segmented_img=None,
            cloth_bbox=None,
            cropped_point_cloud=None,
            edge_point_cloud=None,
        )
    )

    # cloth segmentation
    processing_dir = sample_dir / "processing"
    mask = seg.segmentation(image=sample.observation.image_left, output_dir=processing_dir)
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
        bbox_coordinates = cloth_bbox,
        depth_image = sample.observation.depth_image,
        camera_intrinsics = sample.observation.camera_intrinsics,
        camera_extrinsic = sample.observation.camera_pose_in_world,
        point_cloud = sample.observation.point_cloud,
        output_dir = processing_dir,
        debug = debug
    )
    sample.processing.cropped_point_cloud = cropped_point_cloud

    # edge extraction
    edge_pointcloud = de.extract_edge(
        pcd = sample.processing.cropped_point_cloud,
        output_dir = processing_dir,
        uniformed = False,
    )  # from Difference_Eigenvalues.py
    sample.processing.edge_point_cloud = edge_pointcloud

    # estimate normal vectors
    sample.processing.cropped_point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    # normal vector 옷 바깥쪽으로 뒤집기
    sample.processing.cropped_point_cloud.orient_normals_consistent_tangent_plane(k=15)

    # find corresponding point in point cloud
    point, normal = find_best_point_and_normal_vector_2(
        pcd = sample.processing.cropped_point_cloud,
        edge_pcd = sample.processing.edge_point_cloud,
        output_dir = processing_dir,
        debug = debug,
    )
    point, normal = find_best_point_and_normal_vector_3(
        pcd = sample.processing.cropped_point_cloud,
        edge_pcd = sample.processing.edge_point_cloud,
        output_dir = processing_dir,
        debug = debug,
    )
    point, normal = find_best_point_and_normal_vector_4(
        pcd=sample.processing.cropped_point_cloud,
        edge_pcd=sample.processing.edge_point_cloud,
        output_dir=processing_dir,
        debug=debug,
    )

    # grasp direction (vector) -> grasp pose (rpy)
    # Calculate world frame's z-axis in camera frame
    z_axis_world = np.array([0, 0, 1])
    plane_normal = np.cross(z_axis_world, normal)
    aligned_normal = np.cross(z_axis_world, plane_normal)
    grasp_z = aligned_normal / np.linalg.norm(aligned_normal)
    grasp_y = z_axis_world
    grasp_y = (-1) * grasp_y
    grasp_x = np.cross(grasp_y, grasp_z)
    grasp_x /= np.linalg.norm(grasp_x)

    # # add grasp depth (0.5cm)
    # point = point + grasp_z * 0.005
    #
    # # grasp pose w.r.t world frame
    # grasp_R = [[grasp_x[0], grasp_y[0], grasp_z[0]], [grasp_x[1], grasp_y[1], grasp_z[1]], [grasp_x[2], grasp_y[2], grasp_z[2]]]
    # T = np.eye(4)
    # T[:3, :3] = grasp_R
    # T[0, 3] = point[0]
    # T[1, 3] = point[1]
    # T[2, 3] = point[2]
    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # mesh.scale(0.1, center=(0,0,0))
    # mesh = copy.deepcopy(mesh).transform(T)
    #
    # # print grasp pose
    # roll, pitch, yaw = rotation_matrix_to_rpy(grasp_R)
    #
    # print("X (meters):", point[0])
    # print("Y (meters):", point[1])
    # print("Z (meters):", point[2])
    # print("Roll (radians):", roll)
    # print("Pitch (radians):", pitch)
    # print("Yaw (radians):", yaw)
    #
    # if debug:
    #     print("Roll (radians):", roll)
    #     print("Pitch (radians):", pitch)
    #     print("Yaw (radians):", yaw)

    # visualize grasp pose at the grasp point
    # start_point = point.tolist()
    # end_point = (point + 0.1 * normal).tolist()
    #
    # line_set = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector([start_point, end_point]),
    #     lines=o3d.utility.Vector2iVector([[0, 1]]),
    # )
    # line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
    # line_set_line_width = 4



    # upload to server
    grasp_pose_fixed = grasp_hanging_cloth_pose(point, grasp_z, 0.05)

    # open 3d visualize
    if debug:
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        mesh.scale(0.1, center=(0,0,0))
        mesh = copy.deepcopy(mesh).transform(grasp_pose_fixed)
        o3d.visualization.draw_geometries([sample.processing.cropped_point_cloud, mesh])

    X_W_C = sample.observation.camera_pose_in_world
    intrinsics = sample.observation.camera_intrinsics

    image_bgr = cv2.cvtColor(sample.observation.image_left, cv2.COLOR_RGB2BGR)
    draw_pose(image_bgr, grasp_pose_fixed, intrinsics, X_W_C, 0.1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if debug:
        plt.figure(figsize=(10, 5))
        plt.imshow(image_rgb)
        plt.title("Example grasp pose")
        plt.show()

    grasps_dir = f"data/grasps_{sample_id}"
    grasp_pose_file = save_grasp_pose(grasps_dir, grasp_pose_fixed)

    if to_server:
        upload_grasp(grasp_pose_file, "test", sample_id, server_url)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"코드 실행 시간: {execution_time:.6f}초")