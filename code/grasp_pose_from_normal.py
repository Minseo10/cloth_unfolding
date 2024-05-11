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
import grasp_planning as gp

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

logfile = f"log_{datetime.now().strftime('%H_%M_%S')}.txt"

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


def print_console_and_file(msg, output_dir):
    print(msg)
    with open(output_dir / logfile, 'a') as f:
        f.write(msg)
        f.write('\n')


# 적당히 뾰족한 점들 중 가장 x 방향으로 카메라와 가까운 점
def select_best_point_1(edge_pcd : o3d.geometry.PointCloud, output_dir: Path = None, debug = False):
    edge_points = np.asarray(edge_pcd.points).copy()
    point_idx = np.argmin(edge_points[:, 0])  # idx in edge points array
    best_point = edge_points[point_idx]

    if debug:
        debug_pcd = color_near_specific_point(
            np.asarray(edge_pcd.points).copy(), np.asarray(edge_pcd.colors).copy(),
            [best_point], [BLUE_COLOR], [0.01]
        )
        if output_dir:
            output_path = str(output_dir / f"best_point_ftn_1_{datetime.now().strftime('%H_%M_%S')}.ply")
            o3d.io.write_point_cloud(output_path, debug_pcd)
            print(f"{output_path} saved")


    return best_point


# # 적당히 뾰족한 점들 중 앞에 나와 있는 점을 추린 후 그 중 가장 뾰족한 점을 best point 로 결정
# def select_best_point_2(edge_pcd, output_dir):
#     edge_points = np.asarray(edge_pcd.points).copy()
#     colors = np.asarray(edge_pcd.colors).copy()
#
#     condition = edge_points[:, 0] < np.min(edge_points[:, 0]) + 0.05
#     colors[~condition] = [0, 0, 0]
#     point_idx = np.argmax(colors[:, 0])
#     best_point = edge_points[point_idx]
#
#     # check min x point with blue color
#     if debug:
#         debug_pcd = color_near_specific_point(
#             np.asarray(edge_pcd.points).copy(), np.asarray(edge_pcd.colors).copy(),
#             [best_point], [BLUE_COLOR], [0.01]
#         )
#
#         if output_dir:
#             output_path = str(output_dir / "best_point_ftn_2.ply")
#             o3d.io.write_point_cloud(output_path, debug_pcd)
#             print(f"{output_path} saved")
#
#
#     return best_point


# 적당히 뾰족한 애들 중(top 80%) 제일 앞에 튀어나와 있는 점 100개를 추리고 그 중 가장 밑에 있는 점 찾기
def select_best_point_3(min_x, max_x, edge_pcd, output_dir):
    edge_points = np.asarray(edge_pcd.points).copy()
    edge_colors = np.asarray(edge_pcd.colors).copy()

    sharp_percent = 0.8
    sharp_cutline = np.min(edge_colors[:, 0]) * sharp_percent + (1 - sharp_percent) * np.max(edge_colors[:, 0])
    sharp_mask = edge_colors[:, 0] > sharp_cutline

    edge_points = edge_points[sharp_mask]
    sorted_indices = np.argsort(edge_points[:, 0])
    sorted_points = edge_points[sorted_indices]

    # x 값이 해당 범위 안에 있는 포인트 추출
    small_x_points = sorted_points[min_x:max_x]

    # 그 중 z 값이 가장 작은 포인트 추출
    min_z_index = np.argmin(small_x_points[:, 2])
    best_point = small_x_points[min_z_index]

    # check min x point with blue color
    if debug:
        debug_pcd = color_near_specific_point(
            np.asarray(edge_pcd.points).copy(), np.asarray(edge_pcd.colors).copy(),
            [best_point], [BLUE_COLOR], [0.01]
        )

        if output_dir:
            output_path = str(output_dir / f"best_point_ftn_3_{datetime.now().strftime('%H_%M_%S')}.ply")
            o3d.io.write_point_cloud(output_path, debug_pcd)
            print(f"{output_path} saved")

    return best_point


# 3번째 방식으로 best point 2개 뽑기
def select_best_two_point(edge_pcd, output_dir):
    edge_points = np.asarray(edge_pcd.points).copy()
    edge_colors = np.asarray(edge_pcd.colors).copy()

    sharp_percent = 0.8
    sharp_cutline = np.min(edge_colors[:, 0]) * sharp_percent + (1 - sharp_percent) * np.max(edge_colors[:, 0])
    sharp_mask = edge_colors[:, 0] > sharp_cutline

    edge_points = edge_points[sharp_mask]
    sorted_indices = np.argsort(edge_points[:, 0])
    sorted_points = edge_points[sorted_indices]

    best_points = []

    def select_near_x_range_and_smallest_z_point(min_x, max_x):
        small_x_points = sorted_points[min_x:max_x]

        # 그 중 z 값이 가장 작은 포인트 추출
        min_z_index = np.argmin(small_x_points[:, 2])
        best_point = small_x_points[min_z_index]

        return best_point

    best_points.append(select_near_x_range_and_smallest_z_point(0, 100))
    best_points.append(select_near_x_range_and_smallest_z_point(400, 500))

    # check min x point with blue color
    if debug:
        debug_pcd = color_near_specific_point(
            np.asarray(edge_pcd.points).copy(), np.asarray(edge_pcd.colors).copy(),
            best_points, [BLUE_COLOR, GREEN_COLOR], [0.01, 0.01]
        )

        if output_dir:
            output_path = str(output_dir / f"best_two_points_ftn_3_{datetime.now().strftime('%H_%M_%S')}.ply")
            o3d.io.write_point_cloud(output_path, debug_pcd)
            print(f"{output_path} saved")

    return best_points


def calculate_normal_vector(pcd, point, output_dir):
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
        print_console_and_file(f"Normal vector: {normal}", output_dir)

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
            print_console_and_file(f"{output_path} saved", output_dir)

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

    o3d.visualization.draw_geometries([pcd])

    return pcd


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


# # normal vs (mean point - grasp point)
# def get_strategies_priorities(sample, processing_dir, debug):
#     grasp_point = select_best_point_3(
#         edge_pcd=sample.processing.edge_point_cloud,
#         output_dir=processing_dir,
#     )
#     mean_point = calculate_mean_point_near_specific_point(
#         pcd=sample.processing.cropped_point_cloud,
#         specific_point=grasp_point,
#         debug=debug,
#         output_dir=processing_dir
#     )
#     approach = mean_point - grasp_point
#     distance = np.linalg.norm(approach)
#     is_inside_point = distance < 0.018
#
#     print(f"distance: {distance}")
#     print(f"is inside point: {is_inside_point}")
#
#     if is_inside_point:
#         priorities = [1, 3, 2, 0] ###### TODO: must check #####
#     else:
#         priorities = [0, 1, 3, 2]
#
#     print(f"priorities: {priorities}")
#     return priorities


def visualize_grasp_pose(sample, grasp_pose_fixed, output_path, debug):
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(0.1, center=(0, 0, 0))
    mesh = copy.deepcopy(mesh).transform(grasp_pose_fixed)

    if debug:
        o3d.visualization.draw_geometries([sample.processing.cropped_point_cloud, mesh])

    X_W_C = sample.observation.camera_pose_in_world
    intrinsics = sample.observation.camera_intrinsics

    image_bgr = cv2.cvtColor(sample.observation.image_left, cv2.COLOR_RGB2BGR)
    draw_pose(image_bgr, grasp_pose_fixed, intrinsics, X_W_C, 0.1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))
    plt.imshow(image_rgb)
    plt.title("Example grasp pose")

    if output_path:
        plt.savefig(output_path)
        print(f"{output_path} saved")

    plt.show()


def method1(x_offset, sample, processing_dir, debug):
    print_console_and_file('method1 is selected.', processing_dir)
    grasp_point = select_best_point_3(
        min_x=x_offset, max_x=x_offset+100,
        edge_pcd=sample.processing.edge_point_cloud,
        output_dir=processing_dir
    )

    mean_point = calculate_mean_point_near_specific_point(
        pcd=sample.processing.cropped_point_cloud,
        specific_point=grasp_point,
        debug=debug,
        output_dir=processing_dir
    )

    approach = mean_point - grasp_point
    distance = np.linalg.norm(approach)
    is_inside_point = distance < 0.018
    grasp_pose_fixed = None

    print_console_and_file(f"distance: {distance}", processing_dir)
    print_console_and_file(f"is inside point: {is_inside_point}", processing_dir)

    if is_inside_point:
        approach = calculate_normal_vector(
            pcd=sample.processing.cropped_point_cloud,
            point=grasp_point,
            output_dir=processing_dir
        )

        # grasp direction (vector) -> grasp pose (rpy)
        # Calculate world frame's z-axis in camera frame
        z_axis_world = np.array([0, 0, 1])
        plane_normal = np.cross(z_axis_world, approach)
        aligned_normal = np.cross(z_axis_world, plane_normal)
        grasp_z = aligned_normal / np.linalg.norm(aligned_normal)
        grasp_y = z_axis_world
        grasp_y = (-1) * grasp_y
        grasp_x = np.cross(grasp_y, grasp_z)
        grasp_x /= np.linalg.norm(grasp_x)

        grasp_pose_fixed = grasp_hanging_cloth_pose(grasp_point, grasp_z, 0.07)

    else:
        approach = mean_point - grasp_point
        grasp_pose_fixed = grasp_hanging_cloth_pose(grasp_point, approach, 0.07)

    return grasp_pose_fixed


def method2(sample, processing_dir, debug):
    print_console_and_file('method2 is selected.', processing_dir)
    grasp_point = select_best_point_1(
        edge_pcd=sample.processing.edge_point_cloud,
        output_dir=processing_dir,
        debug=debug
    )

    mean_point = calculate_mean_point_near_specific_point(
        pcd=sample.processing.cropped_point_cloud,
        specific_point=grasp_point,
        debug=debug,
        output_dir=processing_dir
    )

    tuning_vector = mean_point - grasp_point
    distance = np.linalg.norm(tuning_vector)
    is_inside_point = distance < 0.018

    print_console_and_file(f"distance: {distance}", processing_dir)
    print_console_and_file(f"is inside point: {is_inside_point}", processing_dir)

    if not is_inside_point:
        tuning_vector = tuning_vector / np.linalg.norm(tuning_vector)
        tuning_scale = 0.02

        if debug:
            point_1 = grasp_point
            point_2 = grasp_point + tuning_vector * tuning_scale

            mesh1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=point_1)
            mesh2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=point_2)
            o3d.visualization.draw_geometries([sample.processing.edge_point_cloud, mesh1, mesh2])

        grasp_point += tuning_vector * tuning_scale

    approach = np.asarray([1, 0, 0])
    return grasp_hanging_cloth_pose(grasp_point, approach, 0.07)


if __name__ == '__main__':
    debug = False
    from_server = False
    to_server = False

    for i in range(500):
        start_time = time.time()

        server_url = "https://robotlab.ugent.be"

        # download input files
        if from_server:
            data_dir = Path("../datasets")
            dataset_dir = data_dir / "downloaded_dataset_0000"

            observation_dir, sample_id = download_latest_observation(dataset_dir, server_url)
            sample_dir = Path(observation_dir + "/../")

        else:
            sample_id = f"sample_{'{0:06d}'.format(i)}"
            sample_dir = Path(f"../datasets/cloth_competition_dataset_0000/{sample_id}")
            observation_dir = sample_dir / "observation_start"

        print(f">>> processing {sample_dir}")
        middle_time = time.time()

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

        if debug:
            o3d.visualization.draw_geometries([edge_pointcloud])

        idx = 1
        x_offset = 0
        grasp_pose_fixed = None
        is_success = None

        while True:
            if idx == 1 or idx == 2 or idx % 2 == 0:
                grasp_pose_fixed = method1(x_offset, sample, processing_dir, debug)
                x_offset += 300
            else:
                grasp_pose_fixed = method2(sample, processing_dir, debug)

            # check motion planning
            is_success = gp.is_grasp_executable_fn(sample.observation, grasp_pose_fixed)

            if not is_success:
                for angle_idx, angle in enumerate([0, np.pi / 2, np.pi, 3 * np.pi / 2]):
                    print_console_and_file(f"grasp_pose_fixed: \n{grasp_pose_fixed}", processing_dir)
                    rotated_pose = grasp_pose_fixed.copy()

                    gp.rotate_grasps(rotated_pose, angle, 0.5)
                    print_console_and_file(f"rotated_pose_{angle_idx}: \n{rotated_pose}", processing_dir)

                    if gp.is_grasp_executable_fn(sample.observation, rotated_pose):
                        is_success = True
                        grasp_pose_fixed = rotated_pose

                    else:
                        is_success = False
                        visualize_grasp_pose(sample, rotated_pose, processing_dir / f"grasp_pose_failed_idx{idx}_angle{angle_idx}.png", debug)
    
            if is_success:
                print_console_and_file(f"Planning succeed!: {processing_dir}", processing_dir)
                print_console_and_file(f"Grasp pose is {grasp_pose_fixed}", processing_dir)
                visualize_grasp_pose(sample, grasp_pose_fixed, processing_dir / f"grasp_pose_success_idx{idx}.png", debug)
                break

            idx = idx + 1

        # save
        grasps_dir = f"data/grasps_{sample_id}"
        grasp_pose_file = save_grasp_pose(grasps_dir, grasp_pose_fixed)

        if to_server:
            upload_grasp(grasp_pose_file, "Ewha Glab", sample_id, server_url)

        end_time = time.time()
        download_time = middle_time - start_time
        execution_time = end_time - start_time
        print_console_and_file(f"다운로 시간: {download_time:.6f}초", processing_dir)
        print_console_and_file(f"코드 실행 시간: {execution_time:.6f}초", processing_dir)