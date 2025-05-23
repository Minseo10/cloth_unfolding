import numpy as np
import cv2
import torch
import albumentations as albu
import os
import pyrealsense2 as rs
import open3d as o3d
import json
from cloth_unfolding.code.color3dmapper import ColorTo3DMapper
from PIL import Image
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from cloth_unfolding.cloths_segmentation.cloths_segmentation.pre_trained_models import create_model
from pathlib import Path
from airo_typing import (
    CameraExtrinsicMatrixType,
    CameraIntrinsicsMatrixType,
    NumpyIntImageType,
    PointCloud,
)

# from grasp_pose_from_normal import convert_to_o3d_pcd

def convert_to_o3d_pcd(pcd: PointCloud):
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.points)
    o3d_pcd.colors = o3d.utility.Vector3dVector(pcd.colors / 255.0)

    return o3d_pcd


def segmentation(image: NumpyIntImageType, output_dir: Path):
    model = create_model("Unet_2020-10-30")
    model.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_image_path = output_dir / "segmentation.png"

    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)

    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

    with torch.no_grad():
        prediction = model(x)[0][0]

    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)

    rgb_image = np.full((mask.shape[0], mask.shape[1], 3), [128, 0, 128], dtype=np.uint8)
    rgb_image[mask == 1] = [255, 255, 0]

    pil_image = Image.fromarray(rgb_image)
    pil_image.save(output_image_path)

    return mask


def contour(binary_image: NumpyIntImageType, output_dir: Path, debug: bool = False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    height = binary_image.shape[0]
    width = binary_image.shape[1]

    image = np.zeros((height, width), dtype=binary_image.dtype)

    min_width = int(width * 0.2)
    max_width = int(width * 0.8)

    # 좌우 자르기 (모니터가 옷으로 검출되는 경우가 있음 dataset0/sample65)
    # image[:, min_width:max_width] = (binary_image * 255).astype(np.uint8)[:, min_width:max_width]
    image = (binary_image * 255).astype(np.uint8)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_box_area = [cv2.contourArea(contour) for contour in contours]
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    max_area_index = np.argmax(bounding_box_area)
    largest_bbox_coordinates = bounding_boxes[max_area_index]

    if debug:
        print("Bounding boxes:", bounding_box_area)
        print("Bounding boxes:", bounding_boxes)
        print("largest_bbox_coordinates:", largest_bbox_coordinates)

    image_with_bbox = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    x, y, w, h = largest_bbox_coordinates
    cv2.rectangle(image_with_bbox, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imwrite(str(output_dir / "original_with_bbox.jpg"), image_with_bbox)

    return largest_bbox_coordinates


def crop_pointcloud(bbox_coordinates, depth_image_path, intrinsic_path, input_ply_path, output_ply_path):
    x, y, w, h = bbox_coordinates
    points = [[x, y], [x+w, y], [x, y+h], [x+w, y+h]]

    depth_image_head = Image.open(depth_image_path).convert("L")
    depth_array = np.array(depth_image_head) / 255.
    image_width, image_height = depth_image_head.size

    if contour is not None:
        contour_pixel_points = [(c, r, depth_array[r][c]) for r in range(image_height) for c in range(image_width) if
                                cv2.pointPolygonTest(contour, (c, r), measureDist=False) == 1]

    # save mask countour to an image###########
    points = np.array(contour_pixel_points)

    img = cv2.imread(depth_image_path, 1)
    for item in points:
        cv2.drawMarker(img, (int(item[0]), int(item[1])), (0, 0, 255), markerType=cv2.MARKER_STAR,
                        markerSize=40, thickness=2, line_type=cv2.LINE_AA)
    cv2.imwrite("/home/minseo/contour.png", img)
    contour_world_points = []

    with open(intrinsic_path, 'r') as f:
        data = json.load(f)

    intrinsics = rs.intrinsics()
    intrinsics.width = data["image_resolution"]["width"]
    intrinsics.height = data["image_resolution"]["height"]
    intrinsics.ppx = data["principal_point_in_pixels"]["cx"]
    intrinsics.ppy = data["principal_point_in_pixels"]["cy"]
    intrinsics.fx = data["focal_lengths_in_pixels"]["fx"]
    intrinsics.fy = data["focal_lengths_in_pixels"]["fy"]
    intrinsics.model = rs.distortion.brown_conrady
    # elif cameraInfo.distortion_model == 'equidistant':
    #     self.intrinsics.model = rs.distortion.kannala_brandt4

    for point in contour_pixel_points:
        world_point = rs.rs2_deproject_pixel_to_point(intrinsics, [int(point[0]), int(point[1])],
                                                      depth_array[int(point[1]), int(point[0])])
        contour_world_points.append(world_point)
    world_points = np.array(contour_world_points)
    box_points = o3d.geometry.PointCloud()
    box_points.points = o3d.utility.Vector3dVector(world_points)
    # box_points.colors = highlight_pnts.vertex_colors
    # box_points.normals = highlight_pnts.vertex_normals
    box = box_points.get_axis_aligned_bounding_box()

    pcd = o3d.io.read_point_cloud(input_ply_path)
    cropped = pcd.crop(box)
    o3d.io.write_point_cloud(output_ply_path, cropped)

def camera_to_world(camera_extrinsics:CameraExtrinsicMatrixType, point):
    extrinsics_array = np.asarray(camera_extrinsics)

    x = extrinsics_array[0][3]
    y = extrinsics_array[1][3]
    z = extrinsics_array[2][3]

    # Create translation matrix
    p = np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])

    # Combine rotation and translation to get transformation matrix
    R = extrinsics_array[0:3, 0:3]
    T = np.dot(p, np.vstack([ np.hstack([R, np.zeros((3, 1))]), [0, 0, 0, 1] ]))

    point_camera_frame = np.array([point[0], point[1], point[2], 1])  # Homogeneous coordinates
    point_world_frame = np.dot(T, point_camera_frame)
    x_world = point_world_frame[0]
    y_world = point_world_frame[1]
    z_world = point_world_frame[2]

    return [x_world, y_world, z_world]


def world_to_camera(camera_extrinsics: CameraExtrinsicMatrixType, point):
    extrinsics_array = np.asarray(camera_extrinsics)

    x = extrinsics_array[0][3]
    y = extrinsics_array[1][3]
    z = extrinsics_array[2][3]

    # Create translation matrix
    p = np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])

    # Combine rotation and translation to get transformation matrix
    R = extrinsics_array[0:3, 0:3]
    T = np.dot(p, np.vstack([np.hstack([R, np.zeros((3, 1))]), [0, 0, 0, 1]]))

    # Compute the inverse transformation (world to camera)
    T_inv = np.linalg.inv(T)

    # Convert input point to homogeneous coordinates
    point_world = np.array([point[0], point[1], point[2], 1])

    # Apply inverse transformation
    point_camera = np.dot(T_inv, point_world)

    return [point_camera[0], point_camera[1], point_camera[2]]


def get_camera_axes_in_world(camera_extrinsics: CameraExtrinsicMatrixType):
    extrinsics_array = np.asarray(camera_extrinsics)

    # Extract rotation matrix (3x3)
    R = extrinsics_array[0:3, 0:3]

    # Each column of R is a camera axis expressed in world frame
    x_axis_world = R[:, 0]  # Camera x-axis in world frame
    y_axis_world = R[:, 1]  # Camera y-axis in world frame
    z_axis_world = R[:, 2]  # Camera z-axis in world frame

    return {
        "x": x_axis_world,
        "y": y_axis_world,
        "z": z_axis_world
    }


# high: 커질수록 위쪽, width: 작아질수록 오른쪽
def crop_condition(points, high, width):
    z = high * np.max(points[:, 2]) + (1 - high) * np.min(points[:, 2])
    y = width * np.max(points[:, 1]) + (1 - width) * np.min(points[:, 1])

    y_condition = (points[:, 1] >= y)
    z_condition = (points[:, 2] >= z)

    return np.where(y_condition & z_condition)[0]


# 1. crop with segmented bbox
# 2. crop robot arm
# 3. remove outlier
def crop(bbox_coordinates: list, rgb_image_path:str, depth_image_path: str,
         camera_intrinsics: CameraIntrinsicsMatrixType,
         point_cloud: PointCloud, output_dir: Path, debug = False):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Instantiate mapper and project to 3D
    depth_K = {"fx": 394.7567, "fy": 394.7567, "cx": 321.3488, "cy": 240.4375}
    color_K = {"fx": 385.767, "fy": 385.089, "cx": 327.746, "cy": 244.966}
    R = np.array([
        [0.9999993, -0.0011171, -0.0002352],
        [0.0011170, 0.9999993, -0.0004513],
        [0.0002357, 0.0004511, 0.9999999]
    ])
    t = np.array([-0.05923, -0.000169, 0.0002484])

    mapper = ColorTo3DMapper(depth_image_path, rgb_image_path, depth_K, color_K, R, t)

    def crop_with_bbox(pcd: o3d.geometry.PointCloud)-> o3d.geometry.PointCloud:
        x, y, w, h = bbox_coordinates
        x_min, x_max = int(x), int(x + w)
        y_min, y_max = int(y), int(y + h)

        # bbox 안에 있는 projected_uvs의 index 추출
        mask = (
                (mapper.projected_uvs[:, 0] >= x_min) &
                (mapper.projected_uvs[:, 0] < x_max) &
                (mapper.projected_uvs[:, 1] >= y_min) &
                (mapper.projected_uvs[:, 1] < y_max)
        )

        points_in_bbox = mapper.points[mask]

        if points_in_bbox.shape[0] == 0:
            print("No valid points in bbox")
            return pcd

        # axis-aligned bounding box 계산
        min_bound = points_in_bbox.min(axis=0)
        max_bound = points_in_bbox.max(axis=0)

        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
        cropped_pcd = pcd.crop(bbox)
        return cropped_pcd


    def crop_robot_arm(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        # high: 커질수록 위쪽, width: 작아질수록 오른쪽
        robot_indicies_1 = crop_condition(points, 0, 0.9)
        if debug:
            colors[robot_indicies_1] = [1, 0, 0]
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([pcd])

        robot_indicies_2 = crop_condition(points, 0.3, 0.7)
        if debug:
            colors[robot_indicies_2] = [1, 1, 0]
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([pcd])

        robot_indicies_3 = crop_condition(points, 0.5, 0.5)
        if debug:
            colors[robot_indicies_3] = [0, 0, 1]
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([pcd])

        robot_indicies_4 = crop_condition(points, 0.8, 0.3)
        if debug:
            colors[robot_indicies_4] = [1, 0, 1]
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([pcd])

        # 모든 robot_indices 배열을 결합하여 색상이 변경된 모든 포인트의 인덱스 배열 생성
        combined_robot_indices = np.concatenate([robot_indicies_1, robot_indicies_2, robot_indicies_3, robot_indicies_4])

        exclude_robot_points = np.delete(points, combined_robot_indices, axis=0)
        exclude_robot_colors = np.delete(colors, combined_robot_indices, axis=0)

        pcd.points = o3d.utility.Vector3dVector(exclude_robot_points)
        pcd.colors = o3d.utility.Vector3dVector(exclude_robot_colors)

        if debug:
            o3d.visualization.draw_geometries([pcd])

        return pcd


    # 가장자리 조금 자르기 remove outlier
    def cut_outlier(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        iqr_multiplier = 0.5 # 낮출수록 많이 잘려나감

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        # 열별로 IQR를 계산합니다.
        q1 = np.percentile(points, 25, axis=0)
        q3 = np.percentile(points, 75, axis=0)
        iqr = q3 - q1

        # 각 열에 대한 하한과 상한을 계산합니다.
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        # 각 열에서 아웃라이어를 식별합니다.
        outlier_mask = (points < lower_bound) | (points > upper_bound)

        # 행 중에서 모든 열이 아웃라이어가 아닌 경우를 찾아 데이터에서 제거합니다.
        points = points[~np.any(outlier_mask, axis=1)]
        colors = colors[~np.any(outlier_mask, axis=1)]

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        if debug:
            o3d.visualization.draw_geometries([pcd], window_name="cut outlier")

        return pcd

    pcd = convert_to_o3d_pcd(point_cloud)

    crop1_pcd = crop_with_bbox(pcd)
    o3d.io.write_point_cloud(str(output_dir / "crop1.ply"), crop1_pcd)

    # crop2_pcd = crop_robot_arm(crop1_pcd)
    # o3d.io.write_point_cloud(str(output_dir / "crop2.ply"), crop1_pcd)

    crop3_pcd = cut_outlier(crop1_pcd)
    o3d.io.write_point_cloud(str(output_dir / "crop3.ply"), crop3_pcd)

    return crop3_pcd

def remove_table(pcd: o3d.geometry.PointCloud, output_dir: Path, distance_threshold=0.01, debug=False):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    table_cloud = pcd.select_by_index(inliers)
    table_cloud.paint_uniform_color([1.0, 0, 0])
    object_cloud = pcd.select_by_index(inliers, invert=True)

    o3d.io.write_point_cloud(str(output_dir / "without_table.ply"), object_cloud)

    if debug:
        o3d.visualization.draw_geometries([table_cloud, object_cloud], window_name="Plane Segmentation")

    return object_cloud

# if __name__ == '__main__':
#     # model = create_model("Unet_2020-10-30")
#     # model.eval()
#     #
#     # for i in range(377):
#     # # for i in [1]:
#     #     print(f"{i}/377")
#     #     # root_dir = f"./cloth_competition_dataset_0000/sample_{'{0:06d}'.format(i)}/"
#     #     root_dir = f"./cloth_competition_dataset_0001/sample_{'{0:06d}'.format(i)}/"
#     #     input_img_path = root_dir + "observation_start/image_left.png"
#     #     output_dir = root_dir + "detected_edge"
#     #     output_img_path = output_dir + f"/segmentation_{'{0:06d}'.format(i)}.png"
#     #     if not os.path.exists(output_dir):
#     #         os.makedirs(output_dir)
#     #     segmentation(model, input_img_path, output_img_path)
#     root_path = "/home/minseo/cloth_competition_dataset_0001/sample_000100/"
#     depth_image_path = root_path + f"observation_start/depth_image.jpg"
#     intrinsic_path = root_path + f"observation_start/camera_intrinsics.json"
#
#     mask, output_dir = segmentation(root_path)
#     input_ply_path = root_path + f"observation_start/point_cloud.ply"
#     output_ply_path = output_dir + "/crop.ply"
#
#     largest_bbox_coordinates, contour = contour(mask, output_dir)
#     camera_pose_filename = root_path + "observation_start/camera_pose_in_world.json"
#     front_vector, look_at_vector, up_vector = cal_camera_vec.cal_camera_vec_from_json(camera_pose_filename)
#     look_at_vector[0] += 2
#     crop(largest_bbox_coordinates, contour, depth_image_path, intrinsic_path,camera_pose_filename, input_ply_path, output_ply_path, front_vector, look_at_vector, up_vector)
#
