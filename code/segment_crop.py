import numpy as np
import cv2
import torch
import albumentations as albu
import os
from PIL import Image
import sys
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
import pyrealsense2 as rs
import open3d as o3d
import json
import cal_camera_vec
sys.path.append("/home/minseo/robot_ws/src")

from cloths_segmentation.cloths_segmentation.pre_trained_models import create_model


def segmentation(root_dir):
    model = create_model("Unet_2020-10-30")
    model.eval()
    input_image_path = root_dir + "observation_start/image_left.png"
    output_dir = root_dir + "detected_edge"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_image_path = output_dir + f"/segmentation.png"
    image = load_rgb(input_image_path)

    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)

    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

    with torch.no_grad():
        prediction = model(x)[0][0]

    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)
    mask_path = output_dir + f"/mask.npy"
    np.save(mask_path, mask)  # save mask npy file

    # imshow(mask)
    # rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    #
    # for i in range(mask.shape[0]):
    #     for j in range(mask.shape[1]):
    #         if mask[i, j] == 1:
    #             rgb_image[i, j] = [255, 255, 0]
    #         else:
    #             rgb_image[i, j] = [128, 0, 128]

    # rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # rgb_image[mask == 1] = [255, 255, 0]
    # rgb_image[mask != 1] = [128, 0, 128]

    rgb_image = np.full((mask.shape[0], mask.shape[1], 3), [128, 0, 128], dtype=np.uint8)
    rgb_image[mask == 1] = [255, 255, 0]

    pil_image = Image.fromarray(rgb_image)
    pil_image.save(output_image_path)

    return mask, output_dir

def contour(mask, output_dir):
    image_array = np.load(output_dir + f"/mask.npy")

    mask = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_box_area = [cv2.contourArea(contour) for contour in contours]
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    max_area_index = np.argmax(bounding_box_area)
    largest_bbox_coordinates = bounding_boxes[max_area_index]

    print("Bounding boxes:", bounding_box_area)
    print("Bounding boxes:", bounding_boxes)
    print("largest_bbox_coordinates:", largest_bbox_coordinates)

    image_with_bbox = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    x, y, w, h = largest_bbox_coordinates
    cv2.rectangle(image_with_bbox, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imwrite(output_dir + f"/original_with_bbox.jpg", image_with_bbox)

    return largest_bbox_coordinates, contours[max_area_index]


def crop_pointcloud(bbox_coordinates, contour, depth_image_path, intrinsic_path, input_ply_path, output_ply_path):
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

def camera_to_world(json_path, point):
    with open(json_path, 'r') as f:
        data = json.load(f)

    x = data["position_in_meters"]["x"]
    y = data["position_in_meters"]["y"]
    z = data["position_in_meters"]["z"]

    # Grasp direction
    roll, pitch, yaw = data["rotation_euler_xyz_in_radians"]["roll"], data["rotation_euler_xyz_in_radians"]["pitch"], data["rotation_euler_xyz_in_radians"]["yaw"]

    # Convert Euler angles to rotation matrix
    R = np.array([
        [np.cos(yaw) * np.cos(pitch), np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll),
         np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
        [np.sin(yaw) * np.cos(pitch), np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll),
         np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
        [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]
    ])
    # Create translation matrix
    p = np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])

    # Combine rotation and translation to get transformation matrix
    # np.vstack([ np.hstack([R, np.zeros((3, 1))]), [0, 0, 0, 1] ])
    T = np.dot(p, np.vstack([ np.hstack([R, np.zeros((3, 1))]), [0, 0, 0, 1] ]))

    point_camera_frame = np.array([point[0], point[1], point[2], 1])  # Homogeneous coordinates
    point_world_frame = np.dot(T, point_camera_frame)
    x_world = point_world_frame[0]
    y_world = point_world_frame[1]
    z_world = point_world_frame[2]

    return [x_world, y_world, z_world]

def crop(bbox_coordinates, contour, depth_image_path, intrinsic_path, extrinsic_path, input_ply_path, output_ply_path, front_vector, look_at_vector, up_vector):
    pcd = o3d.io.read_point_cloud(input_ply_path)

    x, y, w, h = bbox_coordinates
    # box_points = [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]
    center = [x + w/2, y + h/2]
    box_points = [[x + w/2, y + h/2]]

    depth_image_head = Image.open(depth_image_path).convert("L")
    depth_array = np.array(depth_image_head) / 255.
    image_width, image_height = depth_image_head.size

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

    box_points_world = []
    for point in box_points:
        point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [int(point[0]), int(point[1])],
                                                      depth_array[int(center[0]), int(center[1])])
        box_points_world.append(point_3d)

    box_3d = []
    max_coordinates = [-100, -100, -100]
    min_coordinates = [100, 100, 100]
    for point in box_points_world:
        new_point_1 = [point[0], point[1], point[2]]
        new_point_2 = [point[0], point[1], point[2]]

        # camera -> world frame
        new_point_1 = camera_to_world(extrinsic_path, new_point_1)
        new_point_2 = camera_to_world(extrinsic_path, new_point_2)

        box_3d.append(new_point_1)
        box_3d.append(new_point_2)

        for i in range(3):
            if new_point_1[i] > max_coordinates[i]:
                max_coordinates[i] = new_point_1[i]
            if new_point_1[i] < min_coordinates[i]:
                min_coordinates[i] = new_point_1[i]
            if new_point_2[i] > max_coordinates[i]:
                max_coordinates[i] = new_point_2[i]
            if new_point_2[i] < min_coordinates[i]:
                min_coordinates[i] = new_point_2[i]

    print("3D Bounding box points: \n", box_3d)
    print("Max Min bounds: \n", max_coordinates, "\n", min_coordinates)

    # bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-0.5, -0.3, -5), max_bound=(0.5, 0.2, 3))
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(min_coordinates[0]-0.5, min_coordinates[1]-0.3, min_coordinates[2]-0.6), max_bound=(max_coordinates[0]+1.0, max_coordinates[1]+0.3, max_coordinates[2]+0.5))
    # bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-0.5, -0.5, -3), max_bound=(0.5, 0.5, 0.9))
    # bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(min, min, min), max_bound=(max, 2, max))
    cropped = pcd.crop(bbox)

    # o3d.visualization.draw_geometries([cropped],
    #                                   zoom=0.1,
    #                               front=front_vector,
    #                               lookat=look_at_vector,
    #                               up=up_vector,
    #                               # width=2208,
    #                               # height=1242,
    #                               left=0,
    #                               top=0,
    #                               )

    o3d.io.write_point_cloud(output_ply_path, cropped)


if __name__ == '__main__':
    # model = create_model("Unet_2020-10-30")
    # model.eval()
    #
    # for i in range(377):
    # # for i in [1]:
    #     print(f"{i}/377")
    #     # root_dir = f"./cloth_competition_dataset_0000/sample_{'{0:06d}'.format(i)}/"
    #     root_dir = f"./cloth_competition_dataset_0001/sample_{'{0:06d}'.format(i)}/"
    #     input_img_path = root_dir + "observation_start/image_left.png"
    #     output_dir = root_dir + "detected_edge"
    #     output_img_path = output_dir + f"/segmentation_{'{0:06d}'.format(i)}.png"
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #     segmentation(model, input_img_path, output_img_path)
    root_path = "/home/minseo/cloth_competition_dataset_0001/sample_000100/"
    depth_image_path = root_path + f"observation_start/depth_image.jpg"
    intrinsic_path = root_path + f"observation_start/camera_intrinsics.json"

    mask, output_dir = segmentation(root_path)
    input_ply_path = root_path + f"observation_start/point_cloud.ply"
    output_ply_path = output_dir + "/crop.ply"

    largest_bbox_coordinates, contour = contour(mask, output_dir)
    camera_pose_filename = root_path + "observation_start/camera_pose_in_world.json"
    front_vector, look_at_vector, up_vector = cal_camera_vec.cal_camera_vec_from_json(camera_pose_filename)
    look_at_vector[0] += 2
    crop(largest_bbox_coordinates, contour, depth_image_path, intrinsic_path,camera_pose_filename, input_ply_path, output_ply_path, front_vector, look_at_vector, up_vector)

