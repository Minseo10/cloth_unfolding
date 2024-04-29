from cloth_tools.visualization.opencv import draw_pose
import cv2
import copy
import open3d as o3d
import matplotlib.pyplot as plt
import json
import numpy as np
from cloth_tools.dataset.format import load_competition_observation, CompetitionObservation

def read_6d_pose(json_path):
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

    T = np.eye(4)
    T[:3, :3] = R
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T

draw_image = False
open3d = True
observation_dir = "../datasets/downloaded_dataset_0000/sample_1/observation_start"
grasp_path = "data/sample_1/grasp_pose_2024-04-26_08-37-30-646753.json"
pcd_path = "../datasets/downloaded_dataset_0000/sample_1/processing/crop3.ply"

# read dry run data
sample = load_competition_observation(observation_dir)
image_bgr = cv2.cvtColor(sample.image_left, cv2.COLOR_RGB2BGR)
X_W_C = sample.camera_pose_in_world
intrinsics = sample.camera_intrinsics
grasp_pose_fixed = read_6d_pose(grasp_path)
pcd = o3d.io.read_point_cloud(pcd_path)

# draw pose in image
if draw_image:
    print(grasp_pose_fixed)
    print(intrinsics)
    draw_pose(image_bgr, grasp_pose_fixed, intrinsics, X_W_C, 0.1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))
    plt.imshow(image_rgb)
    plt.title("Example grasp pose")
    plt.savefig(grasp_path + "/../frontal_image_grasp.jpg")

# open 3d visualize
if open3d:
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(0.1, center=(0,0,0))
    mesh = copy.deepcopy(mesh).transform(grasp_pose_fixed)
    o3d.visualization.draw_geometries([pcd, mesh])
