import open3d as o3d
import numpy as np
import cal_camera_vec
import Difference_Eigenvalues as de
import crop_pcd
import json
import os

root_path = "/home/minseo/cc_dataset/sample_000003/"
min = -4
max = 4


def calculate_grasp_point(json_path):
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

    # Grasp direction vector
    grasp_dir_vec = np.dot(R, np.array([0, 0, 1]))
    distance = 0.05

    # Calculate actual grasp point by moving grasp point along grasp direction by given distance
    actual_grasp_point = np.array([x, y, z]) - distance * grasp_dir_vec

    return actual_grasp_point


# camera info
camera_pose_filename = root_path + "observation_start/camera_pose_in_world.json"
front_vector, look_at_vector, up_vector = cal_camera_vec.cal_camera_vec_from_json(camera_pose_filename)
look_at_vector[0] += 2


# cloth segmentation



# crop pointcloud.ply along segmentation
# 일단은 임의로 crop
input_ply_path = root_path + "observation_start/point_cloud.ply"
pcd_dir = root_path + "detected_edge/"
pcd_filepath = pcd_dir + "cropped_pcd.ply"
if not os.path.exists(pcd_dir):
    os.makedirs(pcd_dir)
crop_pcd.crop(input_ply_path, pcd_filepath, front_vector, look_at_vector, up_vector)


# edge extraction
pcd = o3d.io.read_point_cloud(pcd_filepath)
edge_filepath = root_path + "detected_edge/edges.ply"
de.extract_edge(pcd_filepath, edge_filepath)  # from Difference_Eigenvalues.py
edge_pcd = o3d.io.read_point_cloud(edge_filepath)
edge_points = np.asarray(edge_pcd.points)
print(edge_points)
nearest_point = edge_points[edge_points[:, 0] < -0.12883484]


# estimate normal vectors
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
)


# best point selection
# 현재는 bring real grasp point from grasp_pose.json
grasp_pose_filepath = root_path + "grasp/grasp_pose.json"
best_point = calculate_grasp_point(grasp_pose_filepath)


# find corresponding point in point cloud
points_array = np.asarray(pcd.points)
normals_array = np.asarray(pcd.normals)
min_dist = 1000000
best_point_idx = 0

for i in range(points_array.shape[0]):
    dist = np.linalg.norm(points_array[i] - best_point)
    if dist < min_dist:
        min_dist = dist
        best_point_idx = i
point = points_array[best_point_idx]
normal = normals_array[best_point_idx]
print("Best point: ", point, "\n")
print("Normal vector: ", normal, "\n")


# normal vector 옷 바깥쪽으로 뒤집기
pcd.orient_normals_consistent_tangent_plane(k=15)


# grasp direction (vector) -> grasp pose (rpy)


# visualize grasp pose at the grasp point
start_point = point.tolist()
end_point = (point + 0.1 * normal).tolist()

line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector([start_point, end_point]),
    lines=o3d.utility.Vector2iVector([[0, 1]]),
)
line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
line_set_line_width = 1

o3d.visualization.draw_geometries([pcd, line_set],
                                  zoom=0.1,
                                  front=front_vector,
                                  lookat=look_at_vector,
                                  up=up_vector,
                                  point_show_normal=False)
