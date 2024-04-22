import open3d as o3d
import numpy as np
import cal_camera_vec
import Difference_Eigenvalues as de
import crop_pointcloud
import segment_crop as seg
import json
import os

MIN = -4
MAX = 4


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

    return actual_grasp_point, R

def rotation_matrix_to_rpy(rotation_matrix):
    # Extract roll (x-axis rotation)
    roll = np.arctan2(rotation_matrix[2][1], rotation_matrix[2][2])

    # Extract pitch (y-axis rotation)
    pitch = np.arctan2(-rotation_matrix[2][0], np.sqrt(rotation_matrix[2][1]**2 + rotation_matrix[2][2]**2))

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

    print("Best point: ", point, "\n")
    print("Normal vector: ", normal, "\n")

    return point, normal


# sharp edge 위의 점들 중 x 축으로 가장 톡 튀어 나와있는 점을 best point 라고 찾고, 그 점에서의 normal vector 예측값을 grasp direction 으로 찾는 방식
def find_best_point_and_normal_vector_2(origin_pcd, edge_pcd, output_path):
    edge_points = np.asarray(edge_pcd.points)
    point_idx = np.argmin(edge_points[:, 0]) # idx in edge points array
    best_point = edge_points[point_idx]

    # check min x point with blue color
    point_colors = np.asarray(edge_pcd.colors)

    # check min x point with blue color
    tolerance = 0.01
    x_condition = (edge_points[:, 0] >= best_point[0] - tolerance) & (edge_points[:, 0] <= best_point[0] + tolerance)
    y_condition = (edge_points[:, 1] >= best_point[1] - tolerance) & (edge_points[:, 1] <= best_point[1] + tolerance)
    z_condition = (edge_points[:, 2] >= best_point[2] - tolerance) & (edge_points[:, 2] <= best_point[2] + tolerance)
    matching_indices = np.where(x_condition & y_condition & z_condition)[0]

    point_colors[:] = [1, 0, 0]
    point_colors[matching_indices] = [0, 0, 1]
    edge_pcd.colors = o3d.utility.Vector3dVector(point_colors)
    o3d.visualization.draw_geometries([edge_pcd])

    if output_path:
        o3d.io.write_point_cloud(output_path, edge_pcd)

    distances = np.linalg.norm(origin_pcd.points - best_point, axis=1)
    min_distance_index = np.argmin(distances)
    normal = np.asarray(origin_pcd.normals)[min_distance_index]

    print("Best point: ", best_point, "\n")
    print("Normal vector: ", normal, "\n")

    return best_point, normal


# 가장 뾰족한 점은 max sigma (red color value) 를 갖고 있고, 상당히 앞에 나와 있는 점들 중 가장 뾰족한 점을 best point 로 선정, grasp direction 은 normal vector
def find_best_point_and_normal_vector_3(origin_pcd, edge_pcd, output_path):
    edge_points = np.asarray(edge_pcd.points)
    colors = np.asarray(edge_pcd.colors)

    condition1 = edge_points[:, 0] < np.min(edge_points[:, 0]) + 0.05
    condition2 = edge_points[:, 1] < np.max(edge_points[:, 1]) - 0.05
    condition = condition1 & condition2
    colors[~condition] = [0, 0, 0]
    point_idx = np.argmax(colors[:, 0])
    best_point = edge_points[point_idx]

    # check min x point with blue color
    tolerance = 0.01
    x_condition = (edge_points[:, 0] >= best_point[0] - tolerance) & (edge_points[:, 0] <= best_point[0] + tolerance)
    y_condition = (edge_points[:, 1] >= best_point[1] - tolerance) & (edge_points[:, 1] <= best_point[1] + tolerance)
    z_condition = (edge_points[:, 2] >= best_point[2] - tolerance) & (edge_points[:, 2] <= best_point[2] + tolerance)
    matching_indices = np.where(x_condition & y_condition & z_condition)[0]

    colors[:] = [1, 0, 0]
    colors[matching_indices] = [0, 0, 1]
    edge_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([edge_pcd])

    if output_path:
        o3d.io.write_point_cloud(output_path, edge_pcd)

    distances = np.linalg.norm(origin_pcd.points - best_point, axis=1)
    min_distance_index = np.argmin(distances)
    normal = np.asarray(origin_pcd.normals)[min_distance_index]

    print("Best point: ", point, "\n")
    print("Normal vector: ", normal, "\n")

    return best_point, normal


if __name__ == '__main__':
    root_path = "/home/minseo/cloth_competition_dataset_0001/sample_000002/"
    root_path = "../datasets/cloth_competition_dataset_0001/sample_000002/"

    # Example rotation matrix
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(np.pi/4), -np.sin(np.pi/4)],
        [0, np.sin(np.pi/4), np.cos(np.pi/4)]
    ])


    # Calculate roll, pitch, yaw
    roll, pitch, yaw = rotation_matrix_to_rpy(rotation_matrix)

    # camera info
    camera_pose_filename = root_path + "observation_start/camera_pose_in_world.json"
    front_vector, look_at_vector, up_vector = cal_camera_vec.cal_camera_vec_from_json(camera_pose_filename)
    look_at_vector[0] += 2


    # cloth segmentation
    mask, output_dir = seg.segmentation(root_path)
    largest_bbox_coordinates, contour = seg.contour(mask, output_dir)

    # crop pointcloud.ply along segmentation
    input_ply_path = root_path + "observation_start/point_cloud.ply"
    pcd_dir = root_path + "detected_edge/"
    depth_image_path = root_path + f"observation_start/depth_image.jpg"
    intrinsic_path = root_path + f"observation_start/camera_intrinsics.json"
    pcd_filepath = pcd_dir + "crop.ply"
    if not os.path.exists(pcd_dir):
        os.makedirs(pcd_dir)
    seg.crop(largest_bbox_coordinates, contour, depth_image_path, intrinsic_path, camera_pose_filename, input_ply_path, pcd_filepath, front_vector, look_at_vector, up_vector)


    # edge extraction
    pcd = o3d.io.read_point_cloud(pcd_filepath)
    edge_output_dir = root_path + "detected_edge/"
    de.extract_edge(pcd_filepath, edge_output_dir)  # from Difference_Eigenvalues.py

    # estimate normal vectors
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    # find corresponding point in point cloud
    edge_pcd_filepath = edge_output_dir + de.EDGE_FILENAME
    edge_pcd = o3d.io.read_point_cloud(edge_pcd_filepath) # TODO:: delete

    # point, normal = find_best_point_and_normal_vector(root_path, pcd, edge_pcd)
    point, normal = find_best_point_and_normal_vector_2(pcd, edge_pcd, edge_output_dir + "ftn2.ply")
    point, normal = find_best_point_and_normal_vector_3(pcd, edge_pcd, edge_output_dir + "ftn3.ply")

    # normal vector 옷 바깥쪽으로 뒤집기
    pcd.orient_normals_consistent_tangent_plane(k=15)


    # grasp direction (vector) -> grasp pose (rpy)
    # Calculate world frame's z-axis in camera frame
    z_axis_world = np.array([0, 0, 1])
    grasp_x = np.cross(z_axis_world, normal)
    grasp_x /= np.linalg.norm(grasp_x)
    grasp_x = (-1) * grasp_x
    grasp_y = z_axis_world
    grasp_y = (-1) * grasp_y
    grasp_z = normal
    grasp_z /= np.linalg.norm(normal)
    grasp_z = (-1) * grasp_z

    # world frame에 대한 grasp pose
    grasp_R = [[grasp_x[0], grasp_y[0], grasp_z[0]], [grasp_x[1], grasp_y[1], grasp_z[1]], [grasp_x[2], grasp_y[2], grasp_z[2]]]
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # grasp_R = mesh.get_rotation_matrix_from_xyz((roll, pitch, yaw)) # grasp pose w.r.t world frame
    mesh.rotate(grasp_R, center=(0, 0, 0)) # change to best point
    # mesh.rotate([[1, 0, 0], [0,1,0], [0,0,1]], center=(0,0,0))

    # roll pitch yaw
    roll, pitch, yaw = rotation_matrix_to_rpy(grasp_R)
    print("Roll (radians):", roll)
    print("Pitch (radians):", pitch)
    print("Yaw (radians):", yaw)


    # visualize grasp pose at the grasp point
    start_point = point.tolist()
    end_point = (point + 0.1 * normal).tolist()

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([start_point, end_point]),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
    line_set_line_width = 4

    o3d.visualization.draw_geometries([pcd, line_set, mesh],
                                      zoom=0.1,
                                      front=front_vector,
                                      lookat=look_at_vector,
                                      up=up_vector,
                                      point_show_normal=False)
    #
    # o3d.visualization.draw_geometries([edge_pcd],
    #                                   zoom=0.1,
    #                                   front=front_vector,
    #                                   lookat=look_at_vector,
    #                                   up=up_vector,
    #                                   point_show_normal=True)
