
import open3d as o3d
import numpy as np
import cal_camera_vec

root_path = "/home/minseo/cloth_competition_dataset_0001/sample_0001000/"

ply_filepath = root_path + "detected_edge/edges.ply"
pcd = o3d.io.read_point_cloud(ply_filepath)

print(pcd)
print(np.asarray(pcd.points))

camera_pose_filename = root_path + "observation_start/camera_pose_in_world.json"
front_vector, look_at_vector, up_vector = cal_camera_vec.cal_camera_vec_from_json(camera_pose_filename)
look_at_vector[0] += 2
o3d.visualization.draw_geometries([pcd], zoom=0.1, front=front_vector, lookat=look_at_vector, up=up_vector,)

