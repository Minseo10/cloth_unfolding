import open3d as o3d
import numpy as np
import cal_camera_vec


root_path = "/home/minseo/cc_dataset/sample_000002/"

camera_pose_filename = root_path + "observation_start/camera_pose_in_world.json"
front_vector, look_at_vector, up_vector = cal_camera_vec.cal_camera_vec_from_json(camera_pose_filename)
look_at_vector[0] += 1

pcd_filepath = root_path + "detected_edge/extracted_pcd.ply"
pcd = o3d.io.read_point_cloud(pcd_filepath)

edge_filepath = root_path + "detected_edge/edges.ply"
edge_pcd = o3d.io.read_point_cloud(edge_filepath)

points = np.asarray(edge_pcd.points)
print(points)
nearest_point = points[points[:, 0] < -0.12883484]

o3d.visualization.draw_geometries([edge_pcd],
                                  zoom=0.1,
                                  front=front_vector,
                                  lookat=look_at_vector,
                                  up=up_vector,
                                  point_show_normal=False)

o3d.visualization.draw_geometries([pcd],
                                  zoom=0.1,
                                  front=front_vector,
                                  lookat=look_at_vector,
                                  up=up_vector,
                                  point_show_normal=True)


pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
)
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.1,
                                  front=front_vector,
                                  lookat=look_at_vector,
                                  up=up_vector,
                                  point_show_normal=True)
