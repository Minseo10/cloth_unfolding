import os
import open3d as o3d
import numpy as np
import cal_camera_vec
import pyrealsense2 as rs

# position_in_meters =  {
#     "x": -1.3044313379641588,
#     "y": 0.02674518353635602,
#     "z": 0.9266801808276479
# }
# rotation_euler_xyz_in_radians = {
#     "roll": -1.9760258764979506,
#     "pitch": 0.003413526105369158,
#     "yaw": -1.5848662024427904
# }
# front_vector, look_at_vector, up_vector = cal_camera_vec.cal_camera_vec(position_in_meters, rotation_euler_xyz_in_radians)

def bbox_to_3d(bbox_coordinates):
    x, y, w, h = bbox_coordinates
    points = [[x, y], [x+w, y], [x, y+h], [x+w, y+h]]

    world_points = []

    for point in points:
        world_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [int(point[0]), int(point[1])],
                                                      depth_frame.get_distance(int(point[0]), int(point[1])))
        if world_point[2] == 0.0:
            continue
        world_points.append(world_point)

    print("3D Bounding box points: \n", world_points)


def crop(input_ply_path, output_ply_path, front_vector, look_at_vector, up_vector):
    pcd = o3d.io.read_point_cloud(input_ply_path)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-0.5, -0.3, 0.1), max_bound=(0.5, 0.2, 3))
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
    # for i in range(377):
    for i in [0]:
        # print(f"{i}/377")

        # root_dir = f"./cloth_competition_dataset_0001/sample_{'{0:06d}'.format(i)}/"
        root_dir = f"/home/hjeong/code/unfolding_cloth/datasets/temp/"

        # input_ply_path = root_dir + "observation_start/point_cloud.ply"
        input_ply_path = root_dir + "point_cloud.ply"

        # output_dir = root_dir + "detected_edge"
        # output_ply_path = output_dir + f"/cropped_point_cloud_{'{0:06d}'.format(i)}.ply"
        output_dir = root_dir
        output_ply_path = root_dir + "crop.ply"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        crop(input_ply_path, output_ply_path)
