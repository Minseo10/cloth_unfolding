import open3d as o3d
from cal_camera_vec import cal_camera_vec

rgb = o3d.io.read_image('sample/sample_000000/observation_start/image_left.png')
d = o3d.io.read_image('sample/sample_000000/observation_start/depth_image.jpg')


image_resolution = {
    "width": 2208,
    "height": 1242
}
focal_lengths_in_pixels = {
    "fx": 1048.80224609375,
    "fy": 1048.80224609375
}
principal_point_in_pixels = {
    "cx": 1104.6346435546875,
    "cy": 621.6849975585938
}


intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width = int(image_resolution["width"]),
    height= int(image_resolution["height"]),
    fx = int(focal_lengths_in_pixels["fx"]),
    fy = int(focal_lengths_in_pixels["fy"]),
    cx = int(principal_point_in_pixels["cx"]),
    cy = int(principal_point_in_pixels["cy"])
)

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color = rgb,
    depth = d,
)
point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
    image = rgbd,
    intrinsic = intrinsic
)

position_in_meters = {"x": -1.3044313379641588, "y": 0.02674518353635602, "z": 0.9266801808276479}
rotation_euler_xyz_in_radians = {"roll": -1.9760258764979506, "pitch": 0.003413526105369158, "yaw": -1.5848662024427904}

front_vector, look_at_vector, up_vector = cal_camera_vec(position_in_meters, rotation_euler_xyz_in_radians)


o3d.visualization.draw_geometries([point_cloud],
            )
