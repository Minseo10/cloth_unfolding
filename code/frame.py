import open3d as o3d
import copy
import numpy as np

# mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
# mesh_tx = copy.deepcopy(mesh).translate((1.3, 0, 0))
# mesh_ty = copy.deepcopy(mesh).translate((0, 1.3, 0))
# print(f'Center of mesh: {mesh.get_center()}')
# print(f'Center of mesh tx: {mesh_tx.get_center()}')
# print(f'Center of mesh ty: {mesh_ty.get_center()}')
# o3d.visualization.draw_geometries([mesh, mesh_tx, mesh_ty])



# mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
# mesh_mv = copy.deepcopy(mesh).translate((2, 2, 2), relative=False)
# print(f'Center of mesh: {mesh.get_center()}')
# print(f'Center of translated mesh: {mesh_mv.get_center()}')
# o3d.visualization.draw_geometries([mesh, mesh_mv])



mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
# mesh_mv = copy.deepcopy(mesh).translate((0, 1, 0), relative=False)
# mesh_r = copy.deepcopy(mesh)
# R = mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, np.pi/2))
R = mesh.get_rotation_matrix_from_xyz((-1.9760258764979506, 0.003413526105369158, -1.5848662024427904))
mesh.rotate(R, center=(0, 0, 0))
# o3d.visualization.draw_geometries([mesh, mesh_r])
o3d.visualization.draw_geometries([mesh])
