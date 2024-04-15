import scipy.io
import numpy as np
from plyfile import PlyData

import open3d as o3d

def ply_to_mat(ply_filename, mat_filename):
    # PLY 파일을 읽기
    # ply_data = PlyData.read(ply_filename)

    # 필요한 데이터 추출
    pcd = o3d.io.read_point_cloud(ply_filename)

    # vertices = np.array([[vertex[0], vertex[1], vertex[2]] for vertex in ply_data['vertex'].data])
    # faces = np.array([[face[0], face[1], face[2]] for face in ply_data['face'].data['vertex_indices']])

    # MAT 파일로 저장
    scipy.io.savemat(mat_filename, {'vertices': pcd.points})

# 변환할 PLY 파일과 저장할 MAT 파일 이름 지정
ply_filename = './extracted_pcd.ply'
mat_filename = './extracted_pcd.mat'

# 변환 함수 호출
ply_to_mat(ply_filename, mat_filename)