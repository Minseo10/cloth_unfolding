from pathlib import Path
from pyntcloud import PyntCloud
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import open3d as o3d

PCD_EDGE_FILENAME = 'pointcloud_edges.ply'
EDGE_FILENAME = 'edges.ply'

def convert_to_pyntcloud(pcd: o3d.geometry.PointCloud.PointCloud):
    points_np = np.asarray(pcd.points)
    colors_np = np.asarray(pcd.colors)

    # pandas 데이터프레임 생성
    df = pd.DataFrame(data=np.hstack([points_np, colors_np]), columns=['x', 'y', 'z', 'red', 'green', 'blue'])

    # pyntcloud의 PyntCloud 객체 생성
    pyntcloud_pc = PyntCloud(df)

    return pyntcloud_pc

def convert_to_o3d_pcd(pcd: PyntCloud):
    points = pcd.points[['x', 'y', 'z']].values
    colors = pcd.points[['red', 'green', 'blue']].values

    # open3d PointCloud 객체 생성
    point_cloud = o3d.geometry.PointCloud()

    # points 설정
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # colors 설정
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud


def extract_edge(pcd: o3d.geometry.PointCloud, output_dir: Path, uniformed=False, k_n = 50, thresh = 0.03):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pcd_np = np.zeros((len(pcd.points), 6))

    pcd = convert_to_pyntcloud(pcd)
    # find neighbors
    kdtree_id = pcd.add_structure("kdtree")
    k_neighbors = pcd.get_neighbors(k=k_n, kdtree=kdtree_id)

    # calculate eigenvalues
    ev = pcd.add_scalar_field("eigen_values", k_neighbors=k_neighbors)

    x = pcd.points['x'].values
    y = pcd.points['y'].values
    z = pcd.points['z'].values

    e1 = pcd.points['e3('+str(k_n+1)+')'].values
    e2 = pcd.points['e2('+str(k_n+1)+')'].values
    e3 = pcd.points['e1('+str(k_n+1)+')'].values

    sum_eg = np.add(np.add(e1,e2),e3)
    sigma = np.divide(e1,sum_eg)
    sigma_value = sigma
    #pdb.set_trace()
    #img = ax.scatter(x, y, z, c=sigma, cmap='jet')

    # visualize the edges
    sigma = sigma>thresh

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Visualize each one of the eigenvalues
    #img = ax.scatter(x, y, z, c=e1, cmap='jet')
    #img = ax.scatter(x, y, z, c=e2, cmap='jet')
    #img = ax.scatter(x, y, z, c=e3, cmap='jet')

    # visualize the edges
    img = ax.scatter(x, y, z, c=sigma, cmap='jet')
    #img = ax.scatter(x, y, z, c=sigma, cmap=plt.hot())

    fig.colorbar(img)
    # plt.show()

    # Save the edges and point cloud
    thresh_min = sigma_value < thresh
    sigma_value[thresh_min] = 0
    if uniformed:
        thresh_max = sigma_value > thresh
        sigma_value[thresh_max] = 255
    else:
        min_val = sigma_value.min()
        max_val = sigma_value.max()
        normalized_arr = (sigma_value - min_val) / (max_val - min_val)
        sigma_value = (normalized_arr * 255).astype(np.uint8)

    pcd_np[:,0] = x
    pcd_np[:,1] = y
    pcd_np[:,2] = z
    pcd_np[:,3] = sigma_value

    edge_np = np.delete(pcd_np, np.where(pcd_np[:,3] == 0), axis=0)

    clmns = ['x','y','z','red','green','blue']
    pcd_pd = pd.DataFrame(data=pcd_np,columns=clmns)
    pcd_pd['red'] = sigma_value.astype(np.uint8)

    #pcd_points = PyntCloud(pd.DataFrame(data=pcd_np,columns=clmns))
    pcd_points = PyntCloud(pcd_pd)
    edge_points = PyntCloud(pd.DataFrame(data=edge_np,columns=clmns))

    # pcd_points.plot()
    # edge_points.plot()

    PyntCloud.to_file(pcd_points, str(output_dir / PCD_EDGE_FILENAME))   # Save the whole point cloud by painting the edge points
    PyntCloud.to_file(edge_points, str(output_dir / EDGE_FILENAME))     # Save just the edge points

    return convert_to_o3d_pcd(edge_points)
    # ply = PyntCloud.from_file(output_dir+'pointcloud_edges.ply')
    # ply.plot()