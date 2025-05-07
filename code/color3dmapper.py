import numpy as np
import cv2
import open3d as o3d

class ColorTo3DMapper:
    def __init__(self, depth_path, rgb_path, depth_K, color_K, R, t, depth_scale=1000.0, max_depth=2.5):
        self.depth_K = depth_K
        self.color_K = color_K
        self.R = R
        self.t = t
        self.depth_scale = depth_scale
        self.max_depth = max_depth

        self.rgb = cv2.imread(rgb_path)
        self.depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        self.height, self.width = self.depth.shape
        self.points, self.colors, self.projected_uvs = self._generate_point_cloud()

    def _generate_point_cloud(self):
        points = []
        colors = []
        projected_uvs = []

        for v in range(self.height):
            for u in range(self.width):
                d = self.depth[v, u] / self.depth_scale
                if d == 0 or d > self.max_depth:
                    continue

                # Depth → Depth 카메라 좌표계
                xd = (u - self.depth_K['cx']) * d / self.depth_K['fx']
                yd = (v - self.depth_K['cy']) * d / self.depth_K['fy']
                zd = d
                p_depth = np.array([xd, yd, zd])

                # Depth → Color 좌표계로 변환
                p_color = self.R @ p_depth + self.t

                # Color 좌표계에서 2D 픽셀 좌표로 투영
                u_c = int((p_color[0] * self.color_K['fx']) / p_color[2] + self.color_K['cx'])
                v_c = int((p_color[1] * self.color_K['fy']) / p_color[2] + self.color_K['cy'])

                if 0 <= u_c < self.rgb.shape[1] and 0 <= v_c < self.rgb.shape[0]:
                    points.append(p_color)
                    colors.append(self.rgb[v_c, u_c, :] / 255.0)
                    projected_uvs.append([u_c, v_c])

        return np.array(points), np.array(colors), np.array(projected_uvs)

    def get_3d_point(self, uv_color):
        """color image 상의 (u, v) 좌표 → 가장 가까운 3D point 반환"""
        uv_color = np.asarray(uv_color).reshape(1, 2)
        dists = np.linalg.norm(self.projected_uvs - uv_color, axis=1)
        idx = np.argmin(dists)
        return self.points[idx], self.colors[idx]

    def get_open3d_pointcloud(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(self.colors)
        return pcd