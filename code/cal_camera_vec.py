import numpy as np
import json

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

# 주어진 JSON 데이터에서 변수를 추출하는 함수
def extract_variables(data):
    position = data['position_in_meters']
    rotation = data['rotation_euler_xyz_in_radians']
    return position, rotation


# 회전 행렬 계산 함수
def euler_to_rotation_matrix(roll, pitch, yaw):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def default_camera_vec(position_in_meters):
    front_vector = [-1, 0, 0]
    up_vector = [0, 0, 1]
    look_at_vector = np.array([position_in_meters["x"], position_in_meters["y"], position_in_meters["z"]]) + front_vector

    return front_vector, look_at_vector, up_vector


def cal_camera_vec_from_json(json_path):
    data = load_json(json_path)
    position, rotation = extract_variables(data)

    R = euler_to_rotation_matrix(rotation["roll"], rotation["pitch"], rotation["yaw"])

    front_vector = np.dot(R, np.array([0, 0, -1]))
    up_vector = np.dot(R, np.array([0, -1, 0]))
    look_at_vector = np.array([position["x"], position["y"], position["z"]]) + front_vector

    # 결과 출력
    print("Look-at vector:", look_at_vector)
    print("Front vector:", front_vector)
    print("Up vector:", up_vector)

    return front_vector, look_at_vector, up_vector


def cal_camera_vec(position_in_meters, rotation_euler_xyz_in_radians):
    # 회전 행렬 생성
    R = euler_to_rotation_matrix(rotation_euler_xyz_in_radians["roll"],
                                 rotation_euler_xyz_in_radians["pitch"],
                                 rotation_euler_xyz_in_radians["yaw"])

    # 카메라 방향 벡터 계산
    front_vector = np.dot(R, np.array([0, 0, -1]))

    # 업 벡터 계산
    up_vector = np.dot(R, np.array([0, -1, 0]))

    look_at_vector = np.array([position_in_meters["x"], position_in_meters["y"], position_in_meters["z"]]) + front_vector
    # norm = np.linalg.norm(look_at_vector)
    # look_at_vector /= norm

    # # 카메라 위치와 바라보는 점
    # camera_position = np.array([position_in_meters["x"], position_in_meters["y"], position_in_meters["z"]])
    # look_at_point = camera_position + front_vector  # 카메라의 위치에서 바라보는 방향으로 떨어진 점
    #
    # # look-at 벡터 계산
    # look_at_vector = look_at_point - camera_position
    # look_at_vector /= np.linalg.norm(look_at_vector)  # 벡터 정규화

    # 결과 출력
    print("Look-at vector:", look_at_vector)
    print("Front vector:", front_vector)
    print("Up vector:", up_vector)

    return front_vector, look_at_vector, up_vector

