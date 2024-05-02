import os
from pathlib import Path
from cloth_tools.dataset.format import load_competition_observation
from cloth_tools.drake.scenes import make_drake_scene_from_observation
from functools import partial
import json
from cloth_tools.kinematics.constants import TCP_TRANSFORM
from cloth_tools.kinematics.inverse_kinematics import inverse_kinematics_in_world_fn
from cloth_tools.drake.scenes import make_dual_arm_collision_checker
from cloth_tools.kinematics.constants import JOINT_BOUNDS
from airo_planner import DualArmOmplPlanner
from cloth_tools.annotation.grasp_annotation import grasp_hanging_cloth_pose
import numpy as np
from cloth_tools.planning.grasp_planning import plan_pregrasp_and_grasp_trajectory
from airo_drake import animate_dual_joint_trajectory
from cloth_tools.planning.grasp_planning import ExhaustedOptionsError
import time
from airo_typing import (
    CameraIntrinsicsMatrixType,
    HomogeneousMatrixType,
    NumpyDepthMapType,
    NumpyIntImageType,
    OpenCVIntImageType,
    PointCloud,
    Vector3DType,
)


def read_6d_pose(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    x = data["position_in_meters"]["x"]
    y = data["position_in_meters"]["y"]
    z = data["position_in_meters"]["z"]

    # Grasp direction
    roll, pitch, yaw = data["rotation_euler_xyz_in_radians"]["roll"], data["rotation_euler_xyz_in_radians"]["pitch"], \
    data["rotation_euler_xyz_in_radians"]["yaw"]

    # Convert Euler angles to rotation matrix
    R = np.array([
        [np.cos(yaw) * np.cos(pitch), np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll),
         np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
        [np.sin(yaw) * np.cos(pitch), np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll),
         np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
        [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]
    ])

    T = np.eye(4)
    T[:3, :3] = R
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


def is_grasp_executable_fn(observation, grasp_pose) -> bool:
    scene = make_drake_scene_from_observation(observation, include_cloth_obstacle=False)
    scene_with_cloth = make_drake_scene_from_observation(observation, include_cloth_obstacle=True)

    X_W_LCB = observation.arm_left_pose_in_world
    X_W_RCB = observation.arm_right_pose_in_world

    inverse_kinematics_left_fn = partial(inverse_kinematics_in_world_fn, X_W_CB=X_W_LCB, tcp_transform=TCP_TRANSFORM)
    inverse_kinematics_right_fn = partial(inverse_kinematics_in_world_fn, X_W_CB=X_W_RCB, tcp_transform=TCP_TRANSFORM)

    collision_checker_no_cloth = make_dual_arm_collision_checker(scene)
    collision_checker_with_cloth = make_dual_arm_collision_checker(scene_with_cloth)

    planner_pregrasp = DualArmOmplPlanner(
        is_state_valid_fn=collision_checker_with_cloth.CheckConfigCollisionFree,
        inverse_kinematics_left_fn=inverse_kinematics_left_fn,
        inverse_kinematics_right_fn=inverse_kinematics_right_fn,
        joint_bounds_left=JOINT_BOUNDS,
        joint_bounds_right=JOINT_BOUNDS,
    )

    try:
        trajectory_pregrasp_and_grasp = plan_pregrasp_and_grasp_trajectory(
            planner_pregrasp,
            grasp_pose,
            observation.arm_left_joints,
            observation.arm_right_joints,
            inverse_kinematics_left_fn,
            inverse_kinematics_right_fn,
            collision_checker_no_cloth.CheckConfigCollisionFree,
            scene.robot_diagram.plant(),
            with_left=False,
        )
        animate_dual_joint_trajectory(
            scene_with_cloth.meshcat,
            scene_with_cloth.robot_diagram,
            scene_with_cloth.arm_left_index,
            scene_with_cloth.arm_right_index,
            trajectory_pregrasp_and_grasp,
        )

        print(f"You can see the trajectory animation at: {scene_with_cloth.meshcat.web_url()}")
    except ExhaustedOptionsError:
        return False

    return True

# modify grasp_hanging_cloth_pose
# rotate grasps
def rotate_grasps(
    position: Vector3DType, approach_direction: Vector3DType, rotate_ang: float, grasp_depth: float = 0.0
) -> HomogeneousMatrixType:
    """Create a pose for grasping a hanging cloth.

    Args:
        position: The position of the grasp.
        approach_direction: The direction the gripper should approach the grasp from.
        grasp_depth: An additional forward (= Z) offset for the grasp to bring it deeper into the cloth.

    Returns:
        pose: The pose for the grasp.
    """
    Z = approach_direction / np.linalg.norm(approach_direction)
    position_with_depth = position + grasp_depth * Z

    # Pointing gripper Y up or down leads to the gripper opening horizontally
    # I chose Y-down here because it's closer the other poses used in the controllers
    # Y = np.array([0, 0, -1])  # default Y
    Y = np.array([0, 0, 1])  # Trying the positive instead to attempt to reduce twisting issue

    # Handle rare case where Z is parallel to default Y
    if np.abs(np.dot(Y, Z)) > 0.99:
        Y = np.array([-1, 0, 0])

    X = np.cross(Y, Z)
    X = X / np.linalg.norm(X)  # Normalize X for the case where Y and Z were not perpendicular

    # Recalculate Y to be guaranteed perpendicular to X and Z
    Y = np.cross(Z, X)
    orientation = np.column_stack([X, Y, Z])

    pose = np.identity(4)
    pose[:3, :3] = orientation
    pose[:3, 3] = position_with_depth

    # rotate 180
    # z축 기준으로 rotate_ang만큼 회전하는 행렬 생성
    rotation_matrix = np.array([
        [np.cos(rotate_ang), -np.sin(rotate_ang), 0],
        [np.sin(rotate_ang), np.cos(rotate_ang), 0],
        [0, 0, 1]
    ])

    # 회전 행렬을 pose의 회전 행렬 부분과 곱하여 rotate_ang만큼 회전
    pose[:3, :3] = np.dot(pose[:3, :3], rotation_matrix)

    return pose



if __name__ == '__main__':
    start_time = time.time()

    # load observation
    sample_id = f"sample_{'{0:06d}'.format(5)}"
    sample_dir = Path(f"../datasets/downloaded_dataset_0000/{sample_id}")
    observation_start_dir = sample_dir / "observation_start"
    observation = load_competition_observation(observation_start_dir)

    # load grasp.json
    dataset_dir = Path(f"data/{sample_id}")
    grasps_dir = dataset_dir / f"grasp_pose_1.json"
    grasp_pose_fixed = read_6d_pose(grasps_dir)

    # check planning
    planning = is_grasp_executable_fn(observation, grasp_pose_fixed)

    if planning:
        print("Planning succeed!")
        print("Grasp pose is ", grasp_pose_fixed)
    else:
        print("Planning failed!")
        print("Grasp pose is ", grasp_pose_fixed)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"코드 실행 시간: {execution_time:.6f}초")