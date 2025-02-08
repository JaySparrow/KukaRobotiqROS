#!/usr/bin/env python

from tf.transformations import quaternion_from_euler, quaternion_matrix, quaternion_from_matrix
import numpy as np

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from math import ceil
from typing import Tuple

## Pose of the Gripper in the End-Effector's reference frame eef_T_g (4, 4), i.e. transformation matrix from Gripper's frame to End-Effector's frame
eef_to_fingertip = 0.26
# translate along the z-axis (in End-Effector's)
eef_p_g = np.array([
    [0.],
    [0.],
    [eef_to_fingertip]
]) 
# rotate 15 degree clock-wisely in z-axis up (in End-Effector's)
degree_diff = -15.
eef_quat_g = quaternion_from_euler(0., 0., -degree_diff/180.*np.pi)
eef_R_g = quaternion_matrix(eef_quat_g)[:3, :3]
# SE(3) Pose
eef_T_g = np.block([
    [eef_R_g, eef_p_g],
    [0., 0., 0., 1.]
])

# print("== Transformation from Gripper's Frame to End-Effector's Frame ==\n", eef_T_g.round(decimals=3))
# print("")

## b_T_eef = b_T_g @ (eef_T_g)^{-1}
# both input and output are in the arm's base frame
def transform_pose_gripper_fingertip_to_eef(b_pose_g): # pose(7) -> pose(7) [x, y, z, rx, ry, rz, rw]
    b_pose_g = b_pose_g.flatten()
    assert b_pose_g.shape[0] == 7

    # gripper position
    b_p_g = b_pose_g[:3].reshape((-1, 1)) # (3, 1)
    # gripper rotation
    b_quat_g = b_pose_g[3:] # (4, )
    b_R_g = quaternion_matrix(b_quat_g)[:3, :3] # (3, 3)
    # gripper pose matrix
    b_T_g = np.block([
        [b_R_g, b_p_g],
        [0., 0., 0., 1.]
    ])

    # end-effector pose matrix
    b_T_eef = np.matmul(b_T_g, np.linalg.inv(eef_T_g)) # (4, 4)
    # end-effector position
    b_p_eef = b_T_eef[:3, 3].flatten() # (3, )
    # end-effector rotation
    b_R_eef = b_T_eef[:3, :3] # (3, 3)
    b_R_eef = np.block([
        [b_R_eef, np.zeros((3, 1), dtype=float)],
        [0., 0., 0., 1.]
    ])
    b_quat_eef = quaternion_from_matrix(b_R_eef).flatten() # (4, )

    return np.concatenate([b_p_eef, b_quat_eef])

## b_T_g = b_T_eef @ eef_T_g
# both input and output are in the arm's base frame
def transform_pose_eef_to_gripper_fingertip(b_pose_eef): # pose(7) -> pose(7) [x, y, z, rx, ry, rz, rw]
    b_pose_eef = b_pose_eef.flatten()
    assert b_pose_eef.shape[0] == 7

    # end-effector position
    b_p_eef = b_pose_eef[:3].reshape((-1, 1)) # (3, 1)
    # end-effector rotation
    b_quat_eef = b_pose_eef[3:].flatten() # (4, )
    b_R_eef = quaternion_matrix(b_quat_eef)[:3, :3] # (3, 3)
    # end-effector pose matrix
    b_T_eef = np.block([
        [b_R_eef, b_p_eef],
        [0., 0., 0., 1.]
    ])

    # gripper pose matrix
    b_T_g = np.matmul(b_T_eef, eef_T_g) # (4, 4)roslaunch tutorial go_to_home.launch
    # gripper position
    b_p_g = b_T_g[:3, 3].flatten() # (3, )
    # gripper rotation
    b_R_g = b_T_g[:3, :3] # (3, 3)
    b_R_g = np.block([
        [b_R_g, np.zeros((3, 1), dtype=float)],
        [0., 0., 0., 1.]
    ])
    b_quat_g = quaternion_from_matrix(b_R_g).flatten() # (4, )
    
    return np.concatenate([b_p_g, b_quat_g])

def transform_gripper_pose(curr_g_pose, transform_pose):
    curr_g_pose = curr_g_pose.flatten()
    transform_pose = transform_pose.flatten()
    assert curr_g_pose.shape[0] == 7

    if len(transform_pose) == 3:
        transform_pose = np.concatenate([transform_pose, np.array([0., 0., 0., 1.])])
    elif len(transform_pose) == 4:
        transform_pose = np.concatenate([np.array([0., 0., 0.]), transform_pose])
    elif len(transform_pose) == 7:
        transform_pose = transform_pose.copy()
    else:
        transform_pose = np.array([0., 0., 0., 0., 0., 0., 1.])
    assert transform_pose.shape[0] == 7

    curr_g_p = curr_g_pose[:3].reshape((-1 ,1))
    curr_g_R = quaternion_matrix(curr_g_pose[3:])[:3, :3]
    curr_g_T = np.block([
        [curr_g_R, curr_g_p],
        [0., 0., 0., 1.]
    ])

    transform_p = transform_pose[:3].reshape((-1 ,1))
    transform_R = quaternion_matrix(transform_pose[3:])
    transform_T = np.block([
        [transform_R[:3, :3], transform_p],
        [0., 0., 0., 1.]
    ])

    # new_g_T = np.matmul(transform_T, curr_g_T)
    new_g_T = np.matmul(curr_g_T, np.linalg.inv(transform_T))
    new_g_p = new_g_T[:3, 3].flatten()
    new_g_R = np.block([
        [new_g_T[:3, :3], np.zeros((3, 1), dtype=float)],
        [0., 0., 0., 1.]
    ])
    new_g_quat = quaternion_from_matrix(new_g_R).flatten()

    return np.concatenate([new_g_p, new_g_quat])

def transform_gripper_pose_local(curr_g_pose, transform_pose):
    curr_g_pose = curr_g_pose.flatten()
    transform_pose = transform_pose.flatten()
    assert curr_g_pose.shape[0] == 7

    if len(transform_pose) == 3:
        transform_pose = np.concatenate([transform_pose, np.array([0., 0., 0., 1.])])
    elif len(transform_pose) == 4:
        transform_pose = np.concatenate([np.array([0., 0., 0.]), transform_pose])
    elif len(transform_pose) == 7:
        transform_pose = transform_pose.copy()
    else:
        transform_pose = np.array([0., 0., 0., 0., 0., 0., 1.])
    assert transform_pose.shape[0] == 7

    curr_g_p = curr_g_pose[:3].reshape((-1 ,1))
    curr_g_R = quaternion_matrix(curr_g_pose[3:])[:3, :3]

    transform_p = transform_pose[:3].reshape((-1 ,1))
    transform_R = quaternion_matrix(transform_pose[3:])[:3, :3]

    new_g_p = (curr_g_p + transform_p).flatten()
    new_g_R = np.matmul(transform_R, curr_g_R)
    new_g_R = np.block([
        [new_g_R, np.zeros((3, 1), dtype=float)],
        [0., 0., 0., 1.]
    ])
    new_g_quat = quaternion_from_matrix(new_g_R).flatten()

    return np.concatenate([new_g_p, new_g_quat])

def trajectory_interpolate(pose_start, pose_end, translation_stepsize, rotation_stepsize):
    r"""
    Interpolate between two poses
    pose_start, pose_end[np.array]: (7, )
    translation_stepsize[float]: in meters
    rotation_stepsize[float]: in radians
    """
    # find number of steps for translation
    p_start = pose_start[:3]
    p_end = pose_end[:3]
    delta_p = p_end - p_start # (3, )
    delta_translation = np.linalg.norm(delta_p)
    translation_steps = ceil(delta_translation / translation_stepsize)

    # find number of steps for rotation
    R_start = R.from_quat(pose_start[3:]).as_matrix()
    R_end = R.from_quat(pose_end[3:]).as_matrix()
    r = R.from_matrix(np.stack([R_start, R_end]))
    rotvec = r.as_rotvec()
    delta_rotvec = rotvec[1] - rotvec[0]
    delta_rotation = np.linalg.norm(delta_rotvec)
    rotation_steps = ceil(delta_rotation / rotation_stepsize)

    interpolation_steps = max(translation_steps, rotation_steps)
    if interpolation_steps <= 1:
        return pose_end.reshape((1, -1))
    # interpolate
    interpolation_times = np.linspace(0, 1, interpolation_steps) # (interpolation_steps, )
    translation_interpolation = p_start + delta_p * interpolation_times[:, None] # (interpolation_steps, 3)

    slerp = Slerp([0, 1], r)
    rotation_interpolation = slerp(interpolation_times).as_quat() # (interpolation_steps, 4)
    rotation_interpolation /= np.linalg.norm(rotation_interpolation, axis=1)[:, None]

    return np.concatenate([translation_interpolation, rotation_interpolation], axis=1) # (interpolation_steps, 7)
 
def zero_out_axis_transformation(start_pose: np.array, goal_pose: np.array, axis: np.array):
    r"""
    start_pose: (7, ), [x, y, z, qx, qy, qz, qw]
    goal_pose: (7, ), [x, y, z, qx, qy, qz, qw]
    axis: (0 < n <= 6, ), {0, 1, 2, 3, 4, 5} representing x, y, z, rx, ry, rz
    """
    start_pose_6d = np.concatenate([
            start_pose[:3],
            R.from_quat(start_pose[3:7]).as_rotvec()
        ]) # (6, )
    goal_pose_6d = np.concatenate([
            goal_pose[:3],
            R.from_quat(goal_pose[3:7]).as_rotvec()
        ]) # (6, )

    goal_pose_6d[axis] = start_pose_6d[axis]

    goal_pose_7d = np.concatenate([
        goal_pose_6d[:3],
        R.from_rotvec(goal_pose_6d[3:7]).as_quat()
    ]) # (7, )
    return goal_pose_7d

def calculate_pose_distance(pose1: np.array, pose2: np.array) -> Tuple[float, float]:
    r"""
    pose1, pose2[np.array]: [x, y, z, qx, qy, qz, qw]
    ---
    translation_distance[float]: in meters
    rotation_distance[float]: in radians
    """
    delta_p = pose1[:3] - pose2[:3]
    translation_distance = np.linalg.norm(delta_p)

    rotvec1 = R.from_quat(pose1[3:]).as_rotvec()
    rotvec2 = R.from_quat(pose2[3:]).as_rotvec()
    delta_rotvec = rotvec1 - rotvec2
    rotation_distance = np.linalg.norm(delta_rotvec)

    return translation_distance, rotation_distance