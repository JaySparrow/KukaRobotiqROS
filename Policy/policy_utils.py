import numpy as np
from typing import Union, Tuple
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from math import ceil

def trajectory_interpolate(T_start, T_end, translation_stepsize, rotation_stepsize):
    r"""
    Interpolate between two poses
    T_start, T_end[np.array]: (4, 4)
    translation_stepsize[float]: in meters
    rotation_stepsize[float]: in radians
    """
    # find number of steps for translation
    p_start = T_start[:3, 3]
    p_end = T_end[:3, 3]
    delta_p = p_end - p_start # (3, )
    delta_translation = np.linalg.norm(delta_p)
    translation_steps = ceil(delta_translation / translation_stepsize)

    # find number of steps for rotation
    R_start = T_start[:3, :3]
    R_end = T_end[:3, :3]
    r = R.from_matrix(np.stack([R_start, R_end]))
    rotvec = r.as_rotvec()
    delta_rotvec = rotvec[1] - rotvec[0]
    delta_rotation = np.linalg.norm(delta_rotvec)
    rotation_steps = ceil(delta_rotation / rotation_stepsize)

    # interpolate
    interpolation_steps = max(translation_steps, rotation_steps)
    interpolation_times = np.linspace(0, 1, interpolation_steps) # (interpolation_steps, )
    translation_interpolation = p_start + delta_p * interpolation_times[:, None] # (interpolation_steps, 3)
    slerp = Slerp([0, 1], r)
    rotation_interpolation = slerp(interpolation_times).as_matrix() # (interpolation_steps, 3, 3)

    # combine translation and rotation
    T_interpolation = np.zeros((interpolation_steps, 4, 4))
    for i in range(interpolation_steps):
        T_interpolation[i, :3, :3] = rotation_interpolation[i]
        T_interpolation[i, :3, 3] = translation_interpolation[i]
        T_interpolation[i, 3, 3] = 1

    return T_interpolation # (interpolation_steps, 4, 4)

def pose_to_T(pose_vec: np.array) -> np.array:
    """
    Convert pose vectors to transformation matrices. Works for both single and batch inputs.
    
    pose_vec[np.array]: (7,) for single pose or (N, 7) for batch of poses.
    ---
    T[np.array]: (4, 4) for single pose or (N, 4, 4) for batch of poses.
    """
    # Check if input is a single pose or a batch of poses
    S = False
    if pose_vec.ndim == 1:  # Single pose
        S = True
        pose_vec = pose_vec.reshape(1, -1)  # Convert to batch of size 1

    N = pose_vec.shape[0]
    
    # Initialize the output array with identity matrices
    T_batch = np.eye(4).reshape(1, 4, 4).repeat(N, axis=0)
    
    # Set translation (x, y, z)
    T_batch[:, :3, 3] = pose_vec[:, :3]
    
    # Convert quaternion (qx, qy, qz, qw) to rotation matrices for the entire batch
    quaternions = pose_vec[:, 3:]  # Extract the quaternion part (N, 4)
    rotation_matrices = R.from_quat(quaternions).as_matrix()  # (N, 3, 3)
    
    # Set the rotation matrices in the corresponding place
    T_batch[:, :3, :3] = rotation_matrices
    
    # Return single or batch output depending on the input
    return T_batch[0] if S else T_batch

def T_to_pose(T: np.array) -> np.array:
    """
    Convert transformation matrices to pose vectors. Works for both single and batch inputs.
    
    T[np.array]: (4, 4) for single transformation matrix or (N, 4, 4) for batch of matrices.
    ---
    pose_vec[np.array]: (7,) for single pose or (N, 7) for batch of poses.
    """
    # Check if input is a single transformation matrix or a batch
    S = False
    if T.ndim == 2:  # Single transformation matrix
        S = True
        T = T.reshape(1, 4, 4)  # Convert to batch of size 1

    N = T.shape[0]
    
    # Pre-allocate the output pose vectors
    pose_vec = np.zeros((N, 7))
    
    # Extract the translation (x, y, z) from the 4th column
    pose_vec[:, :3] = T[:, :3, 3]
    
    # Extract the rotation matrices (first 3x3 block) and convert to quaternions
    rotation_matrices = T[:, :3, :3]  # (N, 3, 3)
    quaternions = R.from_matrix(rotation_matrices).as_quat()  # (N, 4)
    
    # Set the quaternion part (qx, qy, qz, qw)
    pose_vec[:, 3:] = quaternions
    
    # Return single or batch output depending on the input
    return pose_vec[0] if S else pose_vec

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
    