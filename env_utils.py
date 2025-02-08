import torch_jit_utils
import torch

def get_keypoint_offsets(num_keypoints, axis=-1):
    """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""

    keypoint_offsets = torch.zeros((num_keypoints, 3))
    keypoint_offsets[:, axis] = (
        torch.linspace(0.0, 1.0, num_keypoints) - 0.5
    )

    return keypoint_offsets

def check_keypoints_close(kp1, kp2, dist_threshold):
    """Check if two sets of keypoints are close to each other."""

    # Compute keypoint distance
    keypoint_dist = torch.norm(kp2 - kp1, p=2, dim=-1)

    # Check if keypoint distance is below threshold
    is_keypoints_close = torch.where(
        torch.sum(keypoint_dist, dim=-1) < dist_threshold,
        1,
        0,
    ).bool()

    return is_keypoints_close # (N, )

def calc_feasible_nut_quaternions(nut_quat):
    r"""
    1. z-axis pointing up
    2. all six yaw rotations are feasible (60 degrees apart)
    3. sort the quaternions in the order of the positive yaw rotations, with the first yaw rotation closest to the world's yaw rotation
    ---
    nut_quat: (N, 4)
    ---
    return: (N, 6x4), [:, ix4: (i+1)x4] is the i-th feasible quaternion
    """
    N = nut_quat.shape[0]
    device = nut_quat.device
    dtype = nut_quat.dtype

    # z-axis pointing up
    nut_z_axis = torch_jit_utils.quat_axis(nut_quat, axis=2) # (N, 3)
    base_z_axis = torch.tensor([[0, 0, 1]], device=device, dtype=dtype).T # (3, 1)
    is_z_down = (nut_z_axis @ base_z_axis < 0).flatten() # (N, )

    inverting_z_quat = torch_jit_utils.quat_from_angle_axis(
        torch.tensor([torch.pi], device=device, dtype=dtype).repeat(N), 
        torch.tensor([[1, 0, 0]], device=device, dtype=dtype).repeat(N, 1)
    ) # (N, 4)
    feasible_nut_quat = nut_quat.clone()
    feasible_nut_quat[is_z_down] = torch_jit_utils.quat_mul(nut_quat[is_z_down], inverting_z_quat[is_z_down]) # (N, 4)

    # all six yaw rotations are feasible (60 degrees apart)
    rotating_around_z_quat = torch_jit_utils.quat_from_angle_axis(
        torch.tensor([torch.pi/3], device=device, dtype=dtype).repeat(N), 
        torch.tensor([[0, 0, 1]], device=device, dtype=dtype).repeat(N, 1)
    ) # (N, 4)

    world_x_axis = torch.tensor([1, 0, 0], device=device, dtype=dtype).unsqueeze(-1) # (3, 1)

    nut_quat_i = feasible_nut_quat.clone()
    nut_x_axis_i = torch_jit_utils.quat_axis(nut_quat_i, axis=0) # (N, 3)
    cos_dist_i = nut_x_axis_i @ world_x_axis # (N, 1)

    min_cos_dist = cos_dist_i # (N, 1)
    min_nut_quat = nut_quat_i # (N, 4)
    for i in range(5):
        nut_quat_i = torch_jit_utils.quat_mul(nut_quat_i, rotating_around_z_quat)
        nut_x_axis_i = torch_jit_utils.quat_axis(nut_quat_i, axis=0) # (N, 3)
        cos_dist_i = nut_x_axis_i @ world_x_axis # (N, 1)
        is_smaller = (cos_dist_i < min_cos_dist).squeeze(-1)
        min_cos_dist[is_smaller, :] = cos_dist_i[is_smaller, :]
        min_nut_quat[is_smaller, :] = nut_quat_i[is_smaller, :]

    feasible_nut_quaternions = torch.zeros((N, 6*4), device=device, dtype=dtype)
    feasible_nut_quaternions[:, 0:4] = min_nut_quat # feasible_nut_quat
    for i in range(5):
        feasible_nut_quaternions[:, (i+1)*4:(i+2)*4] = torch_jit_utils.quat_mul(feasible_nut_quaternions[:, i*4:(i+1)*4], rotating_around_z_quat)

    return feasible_nut_quaternions