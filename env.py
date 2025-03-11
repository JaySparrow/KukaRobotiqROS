#!/usr/bin/env python
import sys
sys.path.append("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Vision/")
sys.path.append("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Policy/")
sys.path.append("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Control/")

import numpy as np
import cv2
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
import torch
import copy

import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo

from kuka_base import KukaBase
import torch_jit_utils
from vision_utils import visualize_pose_2d
import env_utils

from icecream import ic

class KukaWrenchInsertEnv:
    def __init__(self, cfg: dict):
        self.robot = KukaBase()
        self.cfg = copy.deepcopy(cfg)
        if "cam_T_base" not in self.cfg:
            self.cfg["cam_T_base"] = np.eye(4, dtype=np.float32)
        if "action_as_object_displacement" not in self.cfg:
            self.cfg["action_as_object_displacement"] = False
        if "num_keypoints" not in self.cfg:
            self.cfg["num_keypoints"] = 4
        if "keypoint_scale" not in self.cfg:
            self.cfg["keypoint_scale"] = 0.5
        if "success_height_thresh" not in self.cfg:
            self.cfg["success_height_thresh"] = 0.007 # 0.005
        if "close_error_thresh" not in self.cfg:
            self.cfg["close_error_thresh"] = 0.08 # 0.05

        ## Pose Subscribers ##
        rospy.Subscriber("/pose_tracker/nut1", PoseStamped, self.nut_callback)
        rospy.Subscriber("/pose_tracker/wrench1_head", PoseStamped, self.wrench_head_callback)
        # rospy.Subscriber("/pose_tracker/wrench5", PoseStamped, self.wrench_callback)
        self.nut_pose = None
        self.wrench_head_pose = None
        # self.wrench_pose = None

        ## Camera Subscribers ##
        rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.cam_info_callback)
        self.rgb = None
        self.cam_K = None
        self.bridge = CvBridge()

        ## DIRTY PART: Correct for delta actions
        # simulation: delta applied to gripper in WORLD frame
        # real      : delta applied to gripper in ROBOT BASE frame ##
        self.world_to_robot_base = np.loadtxt("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Control/world_to_robot_base.txt", dtype=np.float32) # (7, )
        self.world_to_robot_base = torch.from_numpy(self.world_to_robot_base)

        self.acquire_task_tensors()

    def acquire_task_tensors(self):
        # Compute pose of wrench head goal in nut's frame
        self.wrench_head_goal_pos_local = torch.tensor([[0.0, 0.0, 0.0]])
        roll = torch.zeros((1, ))
        pitch = -90/180 * torch.pi * torch.ones((1, ))
        yaw = -90/180 * torch.pi * torch.ones((1, ))
        self.wrench_head_goal_quat_local = torch_jit_utils.quat_from_euler_xyz(roll, pitch, yaw)

        # eef force torque threshold
        # self.eef_delta_force_torque_thresh = torch.tensor([[4, 4, 3.0, 0.3, 0.3, 0.4]]) # (1, 6)
        # self.eef_delta_force_torque_thresh = torch.tensor([[3, 3, 3, 0.3, 0.3, 0.4]]) # (1, 6)
        self.eef_delta_force_torque_thresh = torch.tensor([[1.5, 3, 3, 0.3, 0.3, 0.4]]) # (1, 6)

        ## keypoints
        self.keypoint_offsets_wrench_head = (
            env_utils.get_keypoint_offsets(self.cfg["num_keypoints"], axis=0)
            * self.cfg["keypoint_scale"]
        ) # (num_keypoints, 3)
        self.keypoint_offsets_nut = (
            env_utils.get_keypoint_offsets(self.cfg["num_keypoints"], axis=-1)
            * self.cfg["keypoint_scale"]
        ) # (num_keypoints, 3)

        self.identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0]).unsqueeze(0)

    def is_subscriber_ready(self) -> bool:
        return (
            self.nut_pose is not None and 
            self.wrench_head_pose is not None and 
            # self.wrench_pose is not None and
            self.rgb is not None and
            self.cam_K is not None
        )

    def wait_for_subscribers(self, seconds: int=5):
        sub_seconds = 1
        num_iter = int(seconds / sub_seconds)
        for _ in range(num_iter):
            if self.is_subscriber_ready() and self.robot.is_subscriber_ready():
                print("Subscribers are ready!")
                return
            rospy.sleep(sub_seconds)
        print(f"Subscribers are not responding within {seconds} seconds!")
        exit(0)

    def reset(self):
        ## Reset reference force torque
        robot_state = self.robot.get_robot_state()
        self.reference_eef_force_torque = torch.from_numpy(robot_state["eef_force_torque"][None, :]) # (1, 6)
        
        # ## Reset nut standard transform
        # nut_quat = torch.from_numpy(self.nut_pose[None, 3:7])
        # standard_nut_quat = env_utils.calc_feasible_nut_quaternions(nut_quat)[:, 8:12] # (1, 4)
        # self.standard_nut_quat_local = torch_jit_utils.quat_mul(torch_jit_utils.quat_conjugate(nut_quat), standard_nut_quat)

        ## Reset initial pose
        self.move_gripper_to_init_wrench_pose()

        self.progress = 0

        if self.cfg["observation_type"] == 0:
            obs = self.compute_observations()
        elif self.cfg["observation_type"] == 1:
            obs = self.compute_object_observations()
        else:
            obs = self.compute_ablation_observations()

        self.ctrl_target_fingert_pos = torch.from_numpy(robot_state["fingertip_pose"][None, :3])
        self.ctrl_target_fingertip_quat = torch.from_numpy(robot_state["fingertip_pose"][None, 3:7])

        self.recorded_poses = {}
        
        return obs
    
    def step(self, actions: torch.Tensor, update_progress: bool=True):
        # print(f"[step] {self.progress}")
        if self.cfg["action_as_object_displacement"]:
            self.apply_object_actions_as_ctrl_targets(actions, do_scale=True, regularize_with_force=True)
            # obs = self.compute_object_observations()
        else:
            self.apply_actions_as_ctrl_targets(actions, do_scale=True, regularize_with_force=True)
            # obs = self.compute_observations()
        if self.cfg["observation_type"] == 0:
            obs = self.compute_observations()
        elif self.cfg["observation_type"] == 1:
            obs = self.compute_object_observations()
        else:
            obs = self.compute_ablation_observations()

        self.record_poses()

        if update_progress:
            self.progress += 1
        is_done = self.progress >= self.cfg["max_episode_length"] or bool(self.check_wrench_head_inserted_in_nut())
        return obs, is_done
    
    def get_standard_nut_pose(self):
        nut_quat = torch.from_numpy(self.nut_pose[None, 3:7])
        nut_pos = torch.from_numpy(self.nut_pose[None, :3])
        standard_nut_quat = torch_jit_utils.quat_mul(nut_quat, self.standard_nut_quat_local)
        return nut_pos, standard_nut_quat

    def get_wrench_head_pose(self):
        wrench_head_quat = torch.from_numpy(self.wrench_head_pose[None, 3:7])
        wrench_head_pos = torch.from_numpy(self.wrench_head_pose[None, :3])
        return wrench_head_pos, wrench_head_quat

    def get_keypoints(self):
        num_keypoints = self.cfg["num_keypoints"]
        nut_pos, nut_quat = self.get_standard_nut_pose()
        wrench_head_pos, wrench_head_quat = self.get_wrench_head_pose()

        keypoints_wrench_head = torch_jit_utils.tf_combine(
            wrench_head_quat.repeat((num_keypoints, 1)), 
            wrench_head_pos.repeat((num_keypoints, 1)), 
            self.identity_quat.repeat((num_keypoints, 1)),
            self.keypoint_offsets_wrench_head, 
        )[1]
        keypoints_nut = torch_jit_utils.tf_combine(
            nut_quat.repeat((num_keypoints, 1)), 
            nut_pos.repeat((num_keypoints, 1)), 
            self.identity_quat.repeat((num_keypoints, 1)),
            self.keypoint_offsets_nut, 
        )[1]
        return keypoints_wrench_head[None, ...], keypoints_nut[None, ...] # (1, num_keypoints, 3), (1, num_keypoints, 3)

    def check_wrench_head_inserted_in_nut(self):
        nut_pos, _ = self.get_standard_nut_pose()
        wrench_head_pos, _ = self.get_wrench_head_pose()
        is_wrench_head_between_insertion_height = torch.logical_and(
            wrench_head_pos[:, 2] < nut_pos[:, 2] + self.cfg["success_height_thresh"],
            wrench_head_pos[:, 2] > nut_pos[:, 2] - self.cfg["success_height_thresh"],
        )

        keypoints_wrench_head, keypoints_nut = self.get_keypoints()
        is_wrench_head_close_to_nut = env_utils.check_keypoints_close(
            keypoints_wrench_head, keypoints_nut, dist_threshold=self.cfg["close_error_thresh"]
        )

        is_plug_inserted_in_socket = torch.logical_and(
            is_wrench_head_between_insertion_height, 
            is_wrench_head_close_to_nut,
        )
        # ic(is_wrench_head_between_insertion_height, 
        #     is_wrench_head_close_to_nut,)
        return is_plug_inserted_in_socket # (1, )

    def compute_observations(self):
        """Tensor"""
        ## Env state
        # nut_quat = torch.from_numpy(self.nut_pose[None, 3:7])
        # nut_pos = torch.from_numpy(self.nut_pose[None, :3])
        nut_pos, nut_quat = self.get_standard_nut_pose()

        ## Robot state
        robot_state = self.robot.get_robot_state()
        arm_dof_pos = torch.from_numpy(robot_state["joint_pos"][None, :])
        fingertip_quat = torch.from_numpy(robot_state["fingertip_pose"][None, 3:7])
        fingertip_pos = torch.from_numpy(robot_state["fingertip_pose"][None, :3])

        ## Compute the fingertip goal pose
        # wrench head goal pose
        wrench_head_goal_quat, wrench_head_goal_pos = torch_jit_utils.tf_combine(
            nut_quat,
            nut_pos,
            self.wrench_head_goal_quat_local.clone(),
            self.wrench_head_goal_pos_local.clone()
        )
        fingertip_goal_quat, fingertip_goal_pos = self.wrench_head_to_fingertip(wrench_head_goal_quat, wrench_head_goal_pos)

        def rotate(q):
            roll, pitch, yaw = torch_jit_utils.get_euler_xyz(q)
            q = -torch_jit_utils.quat_from_euler_xyz(roll, pitch, (np.pi - yaw) % (2 * np.pi))
            return q

        ## Prepare observation tensors
        delta_pos = fingertip_goal_pos - fingertip_pos
        obs_tensors = [
            arm_dof_pos,
            fingertip_pos,
            rotate(fingertip_quat),
            fingertip_goal_pos,
            rotate(fingertip_goal_quat),
            delta_pos
        ]
        obs_buf = torch.cat(obs_tensors, dim=1) # (1, 24)

        return obs_buf
    
    def compute_ablation_observations(self):
        ## Env state in robot base's frame
        nut_pos_robot, nut_quat_robot = self.get_standard_nut_pose()
        # wrench_head_quat_robot = torch.from_numpy(self.wrench_head_pose[None, 3:7]) # (1, 4)
        # wrench_head_pos_robot = torch.from_numpy(self.wrench_head_pose[None, :3]) # (1, 3)
        wrench_head_pos_robot, wrench_head_quat_robot = self.get_wrench_head_pose()
        ## Robot state in robot base's frame
        robot_state = self.robot.get_robot_state()
        fingertip_quat_robot = torch.from_numpy(robot_state["fingertip_pose"][None, 3:7])
        fingertip_pos_robot = torch.from_numpy(robot_state["fingertip_pose"][None, :3])
        ## Goal state in robot base's frame
        wrench_head_goal_quat_robot, wrench_head_goal_pos_robot = torch_jit_utils.tf_combine(
            nut_quat_robot,
            nut_pos_robot,
            self.wrench_head_goal_quat_local,
            self.wrench_head_goal_pos_local
        )
        fingertip_goal_quat_robot, fingertip_goal_pos_robot = self.wrench_head_to_fingertip(
            wrench_head_goal_quat_robot, 
            wrench_head_goal_pos_robot
        )

        if self.cfg["observation_type"] == 2: # Isidoros (plug+socket in robot base)
            print("> Obs type 2")
            noisy_delta_pos = wrench_head_pos_robot - nut_pos_robot
            ## Define observations (for actor)
            obs_tensors = [
                wrench_head_pos_robot, # 3
                wrench_head_quat_robot, # 4

                nut_pos_robot, # 3
                nut_quat_robot, # 4

                noisy_delta_pos, # 3
            ]  # 17
        
        elif self.cfg["observation_type"] == 3: # (socket current+goal in robot base)
            print("> Obs type 3")
            noisy_delta_pos = wrench_head_pos_robot - wrench_head_goal_pos_robot
            ## Define observations (for actor)
            obs_tensors = [
                wrench_head_pos_robot, # 3
                wrench_head_quat_robot, # 4

                wrench_head_goal_pos_robot, # 3
                wrench_head_goal_quat_robot, # 4

                noisy_delta_pos, # 3
            ]

        elif self.cfg["observation_type"] == 4: # IndustReal (fingertip current+goal in robot base)
            print("> Obs type 4")
            noisy_delta_pos = fingertip_pos_robot - fingertip_goal_pos_robot
            ## Define observations (for actor)
            obs_tensors = [
                fingertip_pos_robot, # 3
                fingertip_quat_robot, # 4

                fingertip_goal_pos_robot, # 3
                fingertip_goal_quat_robot, # 4

                noisy_delta_pos, # 3
            ]
            
        obs_buf = torch.cat(
            obs_tensors, dim=-1
        )  # shape = (num_envs, num_observations)

        return obs_buf
    
    def compute_object_observations(self):
        """Tensor"""
        ## Env state
        nut_pos, nut_quat = self.get_standard_nut_pose()
        nut_quat_inv, nut_pos_inv = torch_jit_utils.tf_inverse(nut_quat, nut_pos)

        ## Compute the current wrench head pose in nut's frame
        # in robot base's frame
        # wrench_head_quat = torch.from_numpy(self.wrench_head_pose[None, 3:7]) # (1, 4)
        # wrench_head_pos = torch.from_numpy(self.wrench_head_pose[None, :3]) # (1, 3)
        wrench_head_pos, wrench_head_quat = self.get_wrench_head_pose()
        # in nut's frame
        wrench_head_quat_nut, wrench_head_pos_nut = torch_jit_utils.tf_combine(
            nut_quat_inv,
            nut_pos_inv,
            wrench_head_quat,
            wrench_head_pos
        )

        ## delta pos of wrench head in nut's frame
        delta_pos = wrench_head_pos_nut - self.wrench_head_goal_pos_local

        ## Prepare observation tensors
        obs_tensors = [
            wrench_head_pos_nut, # 3
            wrench_head_quat_nut, # 4

            self.wrench_head_goal_pos_local, # 3
            self.wrench_head_goal_quat_local, # 4

            delta_pos, # 3
        ] # 17
        obs_buf = torch.cat(obs_tensors, dim=1) # (1, 17)

        return obs_buf
    
    def record_poses(self):
        if not hasattr(self, "recorded_poses") or not isinstance(self.recorded_poses, dict):
            self.recorded_poses = {}

        nut_pos, nut_quat = self.get_standard_nut_pose()
        wrench_head_pos, wrench_head_quat = self.get_wrench_head_pose()
        
        wrench_head_target_quat, wrench_head_target_pos = self.fingertip_to_wrench_head(
            self.ctrl_target_fingertip_quat, self.ctrl_target_fingert_pos)
        
        ## Robot state
        robot_state = self.robot.get_robot_state()
        fingertip_quat = torch.from_numpy(robot_state["fingertip_pose"][None, 3:7])
        fingertip_pos = torch.from_numpy(robot_state["fingertip_pose"][None, :3])

        wrench_head_goal_quat, wrench_head_goal_pos = torch_jit_utils.tf_combine(
            nut_quat,
            nut_pos,
            self.wrench_head_goal_quat_local.clone(),
            self.wrench_head_goal_pos_local.clone()
        )
        fingertip_goal_quat, fingertip_goal_pos = self.wrench_head_to_fingertip(wrench_head_goal_quat, wrench_head_goal_pos)
        
        ## in robot frame
        poses_dict = {
            "wrench_head": torch.cat([wrench_head_pos, wrench_head_quat], dim=-1),
            "wrench_head_target": torch.cat([wrench_head_target_pos, wrench_head_target_quat], dim=-1),
            "wrench_head_goal": torch.cat([wrench_head_goal_pos, wrench_head_goal_quat], dim=-1),
            "fingertip": torch.cat([fingertip_pos, fingertip_quat], dim=-1),
            "fingertip_target": torch.cat([self.ctrl_target_fingert_pos, self.ctrl_target_fingertip_quat], dim=-1),
            "fingertip_goal": torch.cat([fingertip_goal_pos, fingertip_goal_quat], dim=-1),
        }

        def to_socket(pose_in_world):
            socket_quat_inv, socket_pos_inv = torch_jit_utils.tf_inverse(nut_quat, nut_pos)
            quat_in_socket, pos_in_socket =  torch_jit_utils.tf_combine(
                socket_quat_inv, 
                socket_pos_inv, 
                pose_in_world[:, 3:], 
                pose_in_world[:, :3]
            )
            return torch.cat([pos_in_socket, quat_in_socket], dim=-1)
        
        for key, pose in poses_dict.items():
            if key not in self.recorded_poses:
                self.recorded_poses[key] = []
            # self.recorded_poses[key].append((to_socket(pose)))
            if "wrench" in key: # in nut's frame
                self.recorded_poses[key].append(to_socket(pose))
            elif "fingertip" in key: # in robot base's frame
                self.recorded_poses[key].append(pose)
    
    def apply_actions_as_ctrl_targets(self, actions: torch.Tensor, do_scale: bool, regularize_with_force: bool=True):
        """Tensor"""
        actions = torch.clamp(actions, -1.0, 1.0)
        ## Robot state
        robot_state = self.robot.get_robot_state()
        fingertip_quat = torch.from_numpy(robot_state["fingertip_pose"][None, 3:7])
        fingertip_pos = torch.from_numpy(robot_state["fingertip_pose"][None, :3])
        current_eef_force_torque = torch.from_numpy(robot_state["eef_force_torque"][None, :]) # (1, 6)

        ## Regularize with force
        if regularize_with_force:
            delta_force_torque = torch.abs(current_eef_force_torque - self.reference_eef_force_torque) # [f_x, f_y, f_z, t_x, t_y, t_z]
            action_regularizer = (delta_force_torque < self.eef_delta_force_torque_thresh).float()
            actions = actions * action_regularizer
            # print(np.nonzero(actions))

        pos_actions, rot_actions = actions[:, :3], actions[:, 3:6]
        ## Scale
        if do_scale:
            pos_actions = pos_actions @ torch.diag(
                torch.tensor(self.cfg["pos_action_scale"])
            )
            rot_actions = rot_actions @ torch.diag(
                torch.tensor(self.cfg["rot_action_scale"])
            )

        ## Position
        ## DIRTY PART: Correct for delta actions
        # pos_actions = torch_jit_utils.quat_apply(self.world_to_robot_base[None, 3:7], pos_actions) # (1, 3)

        target_fingertip_pos = fingertip_pos # + pos_actions

        wrench_head_pos, wrench_head_quat = self.get_wrench_head_pose()
        nut_pos, nut_quat = self.get_standard_nut_pose()
        
        # if wrench_head_pos[:, 2] - nut_pos[:, 2] < 0.018:
        #     target_fingertip_pos[:, 0] -= pos_actions[:, 0]
        #     target_fingertip_pos[:, 1] -= pos_actions[:, 1]
        #     target_fingertip_pos[:, 2] += pos_actions[:, 2]
        # else: 
        #     target_fingertip_pos += pos_actions
        target_fingertip_pos += pos_actions

        ## Rotation
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_jit_utils.quat_from_angle_axis(angle, axis)
        if self.cfg["clamp_rot"]:
            rot_actions_quat = torch.where(
                angle.unsqueeze(-1).repeat(1, 4) > self.cfg["clamp_rot_thresh"],
                rot_actions_quat,
                torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            )
        ## DIRTY PART: Correct for delta actions
        # rot_actions_quat = torch_jit_utils.quat_mul(
        #     self.world_to_robot_base[None, 3:7],
        #     torch_jit_utils.quat_mul(
        #         rot_actions_quat,
        #         torch_jit_utils.quat_conjugate(self.world_to_robot_base[None, 3:7])
        #     )
        # ) # (1, 4)
        target_fingertip_quat = torch_jit_utils.quat_mul(
            rot_actions_quat, fingertip_quat
        )
        self.ctrl_target_fingert_pos = target_fingertip_pos.clone()
        self.ctrl_target_fingertip_quat = target_fingertip_quat.clone()
        
        ## Apply
        target_fingertip_pose = torch.cat([target_fingertip_pos, target_fingertip_quat], axis=-1)
        self.robot.move_fingertip(target_fingertip_pose.numpy()[0], timeout=5)

    def apply_object_actions_as_ctrl_targets(self, actions: torch.Tensor, do_scale: bool, regularize_with_force: bool=True):
        """Tensor"""
        # Map to [-1, 1]
        actions = torch.clamp(actions, -1.0, 1.0)

        ## Env state
        # in robot base's frame
        # wrench_head_quat = torch.from_numpy(self.wrench_head_pose[None, 3:7]) # (1, 4)
        # wrench_head_pos = torch.from_numpy(self.wrench_head_pose[None, :3]) # (1, 3)
        wrench_head_pos, wrench_head_quat = self.get_wrench_head_pose()

        ## Robot state
        robot_state = self.robot.get_robot_state()
        current_eef_force_torque = torch.from_numpy(robot_state["eef_force_torque"][None, :]) # (1, 6)

        ## Regularize with force (directed)
        if regularize_with_force:
            delta_force_torque = current_eef_force_torque - self.reference_eef_force_torque # [f_x, f_y, f_z, t_x, t_y, t_z]
            action_direction = torch.sign(actions)
            # positive
            pos_action_zeroer = torch.logical_and(delta_force_torque > self.eef_delta_force_torque_thresh, action_direction[:, [2, 1, 0, 5, 4, 3]] > 0)
            # negative
            neg_action_zeroer = torch.logical_and(delta_force_torque < -self.eef_delta_force_torque_thresh, action_direction[:, [2, 1, 0, 5, 4, 3]] < 0)
            action_regularizer = 1 - torch.logical_or(pos_action_zeroer, neg_action_zeroer).float()
            action_regularizer = action_regularizer[:, [2, 1, 0, 5, 4, 3]]
            actions = actions * action_regularizer

        pos_actions, rot_actions = actions[:, :3], actions[:, 3:6]
        ## Scale
        if do_scale:
            pos_actions = pos_actions @ torch.diag(
                torch.tensor(self.cfg["pos_action_scale"])
            )
            rot_actions = rot_actions @ torch.diag(
                torch.tensor(self.cfg["rot_action_scale"])
            )

        ## Convert to quaternion
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_jit_utils.quat_from_angle_axis(angle, axis)
        if self.cfg["clamp_rot"]:
            rot_actions_quat = torch.where(
                angle.unsqueeze(-1).repeat(1, 4) > self.cfg["clamp_rot_thresh"],
                rot_actions_quat,
                torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            )

        ## Calculate the target wrench head pose in the robot base frame
        ctrl_target_wrench_head_quat, ctrl_target_wrench_head_pos = torch_jit_utils.tf_combine(
            wrench_head_quat, wrench_head_pos, rot_actions_quat, pos_actions)
        
        ## Transform to gripper fingertip's action target
        ctrl_target_fingertip_quat, ctrl_target_fingert_pos = self.wrench_head_to_fingertip(
            ctrl_target_wrench_head_quat, ctrl_target_wrench_head_pos)
        self.ctrl_target_fingertip_quat = ctrl_target_fingertip_quat.clone()
        self.ctrl_target_fingert_pos = ctrl_target_fingert_pos.clone()

        ## Apply
        ctrl_target_fingertip_pose = torch.cat([ctrl_target_fingert_pos, ctrl_target_fingertip_quat], axis=-1)
        self.robot.move_fingertip(ctrl_target_fingertip_pose.numpy()[0], timeout=5)

    def get_init_wrench_head_pose(self):
        """Tensor"""
        ## Reset nut standard transform
        nut_quat = torch.from_numpy(self.nut_pose[None, 3:7])
        standard_nut_quat = env_utils.calc_feasible_nut_quaternions(nut_quat)[:, 8:12] # (1, 4)
        self.standard_nut_quat_local = torch_jit_utils.quat_mul(torch_jit_utils.quat_conjugate(nut_quat), standard_nut_quat)
        nut_pos = torch.from_numpy(self.nut_pose[None, :3])
        nut_quat = standard_nut_quat.clone()

        ## Compute the fingertip goal pose
        wrench_head_goal_pos_local = self.wrench_head_goal_pos_local.clone()
        # wrench_head_goal_pos_local[:, 2] += 0.018391 + 0.01
        # wrench_head_goal_pos_local[:, 2] += 0.018391 + 0.006 # industreal
        wrench_head_goal_pos_local[:, 2] += 0.018391 + 0.002 # industreal
        # wrench_head_goal_pos_local[:, :2] += (2 * torch.rand((1, 2)) - 0.5) * 0.01
        # wrench head goal pose
        wrench_head_goal_quat, wrench_head_goal_pos = torch_jit_utils.tf_combine(
            nut_quat,
            nut_pos,
            self.wrench_head_goal_quat_local.clone(),
            wrench_head_goal_pos_local
        )
        return wrench_head_goal_pos, wrench_head_goal_quat

    def move_gripper_to_init_wrench_pose(self, timeout: int=100):
        """Tensor"""
        wrench_head_goal_pos, wrench_head_goal_quat = self.get_init_wrench_head_pose()

        fingertip_goal_quat, fingertip_goal_pos = self.wrench_head_to_fingertip(wrench_head_goal_quat, wrench_head_goal_pos)
        self.ctrl_target_fingert_pos = fingertip_goal_pos.clone()
        self.ctrl_target_fingertip_quat = fingertip_goal_quat.clone()
        
        ## Move gripper to the goal pose
        fingertip_goal_pose = torch.cat([fingertip_goal_pos, fingertip_goal_quat], axis=-1)
        self.robot.move_fingertip(fingertip_goal_pose.numpy()[0], timeout=timeout)

    def rotate_wrench_head_on_nut(self, timeout: int = 30, z_offset: float = 0.0, counter_clockwise: bool = True):
        total_deg = 65
        delta_deg = 10

        # Capture the initial nut pose once
        initial_nut_pos, initial_nut_quat = self.get_standard_nut_pose()

        # Initial wrench head pose relative to the initial nut frame
        wrench_head_goal_pos_local = self.wrench_head_goal_pos_local.clone()
        wrench_head_goal_pos_local[:, 2] += z_offset  # Adjust along the nut's z-axis

        # Initial orientation without any rotation
        initial_wrench_head_quat_local = self.wrench_head_goal_quat_local.clone()

        for i in range(int(total_deg / delta_deg)):
            # Calculate cumulative rotation angle relative to initial frame
            current_deg = (i + 1) * delta_deg  # Total rotation so far
            if not counter_clockwise:
                current_deg = -current_deg

            # Create rotation quaternion for the cumulative angle (around nut's z-axis)
            roll = torch.tensor([current_deg * torch.pi / 180.0])
            pitch = torch.tensor([0.0])
            yaw = torch.tensor([0.0])
            delta_quat = torch_jit_utils.quat_from_euler_xyz(roll, pitch, yaw)

            # Apply cumulative rotation to initial wrench orientation
            wrench_head_quat_local = torch_jit_utils.quat_mul(
                initial_wrench_head_quat_local,
                delta_quat
            )

            # Compute goal pose in base frame using initial nut pose
            wrench_head_quat, wrench_head_pos = torch_jit_utils.tf_combine(
                initial_nut_quat,
                initial_nut_pos,
                wrench_head_quat_local,
                wrench_head_goal_pos_local
            )

            # Convert to fingertip pose and execute
            fingertip_quat, fingertip_pos = self.wrench_head_to_fingertip(wrench_head_quat, wrench_head_pos)
            fingertip_pose = torch.cat([fingertip_pos, fingertip_quat], axis=-1)
            self.robot.move_fingertip(fingertip_pose.numpy()[0], timeout=timeout)

    def lift_wrench_head_recover(self, num_steps: int=25, update_progress: bool=False):
        # actions = torch.tensor([[0.0, 0.0, -1.0, 0.0, 0.0, 0.0]])
        # for _ in range(10):
        #     self.step(actions, update_progress=update_progress)
        actions = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        for _ in range(num_steps):
            self.step(actions, update_progress=update_progress)
            # self.apply_object_actions_as_ctrl_targets(actions, do_scale=True, regularize_with_force=True)
    def lift_wrench_head(self, num_steps: int = 25, update_progress: bool = False):
        # Get stationary nut's frame once at start
        nut_pos, nut_quat = self.get_standard_nut_pose()

        wrench_pos, wrench_quat = self.get_wrench_head_pose()
        
        # Extract nut's Z-axis in world coordinates
        rot_matrix = torch_jit_utils.quaternion_to_matrix(nut_quat[..., [3, 0, 1, 2]])
        nut_z_axis = rot_matrix[..., 2]  # Third column is Z-axis (1, 3)

        # Initial downward movement to disengage threads
        self.step(torch.tensor([[0.0, 0.0, -1.0, 0.0, 0.0, 0.0]]), update_progress)

        # Main lifting sequence
        # for _ in range(num_steps):
        while wrench_pos[:, 2] - nut_pos[:, 2] < 0.03:
            # 1. Get current wrench head orientation
            wrench_pos, wrench_quat = self.get_wrench_head_pose()
            
            # 2. Calculate desired direction in wrench's local frame
            inv_wrench_rot = torch_jit_utils.quat_conjugate(wrench_quat)
            lift_dir_local = torch_jit_utils.quat_apply(inv_wrench_rot, nut_z_axis)
            
            # 3. Create normalized movement command
            action_translation = lift_dir_local / torch.norm(lift_dir_local)
            action = torch.cat([action_translation, torch.zeros(3).repeat((action_translation.size(0), 1))], dim=-1)
            # ic(action[:, :3])
            
            # 4. Execute with smaller steps
            # self.step(action, update_progress)
            self.apply_object_actions_as_ctrl_targets(action, do_scale=True, regularize_with_force=True)

    ## Transformation Functions ##
    def wrench_head_to_fingertip(self, wrench_head_goal_quat: torch.Tensor, wrench_head_goal_pos: torch.Tensor):
        """Tensor"""
        ## Current wrench head pose
        # wrench_head_quat = torch.from_numpy(self.wrench_head_pose[None, 3:7]) # (1, 4)
        # wrench_head_pos = torch.from_numpy(self.wrench_head_pose[None, :3]) # (1, 3)
        wrench_head_pos, wrench_head_quat = self.get_wrench_head_pose()
        ## Current fingertip pose
        robot_state = self.robot.get_robot_state()
        fingertip_quat = torch.from_numpy(robot_state["fingertip_pose"][None, 3:7])
        fingertip_pos = torch.from_numpy(robot_state["fingertip_pose"][None, :3])
        ## Fingertip to wrench head transformation
        wrench_head_quat_inv, wrench_head_pos_inv = torch_jit_utils.tf_inverse(wrench_head_quat, wrench_head_pos)
        fingertip_to_wrench_head_quat, fingertip_to_wrench_head_pos = torch_jit_utils.tf_combine(
            wrench_head_quat_inv,
            wrench_head_pos_inv,
            fingertip_quat,
            fingertip_pos
        )

        ## Apply to target wrench head pose
        fingertip_goal_quat, fingertip_goal_pos = torch_jit_utils.tf_combine(
            wrench_head_goal_quat,
            wrench_head_goal_pos,
            fingertip_to_wrench_head_quat,
            fingertip_to_wrench_head_pos
        )
        return fingertip_goal_quat, fingertip_goal_pos
    
    def fingertip_to_wrench_head(self, fingertip_goal_quat: torch.Tensor, fingertip_goal_pos: torch.Tensor):
        """Tensor"""
        ## Current wrench head pose
        # wrench_head_quat = torch.from_numpy(self.wrench_head_pose[None, 3:7]) # (1, 4)
        # wrench_head_pos = torch.from_numpy(self.wrench_head_pose[None, :3]) # (1, 3)
        wrench_head_pos, wrench_head_quat = self.get_wrench_head_pose()
        ## Current fingertip pose
        robot_state = self.robot.get_robot_state()
        fingertip_quat = torch.from_numpy(robot_state["fingertip_pose"][None, 3:7])
        fingertip_pos = torch.from_numpy(robot_state["fingertip_pose"][None, :3])
        ## Wrench head to fingertip transformation
        fingertip_quat_inv, fingertip_pos_inv = torch_jit_utils.tf_inverse(fingertip_quat, fingertip_pos)
        wrench_head_to_fingertip_quat, wrench_head_to_fingertip_pos = torch_jit_utils.tf_combine(
            fingertip_quat_inv,
            fingertip_pos_inv,
            wrench_head_quat,
            wrench_head_pos
        )

        ## Apply to target fingertip pose
        wrench_head_goal_quat, wrench_head_goal_pos = torch_jit_utils.tf_combine(
            fingertip_goal_quat,
            fingertip_goal_pos,
            wrench_head_to_fingertip_quat,
            wrench_head_to_fingertip_pos
        )
        return wrench_head_goal_quat, wrench_head_goal_pos
    
    ## Visualization ##
    def draw_poses_in_base(self, rgb: np.ndarray, poses_in_base: torch.Tensor, draw_bbox: bool=False) -> np.ndarray:
        r"""Tensor
        rgb: (h, w, 3)
        poses_in_base: (N, 7)
        ---
        return: (h, w, 3) rgb
        """
        vis = rgb.copy()

        Ts_in_base = np.zeros((poses_in_base.shape[0], 4, 4), dtype=np.float32)
        Ts_in_base[:, -1, -1] = 1.
        Ts_in_base[:, :3, :3] = R.from_quat(poses_in_base[:, 3:7]).as_matrix()
        Ts_in_base[:, :3, -1] = poses_in_base[:, :3]
        ## transform poses to camera frame
        Ts_in_cam = np.linalg.inv(self.cfg["cam_T_base"]) @ Ts_in_base # (N, 4, 4)

        ## visualize poses
        for T_in_cam in Ts_in_cam:
            vis = visualize_pose_2d(vis, T_in_cam, self.cam_K, draw_bbox=draw_bbox)

        return vis

    ## Pose Callbacks ##
    def nut_callback(self, msg):
        self.nut_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w], dtype=np.float32)
    def wrench_head_callback(self, msg):
        self.wrench_head_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w], dtype=np.float32)
    # def wrench_callback(self, msg):
    #     self.wrench_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w], dtype=np.float32)

    ## Camera Callbacks ##
    def rgb_callback(self, msg):
        self.rgb = self.bridge.imgmsg_to_cv2(msg, "rgb8") # 8-bit rgb, (h, w, 3)
    def cam_info_callback(self, msg):
        self.cam_K = np.array(msg.K, dtype=np.float32).reshape(3, 3)

if __name__ == "__main__":
    # define ros node
    rospy.init_node('env', anonymous=True)

    cam_T_base = np.loadtxt("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Vision/cam_T_base.txt", dtype=np.float32)

    cfg = {
        # "pos_action_scale": [0.004, 0.004, 0.002],
        # "rot_action_scale": [0.0005, 0.0005, 0.0005],
        "pos_action_scale": [0.001, 0.001, 0.001],
        "rot_action_scale": [0.01, 0.01, 0.01],
        "clamp_rot": True,
        "clamp_rot_thresh": 1.0e-6,
        "max_episode_length": 256,
        "cam_T_base": cam_T_base,
        "action_as_object_displacement": True,
        "observation_type": 1,
    }
    env = KukaWrenchInsertEnv(cfg)
    env.wait_for_subscribers(seconds=10)
    
    obs = env.reset()
    input("Press Enter to continue inserting...")
    is_done = False
    while not is_done:
        actions = torch.tensor([[0.0, 0.0, 0.0, 0.0, -1.0, 0.0]])
        obs, is_done = env.step(actions)

        """
        nut_pos, nut_quat = env.get_standard_nut_pose()
        wrench_head_quat, wrench_head_pos = torch_jit_utils.tf_combine(
            nut_quat,
            nut_pos,
            obs[:, 3:7],
            obs[:, :3]
        )
        wrench_head_goal_quat, wrench_head_goal_pos = torch_jit_utils.tf_combine(
            nut_quat,
            nut_pos,
            obs[:, 10:14],
            obs[:, 7:10]
        )
        vis = env.rgb.copy()
        vis = env.draw_poses_in_base(vis, torch.cat([wrench_head_pos, wrench_head_quat], dim=1), draw_bbox=True)
        vis = env.draw_poses_in_base(vis, torch.cat([wrench_head_goal_pos, wrench_head_goal_quat], dim=1), draw_bbox=False)
        # cv2.imshow("vis", vis[::-1, ::-1, ::-1])
        # cv2.waitKey(1)
        """
