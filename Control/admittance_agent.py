#!/usr/bin/env python

import rospy
from robotiq_3f_gripper_services.srv import ChangeMode
from robotiq_3f_gripper_services.srv import MoveAngle
from geometry_msgs.msg import WrenchStamped, PoseStamped
from iiwa_msgs.msg import JointPosition
from std_msgs.msg import Time

from control_utils import transform_pose_gripper_fingertip_to_eef, transform_pose_eef_to_gripper_fingertip

import numpy as np
import time

from scipy.spatial.transform import Rotation as R

class KukaBase:
    # def __init__(self, pos_action_scale, rot_action_scale, clamp_rot: bool, clamp_rot_thresh: float):
    def __init__(self, ctrl_cfg: dict):
        self.pos_action_scale = np.array(ctrl_cfg["pos_action_scale"])
        self.rot_action_scale = np.array(ctrl_cfg["rot_action_scale"])
        self.clamp_rot = ctrl_cfg["clamp_rot"]
        self.clamp_rot_thresh = ctrl_cfg["clamp_rot_thresh"]
        ## CONTROL SERVICES ##
        # Gripper state
        rospy.wait_for_service("/robotiq_3f/ActivateGripper", timeout=5)
        self.gripper_change_mode_fn = rospy.ServiceProxy("/robotiq_3f/ChangeMode", ChangeMode)
        self.gripper_move_fn = rospy.ServiceProxy("/robotiq_3f/MoveGripper", MoveAngle)
        self.default_open = 20
        self.default_close = 65

        ## CONTROL PUBLIHSERS ##
        # End-effector Cartesian pose publisher
        self.eef_pose_pub = rospy.Publisher("/iiwa/command/CartesianPose", PoseStamped)

        ## READING SUBSCRIBERS ##
        # current eef pose
        rospy.Subscriber("/iiwa/state/CartesianPose", PoseStamped, self.eef_pose_callback)
        self.current_eef_pose = None
        self.current_fingertip_pose = None
        rospy.Subscriber("/iiwa/state/JointPosition", JointPosition, self.joint_pos_callback)
        self.current_joint_pos = None

        # eef force feedback subscriber TODO: consider the torque as well
        rospy.Subscriber("/iiwa/state/CartesianWrench", WrenchStamped, self.eef_wrench_callback)
        self.current_eef_force_torque = None
        self.reference_eef_force_torque = None
        self.eef_delta_force_torque_thresh = np.array([4, 4, 3.0, 0.3, 0.3, 0.4]) # np.array([7.0, 7.0, 7.0, 1.0, 1.0, 1.0])

    def is_subscriber_ready(self) -> bool:
        return self.current_eef_pose is not None and self.current_joint_pos is not None and self.current_eef_force_torque is not None

    def wait_for_subscriber(self, seconds: int=5):
        sub_seconds = 1
        num_iter = int(seconds / sub_seconds)
        for _ in range(num_iter):
            if self.is_subscriber_ready():
                self.reset_eef_force_torque()
                print("Subscribers are ready!")
                return
            rospy.sleep(sub_seconds)
        print(f"Subscribers are not responding within {seconds} seconds!")
        exit(0)

    def reset_eef_force_torque(self):
        self.reference_eef_force_torque = self.current_eef_force_torque.copy()

    ### STATE ###
    def get_robot_state(self):
        return {
            "eef_pose": self.current_eef_pose.copy(),
            "fingertip_pose": self.current_fingertip_pose.copy(),
            "joint_pos": self.current_joint_pos.copy(),
            "eef_force_torque": self.current_eef_force_torque.copy()
        }

    ### DELTA CONTROL ###
    def apply_actions_as_ctrl_targets(self, actions: np.ndarray, do_scale: bool, regularize_with_force: bool=True):

        ## Regularize with force
        if regularize_with_force:
            delta_force_torque = np.abs(self.current_eef_force_torque - self.reference_eef_force_torque) # [f_x, f_y, f_z, t_x, t_y, t_z]
            action_regularizer = (delta_force_torque < self.eef_delta_force_torque_thresh).astype(np.float32)
            actions = actions * action_regularizer
            print(np.nonzero(actions))

        pos_actions, rot_actions = actions[:3], actions[3:6]
        ## Scale
        if do_scale:
            pos_actions = pos_actions * self.pos_action_scale
            rot_actions = rot_actions * self.rot_action_scale

        current_pos, current_quat = self.current_fingertip_pose[:3].copy(), self.current_fingertip_pose[3:7].copy()
        ## Position
        target_fingertip_pos = current_pos + pos_actions

        ## Rotation
        angle = np.linalg.norm(rot_actions, axis=-1, keepdims=True) # (1, )
        rot_actions_quat = R.from_rotvec(rot_actions).as_quat()
        if self.clamp_rot:
            rot_actions_quat = np.where(angle > self.clamp_rot_thresh,
                                           rot_actions_quat,
                                           np.array([0.0, 0.0, 0.0, 1.0]))
            
        target_fingertip_quat = ( R.from_quat(rot_actions_quat) * R.from_quat(current_quat) ).as_quat()
        
        ## Apply
        target_fingertip_pose = np.concatenate([target_fingertip_pos, target_fingertip_quat], axis=-1)
        self.move_fingertip(target_fingertip_pose, is_block=True)

    ### TASK-SPACE CONTROL ###
    def move_eef(self, target_pose: np.ndarray, is_block: bool=False) -> list:
        assert len(target_pose) in {3, 4, 7}, f"target_pose = {target_pose} is invalid"

        current_pos, current_quat = self.current_eef_pose[:3], self.current_eef_pose[3:7]
        if len(target_pose) == 3:
            target_pose = np.concatenate([target_pose, current_quat])
        elif len(target_pose) == 4:
            target_pose = np.concatenate([current_pos, target_pose])

        self.publish_target_pose(target_pose, is_block=is_block)
        return self.current_eef_pose.copy()
    
    def move_fingertip(self, target_pose: np.ndarray, is_block: bool=False) -> list:
        assert len(target_pose) in {3, 4, 7}, f"target_pose = {target_pose} is invalid"

        current_pos, current_quat = self.current_fingertip_pose[:3], self.current_fingertip_pose[3:7]
        if len(target_pose) == 3:
            target_pose = np.concatenate([target_pose, current_quat])
        elif len(target_pose) == 4:
            target_pose = np.concatenate([current_pos, target_pose])

        target_eef_pose = transform_pose_gripper_fingertip_to_eef(target_pose)
        self.publish_target_pose(target_eef_pose, is_block=is_block)
        return self.current_fingertip_pose.copy()
    
    ### GRIPPER CONTROL ###

    def change_gripper_mode(self, mode: str):
        if not self.use_gripper:
            return False
        
        assert isinstance(mode, str) and mode in {'o', 'w', 'p', 'b', 's', 'r'}
        resp = self.gripper_change_mode_fn(mode)
        return resp

    def move_gripper(self, deg: int):
        if not self.use_gripper:
            return False
        
        deg = max(min(deg, 250), 0)
        resp = self.gripper_move_fn(deg)
        return resp

    ### PUBLISHERS ###

    def publish_target_pose(self, pose, is_block: bool=True):
        if isinstance(pose, np.ndarray):
            pose = pose.tolist()
        
        out_msg = PoseStamped()
        out_msg.pose.position.x = pose[0]
        out_msg.pose.position.y = pose[1]
        out_msg.pose.position.z = pose[2]
        out_msg.pose.orientation.x = pose[3]
        out_msg.pose.orientation.y = pose[4]
        out_msg.pose.orientation.z = pose[5]
        out_msg.pose.orientation.w = pose[6]

        self.eef_pose_pub.publish(out_msg)

        if is_block:
            rospy.wait_for_message("/iiwa/state/DestinationReached", Time, timeout=10)

        return True
        
    ### CALLBACKS ###
    def eef_pose_callback(self, msg: PoseStamped):
        self.current_eef_pose = np.array([
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, 
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
        ]) # [x, y, z, qx, qy, qz, qw]
        self.current_fingertip_pose = transform_pose_eef_to_gripper_fingertip(self.current_eef_pose)

    def joint_pos_callback(self, msg: JointPosition):
        self.current_joint_pos = np.array(msg.position) # [j1, j2, j3, j4, j5, j6, j7]

    def eef_wrench_callback(self, msg: WrenchStamped):
        self.current_eef_force_torque = np.array([
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, 
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        ])

# if __name__ == "__main__":
#     # define ros node
#     rospy.init_node('admittance_agent', anonymous=True)

#     agent = KukaBase()
#     agent.wait_for_subscriber(seconds=5)

#     target_pose = np.array([0.491817795854, -0.157159973352, 0.399188107074])
#     reached_pose = agent.current_eef_pose

#     agent.move_fingertip(target_pose)

if __name__ == "__main__":
    # define ros node
    rospy.init_node('admittance_agent', anonymous=True)

    ctrl_cfg = {
        # "pos_action_scale": [0.004, 0.004, 0.002],
        # "rot_action_scale": [0.0005, 0.0005, 0.0005],
        "pos_action_scale": [0.003, 0.003, 0.003],
        "rot_action_scale": [0.01, 0.01, 0.01],
        "clamp_rot": True,
        "clamp_rot_thresh": 1.0e-6
    }

    agent = KukaBase(ctrl_cfg)
    agent.wait_for_subscriber(seconds=5)

    actions = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for _ in range(256*3):
        agent.apply_actions_as_ctrl_targets(actions, do_scale=True, regularize_with_force=True)
