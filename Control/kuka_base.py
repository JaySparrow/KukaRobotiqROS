#!/usr/bin/env python

import rospy
from robotiq_3f_gripper_services.srv import ChangeMode
from robotiq_3f_gripper_services.srv import MoveAngle
from geometry_msgs.msg import WrenchStamped, PoseStamped
from iiwa_msgs.msg import JointPosition
from std_msgs.msg import Time

from control_utils import transform_pose_gripper_fingertip_to_eef, transform_pose_eef_to_gripper_fingertip

import numpy as np

class KukaBase:
    def __init__(self):
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

        # eef force feedback subscriber
        rospy.Subscriber("/iiwa/state/CartesianWrench", WrenchStamped, self.eef_wrench_callback)
        self.current_eef_force_torque = None

    def is_subscriber_ready(self) -> bool:
        return (
            self.current_eef_pose is not None and 
            self.current_joint_pos is not None and 
            self.current_eef_force_torque is not None
        )

    def wait_for_subscribers(self, seconds: int=5):
        sub_seconds = 1
        num_iter = int(seconds / sub_seconds)
        for _ in range(num_iter):
            if self.is_subscriber_ready():
                print("Subscribers are ready!")
                return
            rospy.sleep(sub_seconds)
        print(f"Subscribers are not responding within {seconds} seconds!")
        exit(0)

    ### STATE ###
    def get_robot_state(self):
        return {
            "eef_pose": self.current_eef_pose.astype(np.float32).copy(),
            "fingertip_pose": self.current_fingertip_pose.astype(np.float32).copy(),
            "joint_pos": self.current_joint_pos.astype(np.float32).copy(),
            "eef_force_torque": self.current_eef_force_torque.astype(np.float32).copy()
        }

    ### TASK-SPACE CONTROL ###
    def move_eef(self, target_pose: np.ndarray, timeout: int=-1) -> list:
        assert len(target_pose) in {3, 4, 7}, f"target_pose = {target_pose} is invalid"

        current_pos, current_quat = self.current_eef_pose[:3], self.current_eef_pose[3:7]
        if len(target_pose) == 3:
            target_pose = np.concatenate([target_pose, current_quat])
        elif len(target_pose) == 4:
            target_pose = np.concatenate([current_pos, target_pose])

        self.publish_target_pose(target_pose, timeout=timeout)
        return self.current_eef_pose.copy()
    
    def move_fingertip(self, target_pose: np.ndarray, timeout: int=-1) -> list:
        assert len(target_pose) in {3, 4, 7}, f"target_pose = {target_pose} is invalid"

        current_pos, current_quat = self.current_fingertip_pose[:3], self.current_fingertip_pose[3:7]
        if len(target_pose) == 3:
            target_pose = np.concatenate([target_pose, current_quat])
        elif len(target_pose) == 4:
            target_pose = np.concatenate([current_pos, target_pose])

        target_eef_pose = transform_pose_gripper_fingertip_to_eef(target_pose)
        self.publish_target_pose(target_eef_pose, timeout=timeout)
        return self.current_fingertip_pose.copy()
    
    ### GRIPPER CONTROL ###
    def change_gripper_mode(self, mode: str):
        
        assert isinstance(mode, str) and mode in {'o', 'w', 'p', 'b', 's', 'r'}
        resp = self.gripper_change_mode_fn(mode)
        return resp

    def move_gripper(self, deg: int):
        
        deg = max(min(deg, 250), 0)
        resp = self.gripper_move_fn(deg)
        return resp

    ### PUBLISHERS ###
    def publish_target_pose(self, pose, timeout: int=-1):
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

        if timeout > 0:
            try:
                rospy.wait_for_message("/iiwa/state/DestinationReached", Time, timeout=timeout)
            except:
                self.publish_target_pose(self.current_eef_pose, timeout=-1)
                # print("Command timeout! Stopped!")
                # exit(0)

        return True
        
    ### CALLBACKS ###
    def eef_pose_callback(self, msg: PoseStamped):
        self.current_eef_pose = np.array([
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, 
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
        ]) # [x, y, z, qx, qy, qz, qw]
        self.current_fingertip_pose = transform_pose_eef_to_gripper_fingertip(self.current_eef_pose)

    def joint_pos_callback(self, msg: JointPosition):
        self.current_joint_pos = np.array(
            [msg.position.a1, msg.position.a2, msg.position.a3, msg.position.a4, msg.position.a5, msg.position.a6, msg.position.a7]
        ) # [j1, j2, j3, j4, j5, j6, j7]

    def eef_wrench_callback(self, msg: WrenchStamped):
        self.current_eef_force_torque = np.array([
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, 
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        ])

if __name__ == "__main__":
    from control_utils import trajectory_interpolate
    # define ros node
    rospy.init_node('kuka_base', anonymous=True)

    agent = KukaBase()
    agent.wait_for_subscribers(seconds=5)

    start_pose = agent.current_eef_pose.copy()
    target_pose = np.array([0.580, -0.204, 0.323, -0.416, 0.600, 0.355, 0.584]) # np.array([0.512817199406, -0.0997344442156, 0.399688303359])
    pose_traj = trajectory_interpolate(start_pose, target_pose, 0.02, 0.2)
    # reached_pose = agent.current_eef_pose
    for pose in pose_traj:
        agent.move_eef(pose, timeout=20)
        agent.move_gripper(245)
        agent.change_gripper_mode("p")

    # agent.move_eef(target_pose, timeout=90)
    # agent.move_fingertip(target_pose, timeout=25)
    # print(agent.get_robot_state())
