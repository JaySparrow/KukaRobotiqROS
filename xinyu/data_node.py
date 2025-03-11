#!/usr/bin/env python
import sys
sys.path.append("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Vision/")
sys.path.append("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Policy/")
sys.path.append("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Control/")

import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from iiwa_msgs.msg import JointPosition
from robotiq_3f_gripper_articulated_msgs.msg import Robotiq3FGripperRobotInput

import cv2
from cv_bridge import CvBridge
import numpy as np
import os
import pickle

from control_utils import transform_pose_eef_to_gripper_fingertip

class Recorder:
    def __init__(self):
        ## Camera Subscribers ##
        rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.cam_info_callback)
        self.rgb = None
        self.depth = None
        self.cam_K = None
        self.bridge = CvBridge()

        ## READING SUBSCRIBERS ##
        # current eef pose
        rospy.Subscriber("/iiwa/state/CartesianPose", PoseStamped, self.eef_pose_callback)
        self.current_eef_pose = None
        self.current_fingertip_pose = None
        rospy.Subscriber("/iiwa/state/JointPosition", JointPosition, self.joint_pos_callback)
        self.current_joint_pos = None
        rospy.Subscriber('/Robotiq3FGripperRobotInput', Robotiq3FGripperRobotInput, self.gripper_callback)
        self.gripper_value = None

    ## Camera Callbacks ##
    def rgb_callback(self, msg):
        self.rgb = self.bridge.imgmsg_to_cv2(msg, "rgb8") # 8-bit rgb, (h, w, 3)
    def depth_callback(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg, "16UC1").astype(np.uint16) # 16-bit unsigned, (h, w)
    def cam_info_callback(self, msg):
        self.cam_K = np.array(msg.K, dtype=np.float32).reshape(3, 3)

    ## Robot Callbacks ##
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
    def gripper_callback(self, data):
        self.gripper_value = np.array([data.gPOA, data.gPOB, data.gPOC]) # [g1, g2, g3]

    def is_subscriber_ready(self) -> bool:
        return (
            self.current_eef_pose is not None and 
            self.current_fingertip_pose is not None and
            self.current_joint_pos is not None and 
            self.gripper_value is not None and
            self.rgb is not None and
            self.depth is not None and
            self.cam_K is not None
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

    def run(self, save_dir: str, sleep_sec: float=2):
        rospy.init_node("data_node", anonymous=True)
        # rate = rospy.Rate(freq)
        self.wait_for_subscribers(seconds=5)

        if os.path.exists(save_dir):
            if input(f"Directory {save_dir} exists. Overwrite? (y/n): ").lower() != "y":
                exit(0)
            else:
                os.system(f"rm -rf {save_dir}")
        os.makedirs(save_dir)
        rgb_dir = os.path.join(save_dir, "rgb")
        depth_dir = os.path.join(save_dir, "depth")
        os.makedirs(rgb_dir)
        os.makedirs(depth_dir)

        eef_pose_list = []
        fingertip_pose_list = []
        joint_pos_list = []
        gripper_value_list = []
        i = 0
        try:
            while not rospy.is_shutdown():
                # save rgb
                rgb_path = os.path.join(rgb_dir, f"{i}.png")
                cv2.imwrite(rgb_path, self.rgb.copy()[..., ::-1])
                # save depth
                depth_path = os.path.join(depth_dir, f"{i}.png")
                cv2.imwrite(depth_path, self.depth.copy())
                # record data
                eef_pose_list.append(self.current_eef_pose.copy())
                fingertip_pose_list.append(self.current_fingertip_pose.copy())
                joint_pos_list.append(self.current_joint_pos.copy())
                gripper_value_list.append(self.gripper_value.copy())
                i += 1

                if i % 100 == 0:
                    print(f"[data_node] {i} steps collected!")
                    data = {
                        "save_freq": 1/sleep_sec,
                        "num_steps": i,
                        "cam_K": self.cam_K,
                        "eef_pose_list": eef_pose_list,
                        "fingertip_pose_list": fingertip_pose_list,
                        "joint_pos_list": joint_pos_list,
                        "gripper_value_list": gripper_value_list
                    }
                    with open(os.path.join(save_dir, "data.pkl"), "wb") as f:
                        pickle.dump(data, f)
                #rospy.sleep(sleep_sec)
        finally:
            print(f"[data_node] total {i} steps collected!")
            data = {
                "save_freq": 1/sleep_sec,
                "num_steps": i,
                "cam_K": self.cam_K,
                "eef_pose_list": eef_pose_list,
                "fingertip_pose_list": fingertip_pose_list,
                "joint_pos_list": joint_pos_list,
                "gripper_value_list": gripper_value_list
            }
            with open(os.path.join(save_dir, "data.pkl"), "wb") as f:
                pickle.dump(data, f)

if __name__ == "__main__":
    sleep_sec = 2.0
    save_dir = "/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/xinyu/test_data"
    args = sys.argv
    if len(args) > 1:
        save_dir = args[1]
    if len(args) > 2:
        sleep_sec = float(args[2])

    recorder = Recorder()
    recorder.run(save_dir, sleep_sec)
