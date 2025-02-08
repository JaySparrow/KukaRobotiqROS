#!/usr/bin/env python

import sys
sys.path.append("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Vision/")
sys.path.append("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Policy/")
sys.path.append("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Control/")

from policy import WrenchScrewPolicy
from moveit_impedance_agent_highFreq import Agent
from vision_utils import visualize_pose_2d
from policy_utils import pose_to_T, T_to_pose

import numpy as np
import cv2
from cv_bridge import CvBridge
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo

from numpy_utils import tf_combine2, tf_inverse2

class Pose:
    def __init__(self):
        self.nut2_pose = None
        self.wrench2_head_pose = None
        self.wrench2_pose = None
        rospy.Subscriber("/pose_tracker/nut2", PoseStamped, self.nut2_callback)
        rospy.Subscriber("/pose_tracker/wrench2_head", PoseStamped, self.wrench2_head_callback)
        rospy.Subscriber("/pose_tracker/wrench2", PoseStamped, self.wrench2_callback)

        # rospy.Subscriber("/pose_tracker/nut4", PoseStamped, self.nut2_callback)
        # rospy.Subscriber("/pose_tracker/wrench4_head", PoseStamped, self.wrench2_head_callback)
        # rospy.Subscriber("/pose_tracker/wrench4", PoseStamped, self.wrench2_callback)

    def nut2_callback(self, msg):
        self.nut2_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
    def wrench2_head_callback(self, msg):
        self.wrench2_head_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
    def wrench2_callback(self, msg):
        self.wrench2_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])

class Camera:
    def __init__(self):
        self.rgb = None
        self.cam_K = None
        self.bridge = CvBridge()
        rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.cam_K_callback)
    def rgb_callback(self, msg):
        rgb = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.rgb = rgb
    def cam_K_callback(self, msg):
        self.cam_K = np.array(msg.K).reshape(3, 3)

def get_wrenchHead_to_eef_fn(eef_T_wrenchHead):
    eef_in_wrenchHead_pose = T_to_pose(eef_T_wrenchHead)
    def wrenchHead_to_eef_fn(wrenchHead_pose):
        return tf_combine2(wrenchHead_pose, eef_in_wrenchHead_pose)
    return wrenchHead_to_eef_fn

def get_eef_to_wrenchHead_fn(eef_T_wrenchHead):
    eef_in_wrenchHead_pose = T_to_pose(eef_T_wrenchHead)
    wrenchHead_in_eef_pose = tf_inverse2(eef_in_wrenchHead_pose)
    def eef_to_wrenchHead_fn(eef_pose):
        return tf_combine2(eef_pose, wrenchHead_in_eef_pose)
    return eef_to_wrenchHead_fn

def random_search_fn(var = 0.02, dist_bound = 0.01):
    while True:
        delta = np.random.randn(2) * var
        delta = delta if np.linalg.norm(delta) < dist_bound else delta * dist_bound / np.linalg.norm(delta)
        delta_tf = np.array([0, *delta, 0, 0, 0, 1])
        yield delta_tf

def planar_pattern_search_fn(pattern_waypoint_list, dist_bound = 0.01):
    l = len(pattern_waypoint_list)
    delta_rotation = np.array([0., 0., 0., 1.])
    i = 0
    while True:
        i = i % l
        delta_translation = pattern_waypoint_list[i] * dist_bound
        delta_tf = np.concatenate([delta_translation, delta_rotation])
        yield delta_tf
        i += 1

### PATTERNS ###
WINDMILL = [
    np.array([0., -1., -1.]),
    np.array([0., -1., 0.]),
    np.array([0., 0., 0.]),
    np.array([0., 1., 0.]),
    np.array([0., 1., 1.]),
    np.array([0., 0., 0.]),
    np.array([0., 0., 1.]),
    np.array([0., -1., 1.]),
    np.array([0., 0., 0.]),
    np.array([0., 1., -1.]),
    np.array([0., 0., -1.]),
    np.array([0., 0., 0.]),
]

RASTER = [
    np.array([0., -1., 0.]),
    np.array([0., -1., -0.5]),
    np.array([0., 0., -0.5]),
    np.array([0., 1., -0.5]),
    np.array([0., 1., -1.]),
    np.array([0., 0., -1.]),

    np.array([0., -1., -1.]),
    np.array([0., -1., -0.5]),
    np.array([0., 0., -0.5]),
    np.array([0., 1., -0.5]),
    np.array([0., 1., 0.]),
    np.array([0., 0., 0.]),
]

is_debug = False
rospy.init_node('pseudo_exec', anonymous=True)
pose_collect = Pose()
frame_collect = Camera()

cam_T_base = np.loadtxt("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Vision/cam_T_base.txt", dtype=np.float64)

policy = WrenchScrewPolicy()
agent = Agent(use_gripper=True)

search_cnt = 0
try:
    done = False
    last_state = ""
    while agent.current_eef_pose is None:
        rospy.sleep(0.1)
    while pose_collect.nut2_pose is None or pose_collect.wrench2_head_pose is None:
        rospy.sleep(0.1)
    while frame_collect.rgb is None or frame_collect.cam_K is None:
        rospy.sleep(0.1)
    print("Start...")
    policy.set_init_state("post wrench")
    while not done:

        # get the current object poses
        nut_T_base = pose_to_T(pose_collect.nut2_pose)
        wrenchHead_T_base = pose_to_T(pose_collect.wrench2_head_pose)
        wrench_T_base = pose_to_T(pose_collect.wrench2_pose)

        # # (pseudo) execute the action
        current_eef_pose = agent.current_eef_pose
        current_wrenchHead_T_base = wrenchHead_T_base.copy()
        current_wrench_T_base = wrench_T_base.copy()

        # step policy
        target_eef_pose, done = policy.act(current_eef_pose, nut_T_base, current_wrenchHead_T_base, current_wrench_T_base)
        target_wrenchHead_pose = T_to_pose(pose_to_T(target_eef_pose) @ np.linalg.inv(policy.eef_T_wrenchHead))

        # visualize the target pose
        vis = frame_collect.rgb.copy()
        if policy.state in ["post wrench", "over nut", "on nut", "rotate", "lift", "reset wrench"]:
            wrenchHead_T_cam = np.linalg.inv(cam_T_base) @ pose_to_T(target_wrenchHead_pose)
            vis = visualize_pose_2d(vis, wrenchHead_T_cam, frame_collect.cam_K, draw_bbox=True)

        eef_T_cam = np.linalg.inv(cam_T_base) @ pose_to_T(target_eef_pose)
        vis = visualize_pose_2d(vis, eef_T_cam, frame_collect.cam_K, draw_bbox=True)
        vis = vis[::-1, ::-1, ...].copy()
        cv2.putText(
                    img=vis,
                    text=f"state: {policy.state}",
                    org=(20, 20), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 255, 0),
                    thickness=2, 
                    lineType=2
                )
        cv2.putText(
                    img=vis,
                    text=f"target eef pose: [{' '.join(map(str, target_eef_pose.round(decimals=2)))}]",
                    org=(20, 50), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 255, 0),
                    thickness=2, 
                    lineType=2
                )
        cv2.imshow("vis", vis[..., ::-1])
        if is_debug:
            cv2.waitKey(0)
        else:
            cv2.waitKey(2000)

        if policy.state == "post wrench":
            agent.move_gripper(deg=240)
            rospy.sleep(2.0)

        if policy.state == "on nut":
            reached_pose, is_stuck = agent.move_eef(target_eef_pose, regulate_with_force=True)
            search_cnt = 0
            wrenchHead_to_eef_fn = get_wrenchHead_to_eef_fn(policy.eef_T_wrenchHead)
            eef_to_wrenchHead_fn = get_eef_to_wrenchHead_fn(policy.eef_T_wrenchHead)
            search_fn = random_search_fn(var=0.02, dist_bound=0.015)
            # search_fn = planar_pattern_search_fn(RASTER, dist_bound=0.015)
            nut_position = pose_collect.nut2_pose[:3]
            reached_wrenchHead_position = eef_to_wrenchHead_fn(reached_pose)[:3]
            while np.abs(nut_position[2] - reached_wrenchHead_position[2]) > 0.01:
                # wrenchHead_to_eef_fn = get_wrenchHead_to_eef_fn(policy.eef_T_wrenchHead)
                # eef_to_wrenchHead_fn = get_eef_to_wrenchHead_fn(policy.eef_T_wrenchHead)

                if is_stuck:
                    search_target_wrenchHead_pose = eef_to_wrenchHead_fn(reached_pose)
                    search_target_wrenchHead_pose[2] += 0.005
                    search_target_eef_pose = wrenchHead_to_eef_fn(search_target_wrenchHead_pose)
                    print(f"[stuck] move_eef stopped moving, lift!")
                    reached_pose, is_stuck = agent.move_eef(search_target_eef_pose, regulate_with_force=False)
                    # is_stuck = False
                else:
                    search_target_wrenchHead_pose = eef_to_wrenchHead_fn(target_eef_pose)
                    delta_tf = next(search_fn)
                    search_target_wrenchHead_pose = tf_combine2(search_target_wrenchHead_pose, delta_tf)
                    search_target_eef_pose = wrenchHead_to_eef_fn(search_target_wrenchHead_pose)

                    print(f"[exec] search pose {search_cnt}...")
                    reached_pose, is_stuck = agent.move_eef(search_target_eef_pose, regulate_with_force=True)
                    search_cnt += 1
                # nut_position = pose_collect.nut2_pose[:3]
                reached_wrenchHead_position = eef_to_wrenchHead_fn(reached_pose)[:3]
                # reached_wrenchHead_position = pose_collect.wrench2_head_pose[:3]
            
            reached_pose, is_stuck = agent.move_eef(target_eef_pose, regulate_with_force=True)
            # print(f"\n\n\nreached_pose: {reached_pose}\nsearch_target: {search_target_eef_pose}\ntarget: {target_eef_pose}\n\n\n")
            # exit(0)
        else:
            reached_pose, is_stuck = agent.move_eef(target_eef_pose, regulate_with_force=False)

        if policy.state == "put wrench":
            agent.move_gripper(deg=45)
            rospy.sleep(2.0)

except KeyboardInterrupt:
    print("[Interrupt] number of trials for random insertion search: ", search_cnt)

finally:
    print("[Terminate] number of trials for random insertion search: ", search_cnt)
