#!/usr/bin/env python
import sys
sys.path.append("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Vision/")

# ros
import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
# lib
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation as R
# local
from mask_select import MaskSelect
from bbox_select import BBoxSelect
from pose_tracker_zmq_client import PoseTrackerZMQClient
from vision_utils import visualize_pose_2d

import time

time_dict = dict()
class PoseTrackerRosPublisher:
    def __init__(self, object_names: list, 
                 cam_T_base: np.ndarray=np.eye(4, dtype=np.float64), 
                 sam_checkpoint: str="/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Vision/sam2_b.pt",
                 server_ip: str="172.16.71.50", 
                 port: str="8081"):
        
        ## initialize classes
        # self.mask_selecter = MaskSelect(sam_checkpoint)
        self.mask_selecter = BBoxSelect()
        self.pose_tracker_client = PoseTrackerZMQClient(server_ip=server_ip, port=port)

        self.cam_K = self.receive_camera_intrinsics()
        print(f"---\nCamera intrinsics:\n{self.cam_K}")

        self.object_shape_dict = self.init_tracker(object_names)

        ## inputs
        self.cam_T_base = cam_T_base # (4, 4)

        ## publisher
        # poses
        self.pub_topic_prefix = "/pose_tracker"
        self.pose_pubs = dict()
        
        ## internal attributes
        self.rgb = None
        self.depth = None
        self.frame_id = 0

        self.object_mask_dict = None
        # self.T_cam = None

        ## subsribers (to sensor)
        self.cv_bridge = CvBridge()
        self.rgb_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.__rgb_callback)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.__depth_callback)

    def init_tracker(self, object_names: list):
        ## initialize the Pose Tracker in the Server
        # object_shape_dict: {"object_name": {"bboxCenter_T_localOrigin": ndarry(4, 4), "bbox": ndarray(2, 3)}}
        #   these are requested objects available in the Server
        object_shape_dict = self.pose_tracker_client.request_init(" ".join(object_names), self.cam_K)
        if input(f"---\nObjects with existing mesh files on Server:\n{object_shape_dict.keys()}.\nContinue? (y/[n]): ").lower() != "y":
            exit()
        self.frame_id = 0
        return object_shape_dict

    def receive_camera_intrinsics(self):
        r"""
        Receive and decode the camera info message
        """
        cam_info_msg = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
        cam_K = np.array(cam_info_msg.K).reshape(3, 3)
        return cam_K

    def get_masks(self, rgb: np.ndarray, object_names) -> dict:
        r"""
        Get the masks for the objects in the scene
        ---
        return:
            masks_dict: {"object_name": mask}
                mask: ndarray(h, w) {0: background, 1: object}
        """
        masks_dict = self.mask_selecter.run_gui(rgb, object_names)
        return masks_dict
    
    def __rgb_callback(self, msg):
        r"""
        Callback function for the RGB image
        """
        try: 
            rgb = self.cv_bridge.imgmsg_to_cv2(msg, "rgb8") # 8-bit rgb, (h, w, 3)
            self.rgb = rgb
            rgb_id = int(msg.header.seq)
        except CvBridgeError as e:
            print(e)
    
    def __depth_callback(self, msg):
        r"""
        Callback function for the depth image
        """
        try: 
            depth = self.cv_bridge.imgmsg_to_cv2(msg, "16UC1").astype(np.uint16) # 16-bit unsigned, (h, w)
            self.depth = depth
            depth_id = int(msg.header.seq)
        except CvBridgeError as e:
            print(e)

    def run(self, visualize=True):
        while self.rgb is None or self.depth is None:
            rospy.sleep(0.1)
        frame_id = 0
        while not rospy.is_shutdown():
            # get the aligned frame
            rgb = self.rgb.copy()
            depth = self.depth.copy()
            
            # get poses
            if frame_id == 0:
                # segment the objects
                # object_mask_dict = self.get_masks(rgb, list(self.object_shape_dict.keys()))
                # invert the mask
                object_mask_dict = self.get_masks(rgb[::-1, ::-1, ...], list(self.object_shape_dict.keys()))
                for n, m in object_mask_dict.items():
                    object_mask_dict[n] = m[::-1, ::-1]

                object_names = list(object_mask_dict.keys())
                masks = np.stack(list(object_mask_dict.values()), axis=0)
                self.object_mask_dict = object_mask_dict
                print(f"---\nObjects with masks:\n{object_names}")

                # estimate the poses
                Ts_cam = self.pose_tracker_client.request_estimate(rgb, depth, object_names, masks)
                print(f"---\nEstimated poses in camera frame!")
            else:
                object_names = list(self.object_mask_dict.keys())
                # track the objects in the scene
                Ts_cam = self.pose_tracker_client.request_track(rgb, depth, object_names)

            # publish and visualize the poses
            Ts_base = self.cam_T_base @ Ts_cam
            vis = rgb.copy()
            for object_name, T_cam, T_base in zip(object_names, Ts_cam, Ts_base):
                self.publish_pose(T_base, frame_id, object_name)
                vis = self.visualize_pose(vis, T_cam, object_name)
            if visualize:
                cv2.imshow("poses", vis[::-1, ::-1, ::-1])
                cv2.waitKey(1)
            frame_id += 1

    def publish_pose(self, T_base, frame_id, object_name):
        r"""
        Publish the pose of the object
        """
        if object_name not in self.pose_pubs:
            self.pose_pubs[object_name] = rospy.Publisher(f"{self.pub_topic_prefix}/{object_name}", PoseStamped, queue_size=10)

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = str(frame_id)
        pose_msg.header.stamp = rospy.Time.now()
        quaternion = R.from_matrix(T_base[:3, :3]).as_quat()
        pose_msg.pose.position.x = T_base[0, 3]
        pose_msg.pose.position.y = T_base[1, 3]
        pose_msg.pose.position.z = T_base[2, 3]
        pose_msg.pose.orientation.x = quaternion[0]
        pose_msg.pose.orientation.y = quaternion[1]
        pose_msg.pose.orientation.z = quaternion[2]
        pose_msg.pose.orientation.w = quaternion[3]
        self.pose_pubs[object_name].publish(pose_msg)

    def visualize_pose(self, rgb: np.ndarray, T_cam: np.ndarray, object_name: str) -> np.ndarray:
        r"""
        Visualize the poses of the objects in the scene
        """
        vis = rgb.copy()
        bboxCenter_T_localOrigin = self.object_shape_dict[object_name]["bboxCenter_T_localOrigin"]
        bbox = self.object_shape_dict[object_name]["bbox"]
        vis = visualize_pose_2d(vis, T_cam, self.cam_K, draw_bbox=True, bbox=bbox, bboxCenter_T_localOrigin=bboxCenter_T_localOrigin)
        return vis

if __name__ == "__main__":
    import cv2
    rospy.init_node("pose_tracker_ros_publisher", anonymous=False)
    # object_names = ["nut5", "wrench5", "wrench5_head"]
    object_names = ["nut4", "wrench4", "wrench4_head"]
    # object_names = ["nut3", "wrench3", "wrench3_head"]
    # object_names = ["nut2", "wrench2", "wrench2_head"]
    # object_names = ["nut1", "wrench1", "wrench1_head"]
    cam_T_base = np.loadtxt("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Vision/cam_T_base.txt")
    pose_tracker = PoseTrackerRosPublisher(object_names, cam_T_base, server_ip="172.16.71.27")
    pose_tracker.run()
