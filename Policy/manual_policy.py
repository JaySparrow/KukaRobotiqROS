from .realsense_pose_tracker import RealsensePoseTracker
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple
import cv2

from .policy_utils import pose_visualize, trajectory_interpolate, T_to_pose, pose_to_T, calculate_pose_distance

class ScrewPolicy:
    def __init__(self):
        mesh_dict = {
            "yellow nut": ('/home/robotics/yuhan/Tools/data/mesh/nut/nut2.obj', (255, 255, 0)),
        }
        nut_name = "yellow nut"

        realsense_pose_tracker = RealsensePoseTracker(device_id="109422062805", mesh_dict=mesh_dict)
        # initial pose estimate in camera frame
        pose_dict, (detection_dict, rgb, depth) = realsense_pose_tracker.pose_estimate([nut_name, "blue bolt", "red wrench"])
        pose_vis = realsense_pose_tracker.pose_visualize(rgb.copy(), pose_dict)
        detection_vis = realsense_pose_tracker.object_detector.annotate(rgb.copy(), detection_dict)
        vis = np.concatenate([detection_vis, pose_vis], axis=0)
        cv2.imshow(f"{nut_name} detection & pose estimate", vis[..., ::-1])
        if (cv2.waitKey(0) & 0xFF) == ord('\x1b'): # press Esc
            cv2.destroyAllWindows()
            print("[ScrewPolicy] Terminated.")
            exit(0)
        cv2.destroyAllWindows()

        self.base_T_cam = np.loadtxt('/home/robotics/yuhan/Tools/data/cam_calib/manual/Sep21/base_T_cam.txt')
        self.eef_T_wrenchHead = np.loadtxt('/home/robotics/yuhan/Tools/data/wrench_eef_calib/eef_T_wrenchHead.txt')
        self.realsense_pose_tracker = realsense_pose_tracker
        self.nut_name = nut_name

        # state machine
        self.queue = []
        self.state = "init"
        self.target_eef_pose = None
        self.target_wrenchHead_pose = None
    
    def get_raw_nut_pose(self) -> Tuple[np.array, np.array]: # in base frame
        pose_dict, (_, rgb, _) = self.realsense_pose_tracker.pose_track([self.nut_name])
        nut_T_cam = pose_dict[self.nut_name][0] # (4, 4)
        if nut_T_cam is None:
            return None, None
        nut_T_base = np.linalg.inv(self.base_T_cam) @ nut_T_cam
        return nut_T_base, rgb
    
    def get_rectified_nut_pose(self) -> Tuple[np.array, np.array]:
        r"""
        rotate nut pose in base frame such that:
            1) the nut's z-axis points up in the base frame
            2) the nut's y-axis points between the base's positive y-axis and negative x-axis
        """
        nut_T_base, rgb = self.get_raw_nut_pose()
        if nut_T_base is None:
            return None, None
        
        rectified_nut_T_base = nut_T_base.copy()
        # z-axis
        base_z_axis = np.array([0, 0, 1])
        nut_z_axis = rectified_nut_T_base[:3, 2]
        if np.dot(base_z_axis, nut_z_axis) < 0:
            inv_z_T = np.block([
                [R.from_euler("X", 180, degrees=True).as_matrix(), np.zeros((3, 1))],
                [np.zeros(3), np.ones(1)]
            ])
            rectified_nut_T_base = rectified_nut_T_base @ inv_z_T

        # y-axis
        base_x_axis = np.array([1, 0, 0])
        base_y_axis = np.array([0, 1, 0])
        rot_z_T = np.block([
            [R.from_euler("Z", 60, degrees=True).as_matrix(), np.zeros((3, 1))],
            [np.zeros(3), np.ones(1)]
        ])
        y_align_score_list = []
        rectified_nut_T_base_list = []
        for _ in range(6):
            nut_y_axis = rectified_nut_T_base[:3, 1]

            x_align_score = np.dot(base_x_axis, nut_y_axis)
            y_align_score = np.dot(base_y_axis, nut_y_axis)

            if x_align_score > 0 and y_align_score < 0:
                y_align_score_list.append(-y_align_score)
                rectified_nut_T_base_list.append(rectified_nut_T_base)
            
            rectified_nut_T_base = rectified_nut_T_base @ rot_z_T

        max_idx = np.argmax(y_align_score_list)
        rectified_nut_T_base = rectified_nut_T_base_list[max_idx]

        return rectified_nut_T_base, rgb
    
    def wrench_head_to_eef(self, wrenchHead_T_base: np.array) -> np.array: # (..., 4, 4)
        r"""
        convert wrench pose in the base's frame to the end-effector's pose in the base frame
        """
        eef_T_base = wrenchHead_T_base @ self.eef_T_wrenchHead
        return eef_T_base # (..., 4, 4)
    
    def eef_to_wrench_head(self, eef_T_base: np.array) -> np.array: # (..., 4, 4)
        r"""
        convert end-effector pose in the base's frame to the wrench pose in the base frame
        """
        wrenchHead_T_base = eef_T_base @ np.linalg.inv(self.eef_T_wrenchHead)
        return wrenchHead_T_base # (..., 4, 4)
    
    def align_wrench_head_to_nut(self, rectified_nut_T_base: np.array) -> np.array:
        wrenchHead_R_nut = R.from_euler("Z", 90, degrees=True).as_matrix()
        wrenchHead_T_nut = np.block([
            [wrenchHead_R_nut, np.zeros((3, 1))],
            [np.zeros(3), np.ones(1)]
        ])
        wrenchHead_T_nut[0, 3] -= 0.007 # compensate for systematic error Sept 23
        wrenchHead_T_base = rectified_nut_T_base @ wrenchHead_T_nut
        return wrenchHead_T_base
    
    def rotate_wrench_head_around_local_axis(self, wrenchHead_T_base: np.array, rad: float, axis: np.array=np.array([0, 0, 1])) -> np.array:
        r"""
        rotate the wrench head around a local axis by the given angle in radians
        """
        new_T_old = np.block([
            [R.from_rotvec(rad * axis).as_matrix(), np.zeros((3, 1))],
            [np.zeros(3), np.ones(1)]
        ])
        
        rotated_wrenchHead_T_base = wrenchHead_T_base @ new_T_old
        return rotated_wrenchHead_T_base
    
    def is_pose_reached(self, current_pose: np.array, target_pose: np.array, translation_thresh: float=0.001, rotation_thresh: float=2*np.pi/180) -> bool:
        translation_dist, rotation_dist = calculate_pose_distance(current_pose, target_pose)
        return translation_dist < translation_thresh and rotation_dist < rotation_thresh

    def compute_state_actions(self, state: str, current_eef_pose: np.array) -> list:
        if state == "over nut":
            nut_T_base, _ = self.get_rectified_nut_pose()
            wrenchHead_T_base = self.align_wrench_head_to_nut(nut_T_base)
            wrenchHead_T_base[2, 3] += 0.025
            queue = [T_to_pose(self.wrench_head_to_eef(wrenchHead_T_base))]
        
        elif state == "on nut":
            eef_pose_base = current_eef_pose.copy()
            eef_pose_base[2] -= 0.025
            # 0.0105
            queue = [eef_pose_base]
        
        elif state == "rotate":
            current_eef_T = pose_to_T(current_eef_pose)
            current_wrenchHead_T_base = self.eef_to_wrench_head(current_eef_T)
            wrenchHead_T_base = self.rotate_wrench_head_around_local_axis(current_wrenchHead_T_base, np.pi/3)
            # interpolate the trajectory of the wrench head
            wrenchHead_T_base_trajectory = trajectory_interpolate(current_wrenchHead_T_base, wrenchHead_T_base, 0.02, 10*np.pi/180)
            eef_T_base_trajectory = self.wrench_head_to_eef(wrenchHead_T_base_trajectory)
            queue = [T_to_pose(T) for T in eef_T_base_trajectory[1:]]
        else:
            queue = []
        return queue
    
    def act(self, current_eef_pose: np.array) -> Tuple[np.array, bool]:
        is_done = False
        
        
        if self.state == "init" or self.is_pose_reached(current_eef_pose, self.target_eef_pose):
            if len(self.queue) == 0: # state transition
                if self.state == "init":
                    self.state = "over nut"

                elif self.state == "over nut":
                    self.state = "on nut"

                elif self.state == "on nut":
                    self.state = "rotate"
                    
                else:
                    self.state = "done"
                    is_done = True
                
                self.queue = self.compute_state_actions(self.state, current_eef_pose)
            
            if not is_done:
                self.target_eef_pose = self.queue.pop(0)
                self.target_wrenchHead_pose = T_to_pose(self.eef_to_wrench_head(pose_to_T(self.target_eef_pose)))
                print(f"[{self.state}] eef pose/{len(self.queue)}: [{', '.join(map(str, self.target_eef_pose))}]")
        return self.target_eef_pose, is_done

    def pose_visualize(self):
        while True:
            rgb, depth, is_frame_received = self.realsense_pose_tracker.realsense.get_aligned_frames()
            if is_frame_received:
                break

        cam_K = self.realsense_pose_tracker.realsense.cam_K

        # draw poses
        vis = rgb.copy()
        vis = pose_visualize(vis, self.base_T_cam @ pose_to_T(self.target_wrenchHead_pose), cam_K, draw_bbox=True)
        vis = pose_visualize(vis, self.base_T_cam @ pose_to_T(self.target_eef_pose), cam_K, draw_bbox=True)

        # display state and eef pose
        cv2.putText(
                    img=vis,
                    text=f"state: {self.state}",
                    org=(20, 20), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 255, 0),
                    thickness=2, 
                    lineType=2
                )
        cv2.putText(
                    img=vis,
                    text=f"target eef pose: [{' '.join(map(str, self.target_eef_pose.round(decimals=3)))}]",
                    org=(20, 50), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 255, 0),
                    thickness=2, 
                    lineType=2
                )
        return vis

if __name__ == "__main__":
    screw_policy = ScrewPolicy()
    target_pose = None
    cv2.imshow("pose", screw_policy.pose_visualize()[..., ::-1])
    cv2.waitKey(2000)
    is_done = False
    i = 1
    while not is_done:
        target_pose, is_done = screw_policy.act(target_pose)
        cv2.imshow("pose", screw_policy.pose_visualize()[..., ::-1])
        cv2.waitKey(2000)
        i += 1
