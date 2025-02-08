import sys
sys.path.append("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Policy/")

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple

from policy_utils import calculate_pose_distance, T_to_pose, pose_to_T, trajectory_interpolate

class WrenchScrewPolicy:
    def __init__(self):
        # transformation between the eef and the wrench head
        self.eef_T_wrenchHead = np.eye(4)

        # state machine
        self.init_state = "init"
        self.state = self.init_state
        self.eef_pose_queue = []
        self.wrenchHead_pose_queue = []
        self.target_eef_pose = None
        self.current_eef_pose = None

        # record
        self.command_history_dict = dict()
    
    def set_init_state(self, init_state):
        self.init_state = init_state
        self.state = init_state

    def calc_eef_T_wrenchHead(self, eef_T_base, wrenchHead_T_base):
        return np.linalg.inv(wrenchHead_T_base) @ eef_T_base
    
    def get_rectified_nut_pose(self, nut_T_base) -> np.array:
        r"""
        rotate nut pose in base frame such that:
            1) the nut's z-axis points up in the base frame
            2) the nut's y-axis points between the base's positive x-axis and negative y-axis
        """        
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
                y_align_score_list.append(x_align_score)
                rectified_nut_T_base_list.append(rectified_nut_T_base)
            
            rectified_nut_T_base = rectified_nut_T_base @ rot_z_T

        max_idx = np.argmax(y_align_score_list)
        rectified_nut_T_base = rectified_nut_T_base_list[max_idx]

        return rectified_nut_T_base

    def align_wrenchHead_to_nut(self, rectified_nut_T_base: np.array, relative_translation: np.array=np.array([0., 0., 0.]), is_xz_aligned: bool=True) -> np.array:
        if is_xz_aligned: # wrench head (x, y, z) -> nut (z, x, y)
            wrenchHead_R_nut = np.array([
                [0., 1, 0], 
                [0, 0, 1], 
                [1, 0, 0]
            ])
        else: # wrench head (x, y, z) -> nut (-z, -x, y)
            wrenchHead_R_nut = np.array([
                [0., -1, 0],
                [0, 0, 1],
                [-1, 0, 0]
            ])
            print("is_xz_aligned:", is_xz_aligned)

        wrenchHead_T_nut = np.block([
            [wrenchHead_R_nut, np.zeros((3, 1))],
            [np.zeros(3), np.ones(1)]
        ])
        wrenchHead_T_nut[:3, 3] += relative_translation
        wrenchHead_T_base = rectified_nut_T_base @ wrenchHead_T_nut
        return wrenchHead_T_base

    def align_eef_to_wrench(self, wrench_T_base: np.array, relative_translation: np.array=np.array([0., 0., 0.14]), is_zz_aligned: bool=False) -> np.array:
        if not is_zz_aligned: # wrench (x, y, z) -> gripper (y, x, -z)
            gripper_R_wrench = np.array([
                [0., 1, 0], 
                [1, 0, 0], 
                [0, 0, -1]
            ])
        else: # wrench (x, y, z) -> gripper (y, -x, z)
            gripper_R_wrench = np.array([
                [0., -1, 0], 
                [1, 0, 0], 
                [0, 0, 1]
            ])
            relative_translation[[0, 2]] *= -1.
            print("is_zz_aligned:", is_zz_aligned)
        gripper_T_wrench = np.block([
            [gripper_R_wrench, relative_translation.reshape((3, 1))],
            [np.zeros(3), np.ones(1)]
        ])

        ## gripper to eef
        eef_T_gripper = np.block([
            [R.from_euler("Z", -12, degrees=True).as_matrix(), np.array([0., 0., -0.14]).reshape((3, 1))],
            [np.zeros(3), np.ones(1)]
        ])
        eef_T_wrench = gripper_T_wrench @ eef_T_gripper

        ## eef in base
        eef_T_base = wrench_T_base @ eef_T_wrench
        return eef_T_base
    
    def rotate_wrenchHead_around_local_axis(self, wrenchHead_T_base: np.array, rad: float, axis: np.array=np.array([0, 0, 1])) -> np.array:
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
    
    def compute_state_actions(self, state: str, nut_T_base: np.array, wrenchHead_T_base: np.array, wrench_T_base: np.array) -> np.array:
        r"""
        return:
            wrenchHead_T_base_trajectory[np.array]: (N, 4, 4)
        """
        nut_T_base = self.get_rectified_nut_pose(nut_T_base)

        eef_T_base = pose_to_T(self.current_eef_pose)
        is_zz_aligned = np.dot(eef_T_base[:3, 2], wrench_T_base[:3, 2]) > 0
        is_xz_aligned = np.dot(wrenchHead_T_base[:3, 0], nut_T_base[:3, 2]) > 0

        if state == "over wrench":
            target_eef_T_base = self.align_eef_to_wrench(wrench_T_base, relative_translation=np.array([0., 0., 0.16]), is_zz_aligned=is_zz_aligned)
            target_eef_pose_queue = [T_to_pose(target_eef_T_base)]
        
        elif state == "on wrench":
            target_eef_T_base = self.align_eef_to_wrench(wrench_T_base, relative_translation=np.array([0., 0., 0.005]), is_zz_aligned=is_zz_aligned)
            target_eef_pose_queue = [T_to_pose(target_eef_T_base)]

        elif state == "post wrench":
            target_eef_pose = self.command_history_dict["on wrench"][-1].copy()
            target_eef_pose[2] += 0.2
            target_eef_pose_queue = [target_eef_pose]

        elif state == "over nut":
            target_wrenchHead_T_base = np.stack([self.align_wrenchHead_to_nut(nut_T_base, relative_translation=np.array([0., 0., 0.07]), is_xz_aligned=is_xz_aligned),
                                                 self.align_wrenchHead_to_nut(nut_T_base, relative_translation=np.array([0., 0., 0.02]), is_xz_aligned=is_xz_aligned),
                                                 self.align_wrenchHead_to_nut(nut_T_base, relative_translation=np.array([0., 0., 0.]), is_xz_aligned=is_xz_aligned)])
                                         
            target_eef_T_base = target_wrenchHead_T_base @ self.eef_T_wrenchHead
            target_eef_pose_queue = list(T_to_pose(target_eef_T_base))
            self.command_history_dict["on nut"] = [target_eef_pose_queue[-1].copy()]
            target_eef_pose_queue = target_eef_pose_queue[:-1]
        
        elif state == "on nut":
            # target_wrenchHead_T_base = self.align_wrenchHead_to_nut(nut_T_base, is_xz_aligned=is_xz_aligned)
            # target_eef_T_base = target_wrenchHead_T_base @ self.eef_T_wrenchHead
            # target_eef_pose_queue = [T_to_pose(target_eef_T_base)]
            target_eef_pose_queue = self.command_history_dict[state]

        elif state == "rotate":
            if is_xz_aligned:
                axis = np.array([1., 0, 0])
            else:
                axis = np.array([-1., 0, 0])
            aligned_wrenchHead_T_base = self.align_wrenchHead_to_nut(nut_T_base, is_xz_aligned=is_xz_aligned)
            start_wrenchHead_T_base = wrenchHead_T_base.copy() 
            start_wrenchHead_T_base[:3, 3] = aligned_wrenchHead_T_base[:3, 3]
            end_wrenchHead_T_base = self.rotate_wrenchHead_around_local_axis(start_wrenchHead_T_base, np.pi/3, axis=axis)
            # interpolate the trajectory of the wrench head
            target_wrenchHead_T_base_trajectory = trajectory_interpolate(start_wrenchHead_T_base, end_wrenchHead_T_base, 0.1, 15*np.pi/180)
            target_eef_T_base_trajectory = target_wrenchHead_T_base_trajectory @ self.eef_T_wrenchHead
            target_eef_pose_queue = list(T_to_pose(target_eef_T_base_trajectory))

        elif state == "lift":
            target_eef_pose = self.command_history_dict["rotate"][-1].copy()
            target_eef_pose[2] += 0.1
            target_eef_pose_queue = [target_eef_pose]

        elif state == "reset wrench":
            target_eef_pose_queue = [self.command_history_dict["post wrench"][-1].copy()]

        elif state == "put wrench":
            target_eef_pose_queue = [self.command_history_dict["on wrench"][-1].copy()]

        elif state == "reset gripper":
            target_eef_pose_queue = [self.command_history_dict["over wrench"][-1].copy(), 
                                     np.array([0.398100546548, 0.00880610458123, 0.610896723217, 0.115789227722, 0.993027448654, 0.00117327863043, -0.0220901001646])]

        else:
            target_eef_pose_queue = [np.array([0.398100546548, 0.00880610458123, 0.610896723217, 0.115789227722, 0.993027448654, 0.00117327863043, -0.0220901001646])]

        return target_eef_pose_queue

    def act(self, current_eef_pose: np.array, nut_T_base: np.array, wrenchHead_T_base: np.array, wrench_T_base: np.array) -> Tuple[np.array, bool]:
        self.eef_T_wrenchHead = self.calc_eef_T_wrenchHead(pose_to_T(current_eef_pose), wrenchHead_T_base)
        self.current_eef_pose = current_eef_pose.copy()

        is_done = False
        if self.state == self.init_state or self.is_pose_reached(current_eef_pose, self.target_eef_pose):
            if len(self.eef_pose_queue) == 0: # state transition
                if self.state == "init":
                    self.state = "over wrench"

                elif self.state == "over wrench":
                    self.state = "on wrench"

                elif self.state == "on wrench":
                    self.state = "post wrench"

                elif self.state == "post wrench":
                    self.state = "over nut"

                elif self.state == "over nut":
                    self.state = "on nut"

                elif self.state == "on nut":
                    self.state = "rotate"
                
                elif self.state == "rotate":
                    self.state = "lift"
                
                elif self.state == "lift":
                    self.state = "reset wrench"

                elif self.state == "reset wrench":
                    self.state = "put wrench"

                elif self.state == "put wrench":
                    self.state = "reset gripper"
                    
                else:
                    self.state = "done"
                    is_done = True
                
                self.eef_pose_queue = self.compute_state_actions(self.state, nut_T_base, wrenchHead_T_base, wrench_T_base)
                self.command_history_dict[self.state] = self.eef_pose_queue.copy()
            
            if not is_done:
                self.target_eef_pose = self.eef_pose_queue.pop(0)
                print(f"[{self.state}-{len(self.eef_pose_queue)}] eef pose: [{', '.join(map(str, self.target_eef_pose))}]")
        
        return self.target_eef_pose, is_done
    
