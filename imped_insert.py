#!/usr/bin/env python

from Control.moveit_impedance_agent_interrupt import Agent, start_event_loop, impedence_manager
from Policy.manual_policy import ScrewPolicy, T_to_pose, pose_to_T, pose_visualize
import os.path as osp
import rospy
import sys
from PIL import Image
import cv2
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from pudb import set_trace
import pickle
import time

def binput(msg, candidates):
    while True:
        v = input(msg)
        if v not in candidates:
            print(f"Answer must be in {candidates}")
        else:
            return v

eef_T_wrenchHead = np.loadtxt('/home/robotics/yuhan/Tools/data/wrench_eef_calib/eef_T_wrenchHead.txt')

def wrench_head_to_eef(wrenchHead_T_base: np.array) -> np.array: # (..., 4, 4)
    eef_T_base = wrenchHead_T_base @ eef_T_wrenchHead
    return eef_T_base # (..., 4, 4)

def eef_to_wrench_head(eef_T_base: np.array) -> np.array: # (..., 4, 4)
    wrenchHead_T_base = eef_T_base @ np.linalg.inv(eef_T_wrenchHead)
    return wrenchHead_T_base

episode_id = os.environ.get('EID', "0")
print(f"episode id = {episode_id}")
RAISE_HEIGHT = 0.04
max_scale = 0.02

if __name__ == "__main__":
    # define ros node
    rospy.init_node('imped_insert', anonymous=True)
    start_event_loop()
    agent = Agent(use_gripper=False)

    target_z = None
    origin_pose = None

    while cmd := binput("Impedence Insertion Commands (i - insert, l - left adjust, r - right adjust, q - quit) :", ["i", "l", "r", "q"]):
        if origin_pose is None:
            origin_pose = agent.current_eef_pose.copy()
            target_z = origin_pose[2] - 0.025

        if cmd == "i":
            _ = origin_pose.copy()
            _[2] = target_z
            impedence_manager["enable"] = True
            impedence_manager["stop_pose"] = None
            agent.move_eef(list(_), max_vel_factor=max_scale)

        elif cmd in ["l", "r"]:
            # raise arm
            impedence_manager['enable'] = False
            time.sleep(2.0)
            agent.move_eef(list(origin_pose), max_vel_factor=0.02)
              

            # adjust
            wrenchHead_T_base = eef_to_wrench_head(pose_to_T(origin_pose))
            sign = 1 if cmd == 'l' else -1
            newWrenchHead_T_wrenchHead = np.eye(4)
            newWrenchHead_T_wrenchHead[1, 3] = 0.003 * sign
            newWrenchHead_T_base = wrenchHead_T_base @ newWrenchHead_T_wrenchHead
            origin_pose = T_to_pose(wrench_head_to_eef(newWrenchHead_T_base))
            agent.move_eef(list(origin_pose), max_vel_factor=0.02)

            # reinsert
            impedence_manager["enable"] = True
            impedence_manager["stop_pose"] = None
            _ = origin_pose.copy()
            _[2] = target_z
            current_pose, is_motion_abort = agent.move_eef(list(_), max_vel_factor=0.02)

        elif cmd == "q":
            break