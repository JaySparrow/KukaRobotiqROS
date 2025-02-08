#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from tutorial.srv import MoveToPoseMaxVel
from robotiq_3f_gripper_services.srv import ChangeMode
from robotiq_3f_gripper_services.srv import MoveAngle

from .control_utils import transform_pose_gripper_fingertip_to_eef, transform_pose_eef_to_gripper_fingertip

import numpy as np
import os, sys
import argparse 

class Agent:
    def __init__(self, use_gripper: bool=False):
        # assume the service is started
        rospy.wait_for_service("/iiwa/move_to_pose_maxvel", timeout=5)
        self.eef_move_to_pose_fn = rospy.ServiceProxy("/iiwa/move_to_pose_maxvel", MoveToPoseMaxVel)

        # assume the services are started and the gripper is activated
        if use_gripper:
            rospy.wait_for_service("/robotiq_3f/ActivateGripper", timeout=5)
            self.gripper_change_mode_fn = rospy.ServiceProxy("/robotiq_3f/ChangeMode", ChangeMode)
            self.gripper_move_fn = rospy.ServiceProxy("/robotiq_3f/MoveGripper", MoveAngle)

            self.default_open = 20
            self.default_close = 65

        self.use_gripper = use_gripper

    def move_eef(self, target_pose: list, max_vel_factor: float=0.05) -> list:
        assert len(target_pose) == 7, f"target_pose = {target_pose}"
        assert max_vel_factor > 0 and max_vel_factor < 1, f"max_vel_factor = {max_vel_factor}"
        resp = self.eef_move_to_pose_fn(target_pose=list(target_pose), max_vel_scaling_factor=max_vel_factor)
        pose_reached = list(resp.reached_pose)
        return pose_reached
    
    def move_gripper(self, target_pose: list, max_vel_factor: float=0.05) -> list:
        assert len(target_pose) == 7

        target_eef_pose = transform_pose_gripper_fingertip_to_eef(np.array(target_pose))
        reached_eef_pose = self.move_eef(list(target_eef_pose), max_vel_factor)

        reached_pose = transform_pose_eef_to_gripper_fingertip(np.array(reached_eef_pose)).round(decimals=3)
        return list(reached_pose)
    
    def reset(self, gripper_only: bool=False):
        is_succ = True
        # arm
        if not gripper_only:
            # home_pose = [0.399975665308, -4.81902681123e-05, 0.628051484975, 0.130520177203, 0.991445660591, -6.44246031522e-05, -4.60826808512e-06] # original go to home with deg = -15
            # home_pose = [0.399975665308, -4.81902681123e-05, 0.628051484975, -8.67979057e-02, 9.96225938e-01, -6.38983397e-05, 9.42166316e-06] # new gripper facing forward with deg = -25
            home_pose = [0.399808, 4.13611e-05, 0.627982, 0.0863636, -0.996263, -0.000163868, 0.00086761] # new gripper facing forward with deg = -25, corrected
            pose_reached = self.move_eef(home_pose)
            if not np.allclose(pose_reached, home_pose):
                is_succ = False

        # gripper
        if self.use_gripper:
            self.change_gripper_mode('p')
            self.open_gripper(self.default_open)

        return is_succ

    def change_gripper_mode(self, mode: str):
        if not self.use_gripper:
            return False
        
        assert isinstance(mode, str) and mode in {'o', 'w', 'p', 'b', 's', 'r'}
        resp = self.gripper_change_mode_fn(mode)
        return resp
    
    def close_gripper(self, deg: int=None):
        if not self.use_gripper:
            return False
        
        if deg is None:
            deg = self.default_close
        else: 
            deg = max(min(deg, 110), 0)
        resp = self.gripper_move_fn(deg)
        return resp
    
    def open_gripper(self, deg: int=None):
        if not self.use_gripper:
            return False
        
        if deg is None:
            deg = self.default_open
        else: 
            deg = max(min(deg, 110), 0)
        resp = self.gripper_move_fn(deg)
        return resp
    
