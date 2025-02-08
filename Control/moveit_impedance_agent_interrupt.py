#!/usr/bin/env python
import sys
sys.path.append("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Control/")

import rospy
from tutorial.srv import MoveToPoseMaxVel
from robotiq_3f_gripper_services.srv import ChangeMode
from robotiq_3f_gripper_services.srv import MoveAngle
from geometry_msgs.msg import WrenchStamped, PoseStamped

from control_utils import transform_pose_gripper_fingertip_to_eef, transform_pose_eef_to_gripper_fingertip

import numpy as np
import time
import queue
import threading

impedence_manager = {'value': False, 'count': 0, 'enable': True,
                     'stop_pose': None}

task_queue = queue.Queue(maxsize=5)


def event_loop():
    rospy.wait_for_service("/iiwa/move_to_pose_maxvel", timeout=5)
    eef_move_to_pose_fn = rospy.ServiceProxy("/iiwa/move_to_pose_maxvel", MoveToPoseMaxVel)
    while True:
        self = task_queue.get()
        self.is_motion_abort = True 
        print(f"[impedance_agent][{impedence_manager['count']}] Motion abort! Stopping... {self.eef_force_reference}")
        if impedence_manager['stop_pose'] is None:
            impedence_manager['stop_pose'] = list(self.current_eef_pose)
        resp = eef_move_to_pose_fn(target_pose=list(impedence_manager['stop_pose']), max_vel_scaling_factor=0.001)
        print(f"[impedance_agent][{impedence_manager['count']}] Finished ")
        impedence_manager["value"] = False

def start_event_loop():
    threading.Thread(target=event_loop, daemon=True).start()

class Agent:
    def __init__(self, use_gripper: bool=False):
        ## CONTROL SERVICES ##
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

        ## READING SUBSCRIBERS ##
        # current eef pose
        self.eef_pose_sub = rospy.Subscriber("/iiwa/state/CartesianPose", PoseStamped, self.eef_pose_callback)
        self.current_eef_pose = None

        # eef force feedback subscriber TODO: consider the torque as well
        self.eef_wrench_sub = rospy.Subscriber("/iiwa/state/CartesianWrench", WrenchStamped, self.eef_wrench_callback)
        self.eef_force_reference = None
        self.eef_force_threshold = 9.0
        self.eef_torque_reference = None
        self.eef_torque_threshold = 2.5
        self.is_motion_abort = False

    def move_eef(self, target_pose: list, max_vel_factor: float=0.05) -> list:
        assert len(target_pose) == 7, f"target_pose = {target_pose}"
        assert max_vel_factor > 0 and max_vel_factor < 1, f"max_vel_factor = {max_vel_factor}"

        while impedence_manager['value']: time.sleep(0.5)
        
        self.is_motion_abort = False
        # execute control
        resp = self.eef_move_to_pose_fn(target_pose=list(target_pose), max_vel_scaling_factor=max_vel_factor)
        pose_reached = list(resp.reached_pose)

        return pose_reached, self.is_motion_abort
            
    def move_gripper(self, target_pose: list, max_vel_factor: float=0.05) -> list:
        assert len(target_pose) == 7

        target_eef_pose = transform_pose_gripper_fingertip_to_eef(np.array(target_pose))
        reached_eef_pose, is_motion_abort = self.move_eef(list(target_eef_pose), max_vel_factor)

        reached_pose = transform_pose_eef_to_gripper_fingertip(np.array(reached_eef_pose)).round(decimals=3)
        return list(reached_pose), is_motion_abort
    
    def reset(self, gripper_only: bool=False):
        is_succ = True
        # arm
        if not gripper_only:
            home_pose = [0.399808, 4.13611e-05, 0.627982, 0.0863636, -0.996263, -0.000163868, 0.00086761] # new gripper facing forward with deg = -25, corrected
            pose_reached, is_motion_abort = self.move_eef(home_pose)
            if is_motion_abort or not np.allclose(pose_reached, home_pose):
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
            deg = max(min(deg, 250), 0)
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
    
    def eef_pose_callback(self, msg: PoseStamped) -> np.array:
        self.current_eef_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, 
                                          msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
    
    def eef_wrench_callback(self, msg: WrenchStamped):
        force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        torque = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
        if self.eef_force_reference is None:
            self.eef_force_reference = force
            self.eef_torque_reference = torque
        else:
            delta_force = np.abs(force - self.eef_force_reference)
            delta_torque = np.abs(torque - self.eef_torque_reference)
            if np.any(delta_force > self.eef_force_threshold) and self.current_eef_pose is not None and not impedence_manager['value'] and impedence_manager["enable"]:
                impedence_manager['count'] += 1
                impedence_manager['value'] = True

                task_queue.put(self)


if __name__ == "__main__":
    # define ros node
    rospy.init_node('impedance_agent', anonymous=True)

    agent = Agent(use_gripper=False)

    target_pose = np.array([0.566045, -0.0928268, 0.441138, -0.475034, -0.87836, -0.00733759, 0.0526481])
    reached_pose = agent.current_eef_pose

    threading.Thread(target=event_loop, daemon=True).start()

    while reached_pose is None or np.linalg.norm(target_pose - reached_pose) > 1e-2:
        reached_pose, is_motion_abort = agent.move_eef(target_pose, max_vel_factor=0.02)
