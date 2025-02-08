#!/usr/bin/env python

import rospy
from tutorial.srv import MoveToPoseMaxVel
from robotiq_3f_gripper_services.srv import ChangeMode
from robotiq_3f_gripper_services.srv import MoveAngle
from geometry_msgs.msg import WrenchStamped, PoseStamped
from std_msgs.msg import Time

# from .control_utils import transform_pose_gripper_fingertip_to_eef, transform_pose_eef_to_gripper_fingertip
from control_utils import transform_pose_gripper_fingertip_to_eef, transform_pose_eef_to_gripper_fingertip, trajectory_interpolate, zero_out_axis_transformation, calculate_pose_distance

import numpy as np
import time

class Agent:
    def __init__(self, use_gripper: bool=False):
        ## CONTROL SERVICES ##
        # assume the service is started
        # rospy.wait_for_service("/iiwa/move_to_pose_maxvel", timeout=5)
        # self.eef_move_to_pose_fn = rospy.ServiceProxy("/iiwa/move_to_pose_maxvel", MoveToPoseMaxVel)
        self.eef_pose_pub = rospy.Publisher("/iiwa/command/CartesianPose", PoseStamped)
        # self.destination_reached_sub = rospy.Subscriber("/iiwa/state/DestinationReached", Time, self.destination_reached_sub_callback)

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
        self.eef_pose_sub = rospy.Subscriber("/iiwa/state/CartesianPose", PoseStamped, self.eef_pose_sub_callback)
        self.current_eef_pose = None

        # eef force feedback subscriber TODO: consider the torque as well
        self.eef_wrench_sub = rospy.Subscriber("/iiwa/state/CartesianWrench", WrenchStamped, self.eef_wrench_callback)
        self.current_eef_force_torque = None
        self.reference_eef_force_torque = None
        self.eef_delta_force_torque_thresh = np.array([4, 4, 3.0, 0.3, 0.3, 0.4]) # np.array([7.0, 7.0, 7.0, 1.0, 1.0, 1.0])

    def is_ready(self):
        while self.current_eef_pose is None:
            rospy.sleep(1)
        return True

    def move_eef(self, target_pose: list, regulate_with_force: bool=False) -> list:
        assert len(target_pose) == 7, f"target_pose = {target_pose}"
        target_pose = np.array(target_pose)

        if regulate_with_force:
            i = 0
            while True:
                plan = trajectory_interpolate(self.current_eef_pose, target_pose, translation_stepsize=0.001, rotation_stepsize=5*np.pi/180)
                i += 1
                if len(plan) < 60:
                    break
                elif i > 10:
                    np.savetxt("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/yuhan/curr_pose.txt", self.current_eef_pose)
                    np.savetxt("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/yuhan/target_pose.txt", target_pose)
                    print(f"[move eef] Interpolation error {len(plan)}!")
                    exit(0)
            print(f"[move_eef] goint to {target_pose.round(decimals=3)} with {len(plan)} steps ...")
        else:
            plan = np.array([target_pose])
        # execute control
        step = 0
        # last_obstructed_axes = np.arange(6)
        last_eef_pose = self.current_eef_pose.copy()
        is_stuck = False
        while step < len(plan):
            pose= plan[step, ...]

            if regulate_with_force:
                # calculate delta force and torque
                delta_force_torque = np.abs(self.current_eef_force_torque - self.reference_eef_force_torque)
                obstructed_axes = np.nonzero(delta_force_torque > self.eef_delta_force_torque_thresh)[0]
                if len(obstructed_axes) > 0:

                    # if np.isin(last_obstructed_axes, obstructed_axes, invert=True).any() or\
                    #     np.isin(obstructed_axes, last_obstructed_axes, invert=True).any(): # obstructed axis changed: replan
                    #     plan = trajectory_interpolate(self.current_eef_pose, target_pose, translation_stepsize=0.002, rotation_stepsize=5*np.pi/180)
                    #     step = 0
                    #     last_obstructed_axes = obstructed_axes.copy()
                    #     continue
                    pose = zero_out_axis_transformation(self.current_eef_pose, pose, axis=obstructed_axes)
        
            self.publish_target_pose(pose, is_block=True)

            # break if not moving
            translation_distance, rotation_distance = calculate_pose_distance(self.current_eef_pose.copy(), last_eef_pose)
            if regulate_with_force and translation_distance < 5e-5 and rotation_distance < 2*np.pi/180:
                # print(f"[move_eef{step}] stopped moving, break!")
                is_stuck = True
                break
            last_eef_pose = self.current_eef_pose.copy()

            step += 1

        return self.current_eef_pose, is_stuck # obstructed_axes

    def publish_target_pose(self, pose, is_block: bool=True):
        out_msg = PoseStamped()
        out_msg.pose.position.x = pose[0]
        out_msg.pose.position.y = pose[1]
        out_msg.pose.position.z = pose[2]
        out_msg.pose.orientation.x = pose[3]
        out_msg.pose.orientation.y = pose[4]
        out_msg.pose.orientation.z = pose[5]
        out_msg.pose.orientation.w = pose[6]
        # self.destination_reached = False
        # t0 = time.time()
        self.eef_pose_pub.publish(out_msg)
        is_reached = False
        if is_block:
            try:
                rospy.wait_for_message("/iiwa/state/DestinationReached", Time, timeout=10)
                is_reached = True
            except:
                # print(f"[controller] target pose {pose.round(decimals=3)} not reached!")
                pass
            # while (not self.destination_reached) or ((time.time() - t0) <= 5):
            #     rospy.sleep(0.5)
        return is_reached
        return self.destination_reached

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
        
    def eef_pose_sub_callback(self, msg: PoseStamped):
        self.current_eef_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, 
                                          msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])

    def eef_wrench_callback(self, msg: WrenchStamped):
        eef_force_torque = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
        self.current_eef_force_torque = eef_force_torque.copy()
        if self.reference_eef_force_torque is None:
            self.reference_eef_force_torque = eef_force_torque.copy()

    # def destination_reached_sub_callback(self, msg: Time):
    #     self.destination_reached = True


if __name__ == "__main__":
    # define ros node
    rospy.init_node('impedance_agent', anonymous=True)

    agent = Agent(use_gripper=False)

    target_pose = np.array([0.566045, -0.0928268, 0.441138, -0.475034, -0.87836, -0.00733759, 0.0526481])
    reached_pose = agent.current_eef_pose

    if agent.is_ready():
        while reached_pose is None or np.linalg.norm(target_pose[:3] - reached_pose[:3]) > 2e-3:
            # reached_pose, obstructed_axes = agent.move_eef(target_pose, max_vel_factor=0.05, return_if_obstructed=True)
            reached_pose, obstructed_axes = agent.move_eef(target_pose, regulate_with_force=True)
            if len(obstructed_axes) > 0:
                print("\n\n pos err: ", np.linalg.norm(target_pose[:3] - reached_pose[:3]).round(decimals=4), "\n\n")
                # input(f"Obstructed: {obstructed_axes}. Continue?")
                target_pose[:2] += np.random.randn(2) * 0.005
        print("reached!")
