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


episode_id = os.environ.get('EID', "0")
print(f"episode id = {episode_id}")
RAISE_HEIGHT = 0.04

if __name__ == "__main__":
    # define ros node
    rospy.init_node('collect_demo', anonymous=True)
    start_event_loop()
    agent = Agent(use_gripper=False)
    policy = ScrewPolicy()

    reached_pose = agent.current_eef_pose

    episode_data = []
    # 
    depth_scale = policy.realsense_pose_tracker.realsense.depth_scale

    is_done = False
    
    # set_trace()
    print("set trace")


    while not is_done:
        # make prediction
        while True:
            rgb_, depth_, is_frame_received_ = policy.realsense_pose_tracker.realsense.get_aligned_frames(depth_processed=False)
            if is_frame_received_:
                break

        intrinsics_ = policy.realsense_pose_tracker.realsense.cam_K.copy()
        extrinsics_ = policy.base_T_cam.copy()
        current_eef_pose_ = agent.current_eef_pose.copy()

        state_ = policy.state
        target_pose, is_done = policy.act(np.array(reached_pose))
        target_eef_pose_ = target_pose.copy()
        current_wh_pose_ = T_to_pose(policy.eef_to_wrench_head(pose_to_T(current_eef_pose_)))
        frame_data = {
            'state': state_,
            'next_state': policy.state,
            'rgb': rgb_,
            'depth': depth_,
            'current_eef_pose': current_eef_pose_,
            'target_eef_pose': target_eef_pose_, 
            'intrinsics': intrinsics_,
            'extrinsics': extrinsics_,
            'depth_scale': depth_scale,
            'adjustment_direction': '',
            'current_wrenchHead_pose': current_wh_pose_
        }
        # set_trace()
        episode_data.append(frame_data)

        # visualize prediction
        cv2.imshow("target pose", policy.pose_visualize()[..., ::-1])
        cv2.waitKey(0)

        # move robot
        max_scale = 0.02
        if policy.state == "over nut":
            max_scale = 0.1
        elif policy.state == "rotate":
            max_scale = 0.08
        reached_pose, is_motion_abort = agent.move_eef(list(target_pose), max_vel_factor=max_scale)

        if is_motion_abort:
            failure_recovery_yes = binput("Do you need failure recovery (Y/N)?", ['Y', 'N']) == 'Y'
        
        
        if is_motion_abort and policy.state == "on nut" and failure_recovery_yes:
            # set_trace()
            # ensure when exiting this block
            # the state is correctly on nut (ready to rotate)
            success = False
            current_state = policy.state


            def raise_arm(target_pose):
                impedence_manager['enable'] = False
                # time.sleep(2)
                target_pose = target_pose.copy()
                target_pose[2] += RAISE_HEIGHT
                reached_pose, is_motion_abort = agent.move_eef(list(target_pose), max_vel_factor=0.02)

            while not success: 
                current_eef_pose_ = agent.current_eef_pose.copy()
                current_wh_pose_ = T_to_pose(policy.eef_to_wrench_head(pose_to_T(current_eef_pose_)))
                while True:
                    rgb_, depth_, is_frame_received_ = policy.realsense_pose_tracker.realsense.get_aligned_frames(depth_processed=False)
                    if is_frame_received_:
                        break
                frame_data = {
                    'state': current_state,
                    'next_state': "recover",
                    'rgb': rgb_,
                    'depth': depth_,
                    'current_eef_pose': current_eef_pose_,
                    'target_eef_pose': None, 
                    'intrinsics': intrinsics_,
                    'extrinsics': extrinsics_,
                    'depth_scale': depth_scale,
                    'adjustment_direction': '',
                    'current_wrenchHead_pose': current_wh_pose_
                }
                
                current_state = "recover"
                raise_arm(target_pose)

                # TO CHANGE: target_pose
                direction = binput("Please provide an adjustment direction: +/-/* (+ means counterclockwise, * means backoff from wrench, > means move to right size):", ['+', '-', '*', '<', '>'])
                wrenchHead_T_base = policy.eef_to_wrench_head(pose_to_T(target_pose))
                if direction in ['+', '-']:
                    deg_delta = 2 * (-1 if direction == '-' else 1) 
                    # wrenchHead_T_base = policy.eef_to_wrench_head(pose_to_T(target_pose))
                    r = R.from_euler("Z", deg_delta, degrees=True) * R.from_matrix(wrenchHead_T_base[:3, :3])
                    wrenchHead_T_base[:3, :3] = r.as_matrix()
                    target_pose = T_to_pose(policy.wrench_head_to_eef(wrenchHead_T_base))
                elif direction == '*':
                    # *
                    newWrenchHead_T_wrenchHead = np.eye(4)
                    newWrenchHead_T_wrenchHead[0, 3] = -0.003
                    newWrenchHead_T_base = wrenchHead_T_base @ newWrenchHead_T_wrenchHead
                    target_pose = T_to_pose(policy.wrench_head_to_eef(newWrenchHead_T_base))
                else:
                    sign = 1 if direction == '<' else -1
                    newWrenchHead_T_wrenchHead = np.eye(4)
                    newWrenchHead_T_wrenchHead[1, 3] = 0.003 * sign
                    newWrenchHead_T_base = wrenchHead_T_base @ newWrenchHead_T_wrenchHead
                    target_pose = T_to_pose(policy.wrench_head_to_eef(newWrenchHead_T_base))

                _target_pose = target_pose.copy()
                _target_pose[2] += RAISE_HEIGHT
                reached_pose, is_motion_abort = agent.move_eef(list(_target_pose), max_vel_factor=0.02)

                if binput("Do you think the adjustment is already (Y/N)?", ['Y', 'N']).upper() == "N":
                    print("NOT ALREADY.")
                    sys.exit(1)
                
                frame_data['target_eef_pose'] = target_pose
                frame_data['adjustment_direction'] = direction
                episode_data.append(frame_data)

                impedence_manager["enable"] = True
                reached_pose, is_motion_abort = agent.move_eef(list(target_pose), max_vel_factor=0.02)
                success = binput("Do you think the insertion is success Y/N?", ['Y', 'N']).strip().upper() 
                assert success in ['Y', 'N']
                success = success == 'Y'


            policy.queue = []
            impedence_manager["enable"] = False


    output_folder = f"/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/outputs/{episode_id}"
    os.makedirs(output_folder, exist_ok=True)
    for i, edata in enumerate(tqdm(episode_data)):
        pose_in_cam = edata['extrinsics'] @ pose_to_T(edata['current_eef_pose'])
        wh_pose_in_cam = edata['extrinsics'] @ pose_to_T(edata['current_wrenchHead_pose'])
        vis = pose_visualize(edata['rgb'], pose_in_cam, edata['intrinsics'], draw_bbox=True)
        vis = pose_visualize(vis.copy(), wh_pose_in_cam, edata['intrinsics'], draw_bbox=True)
        cv2.imshow('pose', vis[..., ::-1])
        cv2.waitKey(0)

        _i = str(i).zfill(3)
        rgb = edata.pop('rgb') # H, W, 3
        depth = edata.pop('depth') # H, W
        rgb = Image.fromarray(rgb)
        rgb.save(osp.join(output_folder, f'{_i}.rgb.png'))
        cv2.imwrite(osp.join(output_folder, f'{_i}.depth.png'), depth)
        with open(osp.join(output_folder, f"{_i}.state.pkl"), "wb") as f:
            pickle.dump(edata, f)
        

