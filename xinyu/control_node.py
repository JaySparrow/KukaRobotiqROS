#!/usr/bin/env python
import sys
sys.path.append("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Vision/")
sys.path.append("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Policy/")
sys.path.append("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Control/")

import rospy
from copy import deepcopy
from kuka_base import KukaBase
from control_utils import trajectory_interpolate
import numpy as np
import json
import random

if __name__ == "__main__":
    with open("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/xinyu/poses.json", "r") as f:
        data = json.load(f)

    # define ros node
    rospy.init_node('kuka_base', anonymous=True)

    agent = KukaBase()
    agent.wait_for_subscribers(seconds=5)
    agent.change_gripper_mode("p")

    # target_pose_pool = [
    #     np.array([0.580, -0.204, 0.423, -0.416, 0.600, 0.355, 0.584]),
    #     np.array([0.580, -0.204, 0.463, -0.416, 0.600, 0.355, 0.584])
    # ]
    target_pose_pool = {int(k): np.array(v) for k, v in data.items()}
    all_keys = list(target_pose_pool.keys())
    # all_keys = [4, 12]

    keys = deepcopy(all_keys)
    random.shuffle(keys)

    while not rospy.is_shutdown():
        start_pose = agent.current_eef_pose.copy()
        if len(keys) > 0:
            sample_id = keys.pop(0)
        else:
            keys = deepcopy(all_keys)
            random.shuffle(keys)
            sample_id = keys.pop(0)

        target_pose = target_pose_pool[sample_id]
        print(f'target is {sample_id}')
        # target_pose = target_pose_pool[int(np.random.randint(len(target_pose_pool)))]
        pose_traj = trajectory_interpolate(start_pose, target_pose, 0.02, 0.2)

        for pose in pose_traj:
            agent.move_eef(pose, timeout=40)
        
        agent.move_gripper(100)
        agent.move_gripper(30)
        rospy.sleep(1)
        
