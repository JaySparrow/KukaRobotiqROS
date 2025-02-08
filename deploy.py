#!/usr/bin/env python
import yaml
from gym.spaces import Box
from rl_games.algos_torch.players import PpoPlayerContinuous
import os
import numpy as np
import rospy
from env import KukaWrenchInsertEnv

num_envs = 1
seed = 1

def get_policy(cfg_path, ckpt_path, device):
    # Load config.yaml used in training
    with open(cfg_path, "r") as f:
        sim_config = yaml.safe_load(f)

    sim_config["num_envs"] = num_envs
    sim_config["seed"] = seed

    # Define env_info dict
    env_info = {
        "observation_space": Box(
            low=-np.Inf,
            high=np.Inf,
            shape=(sim_config["task"]["env"]["numObservations"],),
            dtype=np.float32,
        ),
        "action_space": Box(
            low=-1.0,
            high=1.0,
            shape=(sim_config["task"]["env"]["numActions"],),
            dtype=np.float32,
        ),
    }
    sim_config["train"]["params"]["config"]["env_info"] = env_info

    # Select device
    sim_config["train"]["params"]["config"]["device_name"] = device

    # Create rl-games agent
    policy = PpoPlayerContinuous(params=sim_config["train"]["params"])

    # Load checkpoint
    policy.restore(ckpt_path)

    # If RNN policy, reset RNN states
    policy.reset()

    print("Finished loading an RL policy.")

    return policy

if __name__ == "__main__":
    rospy.init_node('deploy', anonymous=True)

    ckpt_dir = "/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/checkpoints/industreal_deploy"
    # ckpt_id = "17-23-17-56" # industreal
    ckpt_id = "29-11-01-30" # impedance relative
    # ckpt_id = "30-02-35-43" # Obs2
    # ckpt_id = "30-02-41-29" # Obs3
    # ckpt_id = "30-02-45-01" # Obs4
    cam_T_base = np.loadtxt("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Vision/cam_T_base.txt", dtype=np.float32)

    cfg_path = os.path.join(ckpt_dir, ckpt_id, "config.yaml")
    ckpt_path = os.path.join(ckpt_dir, ckpt_id, "model.pth")
    device = "cpu"

    cfg = {
        # "pos_action_scale": [0.004, 0.004, 0.002],
        # "rot_action_scale": [0.0005, 0.0005, 0.0005],
        # "pos_action_scale": [0.01, 0.01, 0.01],
        # "rot_action_scale": [0.01, 0.01, 0.01],
        "pos_action_scale": [0.001, 0.001, 0.001],
        "rot_action_scale": [0.01, 0.01, 0.01],
        "clamp_rot": True,
        "clamp_rot_thresh": 1.0e-6,
        "max_episode_length": 256,
        "cam_T_base": cam_T_base,
        "action_as_object_displacement": True,
        "observation_type": 1,
    }

    obs_history = []
    action_history = []

    policy = get_policy(cfg_path, ckpt_path, device)

    env = KukaWrenchInsertEnv(cfg)
    
    env.wait_for_subscribers(seconds=10)

    """
    try:
        obs = env.reset()
        # input("Press Enter to continue inserting...")
        is_done = False
        while not is_done:
            # vis = env.rgb.copy()
            # vis = env.draw_poses_in_base(vis, obs[:, 7:14], draw_bbox=True)
            # vis = env.draw_poses_in_base(vis, obs[:, 14:21], draw_bbox=False)
            # cv2.imshow("vis", vis[::-1, ::-1, ::-1])
            # cv2.waitKey(1)
            actions = policy.get_action(obs, is_deterministic=True).clone().reshape(1, -1)

            # obs_history.append(obs)
            # action_history.append(actions)

            obs, is_done = env.step(actions)
            # print(obs[0, 7:14])
        env.lift_wrench_head()
    finally:
        print(env.progress)
        # np.savez("history_real.npz",
        #     obs=torch.cat(obs_history, dim=0).numpy(),
        #     actions=torch.cat(action_history, dim=0).numpy(),
        # )
    """
    """
    ## whole system
    try:
        curr_repeat = 0
        while True: # input(f"\n### [{curr_repeat}] Press Enter to continue...") != "q":
            print(f"### Round {curr_repeat} ###")
            print("> Resetting...")
            obs = env.reset()
            # input("Press Enter to continue inserting...")
            print("> Inserting...")
            is_done = False
            while not is_done:
                actions = policy.get_action(obs, is_deterministic=True).clone().reshape(1, -1)
                obs, is_done = env.step(actions)

            if env.progress < 255:
                print("> Rotating...")
                env.rotate_wrench_head_on_nut()
                print("> Recovering...")
                env.lift_wrench_head()
                env.reset(reset_progress=False)
                print(f"---\n[Repeat {curr_repeat}] Number of inserting steps {env.progress}")
                curr_repeat += 1
                if curr_repeat > 10:
                    # print(f"Successfully completed {curr_repeat} repeats.")
                    break
            else:
                print("---\nFailed to insert.")
                env.lift_wrench_head()
                break
    # except:
    #     print(f"    [Repeat {curr_repeat}] Number of inserting steps {env.progress}")
    finally:
        print("\n===\nNumber of repeats:", curr_repeat)
    """

    ## repeated insertion
    try:
        num_repeats = 10

        succ_repeats = 0
        total_steps = 0
        for curr_repeat in range(num_repeats):
            print(f"### Round {curr_repeat} ###")
            print("> Initializing...")
            obs = env.reset()

            print("> Inserting...")
            is_done = False
            while not is_done:
                actions = policy.get_action(obs, is_deterministic=True).clone().reshape(1, -1)
                obs, is_done = env.step(actions)

            if env.progress < 255:
                succ_repeats += 1
                total_steps += env.progress
                print(f"---\n[Repeat {curr_repeat}] SUCCESSFUL. #Steps = {env.progress}")
            else:
                print(f"---\n[Repeat {curr_repeat}] FAILED")
            
            print("> Resetting...")
            env.lift_wrench_head()
            # env.reset(reset_progress=False)
    finally:
        print(f"\n===")
        print(f"#Successes / {num_repeats}:", succ_repeats)
        print(f"Avg Steps / {255}:", total_steps / succ_repeats if succ_repeats>0 else 255)