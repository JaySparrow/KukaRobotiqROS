#!/usr/bin/env python
import yaml
from gym.spaces import Box
from rl_games.algos_torch.players import PpoPlayerContinuous
import os
import numpy as np
import rospy
from env import KukaWrenchInsertEnv
import torch
from copy import deepcopy
import pickle

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

    log_dir = "/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/logs"
    algo_name = "industreal"
    obj_size = "Size1"
    log_batch_idx = 2
    log_path = os.path.join(log_dir, algo_name, obj_size, f"batch_{log_batch_idx}.pkl")

    ckpt_root = "/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/checkpoints"
    ckpt_folder = "industreal"
    ckpt_dir = os.path.join(ckpt_root, ckpt_folder)
    ### INDUSTREAL
    ckpt_id = "17-23-17-56" # industreal

    cam_T_base = np.loadtxt("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Vision/cam_T_base.txt", dtype=np.float32)

    cfg_path = os.path.join(ckpt_dir, ckpt_id, "config.yaml")
    ckpt_path = os.path.join(ckpt_dir, ckpt_id, "model.pth")
    device = "cpu"

    cfg = {
        "pos_action_scale": [0.001, 0.001, 0.001], #  industreal
        "rot_action_scale": [0.001, 0.001, 0.001],
        "clamp_rot": True,
        "clamp_rot_thresh": 1.0e-6,
        "max_episode_length": 128, # 256,
        "cam_T_base": cam_T_base,
        "action_as_object_displacement": False,
        "observation_type": 0,
    }

    obs_history = []
    action_history = []

    if log_dir is not None:
        log_folder = os.path.dirname(log_path)
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        else:
            if os.path.exists(log_path):
                if input("Warning: log directory already exists. Overwrite? (y/n): ").lower() != "y":
                    exit()

    policy = get_policy(cfg_path, ckpt_path, device)

    env = KukaWrenchInsertEnv(cfg)
    env.wait_for_subscribers(seconds=10)

    ## repeated insertion
    num_repeats = 10
    try:
        
        # statistics
        succ_repeats = 0
        total_steps = 0
        # record buffers
        is_success = [0 for _ in range(num_repeats)]
        success_step = [cfg["max_episode_length"] for _ in range(num_repeats)]
        recorded_poses = dict()
        for curr_repeat in range(num_repeats):
            print(f"### Round {curr_repeat} ###")
            print("> Initializing...")
            obs = env.reset()

            print("> Inserting...")
            is_done = False

            ## failure recovery
            search_init_steps = 0
            num_consecutive_failures = 0
            while not is_done:
                actions = policy.get_action(obs, is_deterministic=True).clone().reshape(1, -1)
                
                obs, is_done = env.step(actions)

                if obs[0, 9] < obs[0, 16] or abs(obs[0, 7]-obs[0, 14]) > 0.05 or abs(obs[0, 8]-obs[0, 15]) > 0.05:
                    print("> Terminate due to large deviation!")
                    env.progress = cfg["max_episode_length"]
                    is_done = True

            if env.progress < cfg["max_episode_length"]:
                succ_repeats += 1
                # record success
                is_success[curr_repeat] = 1
                total_steps += env.progress
                print(f"---\n[Repeat {curr_repeat}] SUCCESSFUL. # Steps = {env.progress}")
            else:
                print(f"---\n[Repeat {curr_repeat}] FAILED")
            # record success step
            success_step[curr_repeat] = env.progress
            
            # record poses
            for key, pose in env.recorded_poses.items():
                if key not in recorded_poses:
                    recorded_poses[key] = []
                pose_array = torch.stack(pose, dim=0).squeeze(1).cpu().numpy() # (num_steps, 7)
                recorded_poses[key].append(pose_array)

            ## save recorded information
            # pack the batch
            batch_data = {
                "is_success": is_success,
                "success_step": success_step,
            }
            batch_data["poses"] = deepcopy(recorded_poses)
            # save the batch
            if log_dir is not None:
                with open(log_path, "wb") as f:
                    pickle.dump(batch_data, f)

            print("> Resetting...")
            env.lift_wrench_head()
            env.reset()
    finally:
        ## save recorded information
        # pack the batch
        batch_data = {
            "is_success": is_success,
            "success_step": success_step,
        }
        batch_data["poses"] = deepcopy(recorded_poses)
        # save the batch
        if log_dir is not None:
            with open(log_path, "wb") as f:
                pickle.dump(batch_data, f)
        
        print(f"\n===")
        print(f"# Successes / {num_repeats}:", succ_repeats)
        print(f"Avg Steps / {cfg['max_episode_length']}:", total_steps / succ_repeats if succ_repeats>0 else cfg['max_episode_length'])
