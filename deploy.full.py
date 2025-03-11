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

import sys
sys.path.append("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/failure_prediction/")
sys.path.append("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/")
from failure_prediction.network import MLP
import torch_jit_utils

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

class SuccessPredictor:
    def __init__(self, model_path, history_len, device):
        self.model = MLP(input_dim=14*history_len, hidden_dim=32, output_dim=2).to(device)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        self.history_len = history_len
        self.device = device
    
    def prepare_input(self, recorded_poses):
        trunacted_poses = dict()
        for key, pose_list in recorded_poses.items():
            pose = torch.from_numpy(np.stack(pose_list, axis=1)).to(self.device) # (num_envs, num_steps, 7)
            start_t = len(pose_list) - 1
            if start_t >= self.history_len-1:
                trunacted_pose = pose[:, start_t-self.history_len+1:start_t+1] # (num_envs, history_len, D)
            else: # pad the first few steps with the first step
                trunacted_pose = torch.cat([pose[:, 0].unsqueeze(1).repeat(1, self.history_len-start_t-1, 1), pose[:, :start_t+1]], dim=1) # (num_envs, history_len, D)
            assert trunacted_pose.size(1) == self.history_len
            trunacted_poses[key] = trunacted_pose.clone()
        
        inputs = torch.cat([trunacted_poses["wrench_head"], trunacted_poses["wrench_head_target"]], dim=-1).flatten(start_dim=1) # (num_envs, 14*history_len)
        return inputs
    
    def predict(self, recorded_poses):
        inputs = self.prepare_input(recorded_poses)
        with torch.no_grad():
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
        return predicted # (num_envs, )

if __name__ == "__main__":
    rospy.init_node('deploy', anonymous=True)

    log_dir = "/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/logs/full"
    algo_name = "binary_recovery"
    obj_size = "Size1"
    log_batch_idx = 2
    log_path = os.path.join(log_dir, algo_name, obj_size, f"batch_{log_batch_idx}.pkl")

    ckpt_root = "/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/checkpoints"
    ckpt_folder = "industreal_deploy"
    ckpt_dir = os.path.join(ckpt_root, ckpt_folder)
    ### INDUSTREAL
    # ckpt_id = "17-23-17-56" # industreal
    ### DEPLOY
    ckpt_id = "29-11-01-30" # impedance relative

    cam_T_base = np.loadtxt("/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Vision/cam_T_base.txt", dtype=np.float32)

    cfg_path = os.path.join(ckpt_dir, ckpt_id, "config.yaml")
    ckpt_path = os.path.join(ckpt_dir, ckpt_id, "model.pth")
    device = "cpu"

    cfg = {
        "pos_action_scale": [0.001, 0.001, 0.001],
        "rot_action_scale": [0.01, 0.01, 0.01],
        "clamp_rot": True,
        "clamp_rot_thresh": 1.0e-6,
        "max_episode_length": 128, # 256,
        "cam_T_base": cam_T_base,
        "action_as_object_displacement": True,
        "observation_type": 1,
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

    ## init success predictor
    success_prediction_model_path = "/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/failure_prediction/checkpoints/traj_succ_model.pth"
    success_predictor = SuccessPredictor(success_prediction_model_path, history_len=20, device=device)

    env = KukaWrenchInsertEnv(cfg)
    env.wait_for_subscribers(seconds=10)

    ## repeated insertion
    num_repeats = 20
    retry_steps = 15 # 40
    try:
        # statistics
        succ_repeats = 0
        total_steps = 0
        # record buffers
        is_success = [0 for _ in range(num_repeats)]
        success_step = [cfg["max_episode_length"] for _ in range(num_repeats)]
        num_retry = [0 for _ in range(num_repeats)]
        num_retry_step = [0 for _ in range(num_repeats)]
        recorded_poses = dict()
        for curr_repeat in range(1, num_repeats+1):
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
                """ Failure recovery """
                if env.progress > 0:
                    predicted_succ = success_predictor.predict(env.recorded_poses)
                    if predicted_succ == 0:
                        num_consecutive_failures += 1
                    else:
                        num_consecutive_failures = 0
                    
                    if num_consecutive_failures > 10 and env.progress + retry_steps < cfg["max_episode_length"]: # learned 
                    # if env.progress - search_init_steps > 40: # manual
                        print(f"    > Retrying at {env.progress}...")
                        # print(f"    > Lifting...")
                        # env.lift_wrench_head_recover(num_steps=int(retry_steps*0.25), update_progress=False)
                        print(f"    > Reinitializing...")
                        # env.move_gripper_to_init_wrench_pose(timeout=retry_steps*0.75)
                        for _ in range(5):
                            env.move_gripper_to_init_wrench_pose(timeout=retry_steps/5)
                        # episode length accounts for retry
                        env.progress += retry_steps
                        
                        # # for _ in range(int(retry_steps*0.75)):
                        # for _ in range(int(retry_steps)):
                        #     target_wrench_head_pos, target_wrench_head_quat = env.get_init_wrench_head_pose()
                        #     curr_wrench_head_pos, curr_wrench_head_quat = env.get_wrench_head_pose()
                        #     wrench_head_quat_inv, wrench_head_pos_inv = torch_jit_utils.tf_inverse(curr_wrench_head_quat, curr_wrench_head_pos)
                        #     rot_actions_quat, pos_actions = torch_jit_utils.tf_combine(
                        #         wrench_head_quat_inv, wrench_head_pos_inv, 
                        #         target_wrench_head_quat, target_wrench_head_pos, 
                        #     )
                        #     angle, axis = torch_jit_utils.quat_to_angle_axis(rot_actions_quat)
                        #     rot_actions = angle.unsqueeze(-1) * axis
                        #     actions = torch.cat([pos_actions, rot_actions], dim=-1)
                        #     actions[:, :3] *= 50
                        #     actions[:, 3:] *= 100000

                        #     obs, is_done = env.step(actions)
                        
                        # reset recovery conditions
                        search_init_steps = env.progress
                        num_consecutive_failures = 0

                        # record retry
                        num_retry[curr_repeat] += 1
                        num_retry_step[curr_repeat] += retry_steps
                        print(f" > End of retry at {env.progress}.")
                
                obs, is_done = env.step(actions)

            if env.progress < cfg["max_episode_length"]:
                succ_repeats += 1
                # record success
                is_success[curr_repeat-1] = 1
                total_steps += env.progress
                # record success step
                success_step[curr_repeat-1] = env.progress
                # record poses
                for key, pose in env.recorded_poses.items():
                    if key not in recorded_poses:
                        recorded_poses[key] = []
                    pose_array = torch.stack(pose, dim=0).squeeze(1).cpu().numpy() # (num_steps, 7)
                    recorded_poses[key].append(pose_array)
                print("> Rotating...")
                env.rotate_wrench_head_on_nut(counter_clockwise=False)
                print("> Resetting...")
                env.lift_wrench_head()
                env.rotate_wrench_head_on_nut(z_offset=0.03)#, counter_clockwise=False)
                print(f"---\n[Repeat {curr_repeat}] SUCCESSFUL. # Steps = {env.progress}")
            else:
                print(f"---\n[Repeat {curr_repeat}] FAILED")
                curr_repeat -= 1
                break  
            
    finally:
        ## save recorded information
        # pack the batch
        batch_data = {
            "is_success": is_success,
            "success_step": success_step,
            "num_retry": num_retry,
            "num_retry_step": num_retry_step,
            "num_consecutive_repeat": curr_repeat
        }
        batch_data["poses"] = deepcopy(recorded_poses)
        # save the batch
        if log_dir is not None:
            with open(log_path, "wb") as f:
                pickle.dump(batch_data, f)
        
        print(f"\n===")
        print(f"# Consecutive repeats: {curr_repeat}")
        print(f"# Successes / {num_repeats}:", succ_repeats)
        print(f"Avg Steps / {cfg['max_episode_length']}:", total_steps / succ_repeats if succ_repeats>0 else cfg['max_episode_length'])
