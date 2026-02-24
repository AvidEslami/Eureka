import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import json
import time
import logging
import inspect
import subprocess
from typing import Dict, Tuple
from collections import defaultdict
torch.autograd.set_detect_anomaly(False)

MAX_ROLLOUT_LENGTH = 1000000

def ground_truth_model(door_right_handle_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos, door_left_handle_pos, left_hand_ff_pos, left_hand_mf_pos, left_hand_rf_pos, left_hand_lf_pos, left_hand_th_pos):

    right_hand_finger_dist = (torch.norm(door_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(door_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                            + torch.norm(door_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(door_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                            + torch.norm(door_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
    left_hand_finger_dist = (torch.norm(door_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(door_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                            + torch.norm(door_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(door_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                            + torch.norm(door_left_handle_pos - left_hand_th_pos, p=2, dim=-1))
    # Orientation alignment for the cube in hand and goal cube
    # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    right_hand_dist_rew = right_hand_finger_dist
    left_hand_dist_rew = left_hand_finger_dist

    # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    # action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
    up_rew = torch.zeros_like(right_hand_dist_rew)
    up_rew = torch.where(right_hand_finger_dist < 0.5, torch.where(left_hand_finger_dist < 0.5, torch.abs(door_right_handle_pos[:, 1] - door_left_handle_pos[:, 1]) * 2, up_rew), up_rew)

    # up_rew =  torch.where(right_hand_finger_dist <= 0.3, torch.norm(bottle_cap_up - bottle_pos, p=2, dim=-1) * 30, up_rew)

    # reward = torch.exp(-0.1*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.1*(left_hand_dist_rew * dist_reward_scale))
    reward = 2 - right_hand_dist_rew - left_hand_dist_rew + up_rew

    # resets = torch.where(right_hand_finger_dist >= 1.5, torch.ones_like(reset_buf), reset_buf)
    # resets = torch.where(left_hand_finger_dist >= 1.5, torch.ones_like(resets), resets)

    # Find out which envs hit the goal and update successes count
    # successes = torch.where(successes == 0, 
    #                 torch.where(torch.abs(door_right_handle_pos[:, 1] - door_left_handle_pos[:, 1]) > 0.5, torch.ones_like(successes), successes), successes)

    # resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    # goal_resets = torch.zeros_like(resets)

    # num_resets = torch.sum(resets)
    # finished_cons_successes = torch.sum(successes * resets.float())

    # cons_successes = torch.where(resets > 0, successes * resets, consecutive_successes).mean()
    # reward = successes 

    # return reward, resets, goal_resets, progress_buf, successes, cons_successes
    return reward

input_keys = ["door_right_handle_pos", "right_hand_ff_pos", "right_hand_mf_pos", "right_hand_rf_pos", "right_hand_lf_pos", "right_hand_th_pos", "door_left_handle_pos", "left_hand_ff_pos", "left_hand_mf_pos", "left_hand_rf_pos", "left_hand_lf_pos", "left_hand_th_pos"]

class nn_reward_model(nn.Module):
        def __init__(self, obs_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 100),
                nn.ReLU(),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, 1),
                # 
                # nn.Tanh()  # Ensure the output is in the range [-1, 1]
            )
        def forward(self, input_tensor):
            return self.net(input_tensor)
        
data_folder = "./auto_preference_data_open_outward"

# Files to ignore (not rollout files)
IGNORED_FILES = {"preference_rankings.txt", "ranking_results.json"}

# Pick one test file from the data folder (prefer larger files with actual data)
all_files = [f for f in os.listdir(data_folder) if f.endswith(".txt") and f not in IGNORED_FILES]
# Filter for files > 100KB (smaller files are likely empty/headers only)
large_files = [f for f in all_files if os.path.getsize(os.path.join(data_folder, f)) > 100000]
if large_files:
    filenames = [os.path.join(data_folder, large_files[0])]
else:
    filenames = [os.path.join(data_folder, all_files[0])] if all_files else []

# First load all rollout data
rollout_data_lengths = {}
for i, path in enumerate(filenames):
    with open(os.path.join(data_folder, path), 'r') as f:
        f.readline()  # Skip score line
        rollout_data_lengths[i] = len([line for line in f])
        # rollout_data_lengths[i] = convert_file_length_to_rollout_length(rollout_data_lengths[i], task)

print(len(rollout_data_lengths))

def get_rollout_observations(rollout_path, task, required_keys=None, max_length=None, nn=False):
    if task in ("ShadowHandDoorOpenInward", "ShadowHandDoorOpenOutward"):  # Both use same data format
        with open(rollout_path, 'r') as f:
            f.readline()
            f.readline()
            data = [line for line in f]
            # Tensors to capture (Reference of code running in env):
            # print(f"Object Pos: {self.object_pos.tolist()}")
            # print(f"Object Rot: {self.object_rot.tolist()}")
            # print(f"Goal Pos: {self.goal_pos.tolist()}")
            # print(f"Goal Rot: {self.goal_rot.tolist()}")
            # print(f"Door Left Handle Pos: {self.door_left_handle_pos.tolist()}")
            # print(f"Door Right Handle Pos: {self.door_right_handle_pos.tolist()}")
            # print(f"Left Hand Pos: {self.left_hand_pos.tolist()}")
            # print(f"Right Hand Pos: {self.right_hand_pos.tolist()}")
            # print(f"Right Hand Ff Pos: {self.right_hand_ff_pos.tolist()}")
            # print(f"Right Hand Mf Pos: {self.right_hand_mf_pos.tolist()}")
            # print(f"Right Hand Rf Pos: {self.right_hand_rf_pos.tolist()}")
            # print(f"Right Hand Lf Pos: {self.right_hand_lf_pos.tolist()}")
            # print(f"Right Hand Th Pos: {self.right_hand_th_pos.tolist()}")
            # print(f"Left Hand Ff Pos: {self.left_hand_ff_pos.tolist()}")
            # print(f"Left Hand Mf Pos: {self.left_hand_mf_pos.tolist()}")
            # print(f"Left Hand Rf Pos: {self.left_hand_rf_pos.tolist()}")
            # print(f"Left Hand Lf Pos: {self.left_hand_lf_pos.tolist()}")
            # print(f"Left Hand Th Pos: {self.left_hand_th_pos.tolist()}")
            # print(f"Actions: {actions.tolist()}")
            # print(f"Obs buf: {self.obs_buf.tolist()}")

            # Find the line index that contains: "Object Rot:"
            # object_rot_index = next(i for i, line in enumerate(data) if "Object Rot:" in line)
            # goal_pos_index = next(i for i, line in enumerate(data) if "Goal Pos:" in line)
            # goal_rot_index = next(i for i, line in enumerate(data) if "Goal Rot:" in line)
            door_left_handle_pos_index = next(i for i, line in enumerate(data) if "Door Left Handle Pos:" in line)
            door_right_handle_pos_index = next(i for i, line in enumerate(data) if "Door Right Handle Pos:" in line)
            left_hand_pos_index = next(i for i, line in enumerate(data) if "Left Hand Pos:" in line)
            # right_hand_pos_index = next(i for i, line in enumerate(data) if "Right Hand Pos:" in line)
            right_hand_ff_pos_index = next(i for i, line in enumerate(data) if "Right Hand Ff Pos:" in line)
            right_hand_mf_pos_index = next(i for i, line in enumerate(data) if "Right Hand Mf Pos:" in line)
            right_hand_rf_pos_index = next(i for i, line in enumerate(data) if "Right Hand Rf Pos:" in line)
            right_hand_lf_pos_index = next(i for i, line in enumerate(data) if "Right Hand Lf Pos:" in line)
            right_hand_th_pos_index = next(i for i, line in enumerate(data) if "Right Hand Th Pos:" in line)
            left_hand_ff_pos_index = next(i for i, line in enumerate(data) if "Left Hand Ff Pos:" in line)
            left_hand_mf_pos_index = next(i for i, line in enumerate(data) if "Left Hand Mf Pos:" in line)
            left_hand_rf_pos_index = next(i for i, line in enumerate(data) if "Left Hand Rf Pos:" in line)
            left_hand_lf_pos_index = next(i for i, line in enumerate(data) if "Left Hand Lf Pos:" in line)
            left_hand_th_pos_index = next(i for i, line in enumerate(data) if "Left Hand Th Pos:" in line)
            actions_index = next(i for i, line in enumerate(data) if "Actions:" in line)
            obs_buf_index = next(i for i, line in enumerate(data) if "Obs buf:" in line)
            # Lines 0-object_rot_index are the object pos

            if not nn:
                # object_pos = [eval(data[i].strip())[0] for i in range(0, object_rot_index)]
                # object_rot = [eval(data[i].strip())[0] for i in range(object_rot_index + 1, goal_pos_index)]
                # goal_pos = [eval(data[i].strip())[0] for i in range(goal_pos_index + 1, goal_rot_index)]
                # goal_rot = [eval(data[i].strip())[0] for i in range(goal_rot_index + 1, door_left_handle_pos_index)]
                door_left_handle_pos = [eval(data[i].strip())[0] for i in range(door_left_handle_pos_index + 1, door_right_handle_pos_index)]
                door_right_handle_pos = [eval(data[i].strip())[0] for i in range(door_right_handle_pos_index + 1, left_hand_pos_index)]
                # left_hand_pos = [eval(data[i].strip())[0] for i in range(left_hand_pos_index + 1, right_hand_pos_index)]
                # right_hand_pos = [eval(data[i].strip())[0] for i in range(right_hand_pos_index + 1, right_hand_ff_pos_index)]
                right_hand_ff_pos = [eval(data[i].strip())[0] for i in range(right_hand_ff_pos_index + 1, right_hand_mf_pos_index)]
                right_hand_mf_pos = [eval(data[i].strip())[0] for i in range(right_hand_mf_pos_index + 1, right_hand_rf_pos_index)]
                right_hand_rf_pos = [eval(data[i].strip())[0] for i in range(right_hand_rf_pos_index + 1, right_hand_lf_pos_index)]
                right_hand_lf_pos = [eval(data[i].strip())[0] for i in range(right_hand_lf_pos_index + 1, right_hand_th_pos_index)]
                right_hand_th_pos = [eval(data[i].strip())[0] for i in range(right_hand_th_pos_index + 1, left_hand_ff_pos_index)]
                left_hand_ff_pos = [eval(data[i].strip())[0] for i in range(left_hand_ff_pos_index + 1, left_hand_mf_pos_index)]
                left_hand_mf_pos = [eval(data[i].strip())[0] for i in range(left_hand_mf_pos_index + 1, left_hand_rf_pos_index)]
                left_hand_rf_pos = [eval(data[i].strip())[0] for i in range(left_hand_rf_pos_index + 1, left_hand_lf_pos_index)]
                left_hand_lf_pos = [eval(data[i].strip())[0] for i in range(left_hand_lf_pos_index + 1, left_hand_th_pos_index)]
                left_hand_th_pos = [eval(data[i].strip())[0] for i in range(left_hand_th_pos_index + 1, actions_index)]
                # actions = [eval(data[i].strip())[0] for i in range(actions_index + 1, obs_buf_index)]
                
                input_dicts = []
                usable_length = min(MAX_ROLLOUT_LENGTH, 
                                    len(door_left_handle_pos), len(door_right_handle_pos), 
                                    len(right_hand_ff_pos), len(right_hand_mf_pos),
                                    len(right_hand_rf_pos), len(right_hand_lf_pos), len(right_hand_th_pos),
                                    len(left_hand_ff_pos), len(left_hand_mf_pos), len(left_hand_rf_pos),
                                    len(left_hand_lf_pos), len(left_hand_th_pos))#, len(actions)) len(object_pos), len(object_rot), len(goal_pos), len(goal_rot), len(left_hand_pos), len(right_hand_pos), 
                for i in range(usable_length):
                    full_vars = {
                        # "object_pos": torch.tensor(object_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        # "object_rot": torch.tensor(object_rot[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        # "goal_pos": torch.tensor(goal_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        # "goal_rot": torch.tensor(goal_rot[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "door_left_handle_pos": torch.tensor(door_left_handle_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "door_right_handle_pos": torch.tensor(door_right_handle_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        # "left_hand_pos": torch.tensor(left_hand_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        # "right_hand_pos": torch.tensor(right_hand_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "right_hand_ff_pos": torch.tensor(right_hand_ff_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "right_hand_mf_pos": torch.tensor(right_hand_mf_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "right_hand_rf_pos": torch.tensor(right_hand_rf_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "right_hand_lf_pos": torch.tensor(right_hand_lf_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "right_hand_th_pos": torch.tensor(right_hand_th_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "left_hand_ff_pos": torch.tensor(left_hand_ff_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "left_hand_mf_pos": torch.tensor(left_hand_mf_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "left_hand_rf_pos": torch.tensor(left_hand_rf_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "left_hand_lf_pos": torch.tensor(left_hand_lf_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "left_hand_th_pos": torch.tensor(left_hand_th_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        # "actions": torch.tensor(actions[i], dtype=torch.float32, requires_grad=True).unsqueeze(0)
                    }
                    filtered_vars = {k: full_vars[k] for k in required_keys}
                    input_dicts.append(filtered_vars)
            else:
                obs_buf = [eval(data[i].strip())[0] for i in range(obs_buf_index + 1, len(data))]
                input_dicts = []
                for i in range(len(obs_buf)):
                    obs_buf_tensor = torch.tensor(obs_buf[i], dtype=torch.float32, requires_grad=True).unsqueeze(0)
                    input_dicts.append({"obs_buf": obs_buf_tensor})
            return input_dicts

cached_observations = {}
cached_nn_observations = {}
pairwise_data = []
for file in filenames:
        # i, j = int(comparisons[idx, 0]), int(comparisons[idx, 1])
        # min_length = min(rollout_data_full[i], rollout_data_full[j])
        # for k in [i, j]:
            # if k not in cached_observations:
            #     # Cache the full observation sequence
            #     try:
    cached_observations[file] = get_rollout_observations(os.path.join(data_folder, file), "ShadowHandDoorOpenOutward", input_keys, nn=False)
    cached_nn_observations[file] =  get_rollout_observations(os.path.join(data_folder, file), "ShadowHandDoorOpenOutward", input_keys, nn=True)
            #     except Exception as e:
            #         print(f"Error loading observations for {filenames[k]}: {e}")
            #         # cached_observations[k] = []
            #         continue
            # key = (k, min_length)
            # if key not in rollout_rewards:
    for row in range(len(cached_observations[file])):
        pairwise_data.append((cached_observations[file][row],cached_nn_observations[file][row]))

# nn_model = nn_reward_model(obs_dim=len(cached_nn_observations[file][0]["obs_buf"]))
# Load the pretrained nn_model
pth_path = "best_nn_reward_model_full.pth"
nn_model = torch.load(pth_path)

# Test the nn_model against ground truth model by plotting their outputs over time for the one rollout we have
import matplotlib.pyplot as plt
ground_truth_rewards = []
nn_model_rewards = []
for obs_dict, nn_obs_dict in pairwise_data:
    with torch.no_grad():
        gt_reward = ground_truth_model(
            obs_dict["door_right_handle_pos"],
            obs_dict["right_hand_ff_pos"],
            obs_dict["right_hand_mf_pos"],
            obs_dict["right_hand_rf_pos"],
            obs_dict["right_hand_lf_pos"],
            obs_dict["right_hand_th_pos"],
            obs_dict["door_left_handle_pos"],
            obs_dict["left_hand_ff_pos"],
            obs_dict["left_hand_mf_pos"],
            obs_dict["left_hand_rf_pos"],
            obs_dict["left_hand_lf_pos"],
            obs_dict["left_hand_th_pos"]
        )
        ground_truth_rewards.append(gt_reward.item())

        nn_reward = nn_model(nn_obs_dict["obs_buf"])
        nn_model_rewards.append(nn_reward.item())
# Plotting
plt.figure(figsize=(12, 6))
plt.plot(ground_truth_rewards, label='Ground Truth Reward', color='blue')
plt.plot(nn_model_rewards, label='NN Model Reward', color='orange')
plt.xlabel('Timestep')
plt.ylabel('Reward')
plt.title('Reward Comparison Over Time')
plt.legend()
plt.grid()
plt.show()

model = nn_model
example_input = pairwise_data[0][1]["obs_buf"]
use_trace = False  # Set to True to use tracing, False to use scripting
ts_model = (
    torch.jit.trace(model, example_input) if use_trace
    else torch.jit.script(model)
)

# ---- save ----
pt_path = "best_nn_reward_model_jit.pt"
ts_model.save(pt_path)
print(f"TorchScript model saved ➜  {pt_path}")





