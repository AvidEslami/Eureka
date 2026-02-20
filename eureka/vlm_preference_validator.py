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
VERBOSE = True

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
        
# data_folder = "./validated_vlm_preference_data/auto_preference_data"
data_folder = "./outward_vlm_pref/data"

filenames = [f for f in os.listdir(data_folder) if f.endswith(".txt")]
# First load all rollout data
rollout_data_lengths = {}
for i, path in enumerate(filenames):
    with open(os.path.join(data_folder, path), 'r') as f:
        f.readline()  # Skip score line
        rollout_data_lengths[i] = len([line for line in f])
        # rollout_data_lengths[i] = convert_file_length_to_rollout_length(rollout_data_lengths[i], task)

print(len(rollout_data_lengths))

def get_rollout_observations(rollout_path, task, required_keys=None, max_length=None, nn=False):
    if task == "ShadowHandDoorOpenInward": #Similar to bottlecap setup
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
# pairwise_data = []
for file in filenames:
    if file == "preference_rankings.txt" or file == "ranking_results.json":
        continue
        # i, j = int(comparisons[idx, 0]), int(comparisons[idx, 1])
        # min_length = min(rollout_data_full[i], rollout_data_full[j])
        # for k in [i, j]:
            # if k not in cached_observations:
            #     # Cache the full observation sequence
            #     try:
    # print(f"Loading observations for {file}...")
    cached_observations[file] = get_rollout_observations(os.path.join(data_folder, file), "ShadowHandDoorOpenInward", input_keys, nn=False)
    # cached_nn_observations[file] =  get_rollout_observations(os.path.join(data_folder, file), "ShadowHandDoorOpenInward", input_keys, nn=True)
            #     except Exception as e:
            #         print(f"Error loading observations for {filenames[k]}: {e}")
            #         # cached_observations[k] = []
            #         continue
            # key = (k, min_length)
            # if key not in rollout_rewards:
    # for row in range(len(cached_observations[file])):
    #     pairwise_data.append((cached_observations[file][row],cached_nn_observations[file][row]))

# Open the folder and open the preference_rankings.txt file inside to get the pairwise comparisons
pairwise_comparisons = []
# with open(os.path.join(data_folder, "preference_rankings.txt"), 'r') as f:
#     # Read the entire file then eval to get the list of comparisons, first line is the global order, second line is the tied pairs
#     global_order = eval(f.readline())
#     tied_pairs = eval(f.readline())

# Read ranking_results.json to get the global order and tied pairs
with open(os.path.join(data_folder, "ranking_results.json"), 'r') as f:
    ranking_results = json.load(f)
    global_order = ranking_results["global_order"]
    tied_pairs = ranking_results["tied_pairs"]


name_to_idx = {fn: i for i, fn in enumerate(filenames)}
idx_to_name = {i: fn for fn, i in name_to_idx.items()}


preference_pairs = []
seen_pairs = set()  # To avoid duplicates
for i in range(len(filenames)):
    name = idx_to_name[i]
    if name == "preference_rankings.txt" or name == "ranking_results.json":
        continue
    if name not in global_order:
        for tie in tied_pairs:
            if name in tie:
                # If this name is already in placed, skip it
                print(f"Skipping {name} as it is already in a tied pair: {tie}")
                break

# Make a tied pairs dict for quick lookup
tied_pair_dict = defaultdict(list)
for candidate in global_order:
    for tie in tied_pairs:
        if candidate in tie:
            # Store the other name in the pair
            other_name = tie[0] if tie[1] == candidate else tie[1]
            tied_pair_dict[candidate].append(other_name)
            # break

# Formulate all the pairs
for i in range(len(global_order)):
    # tied_to_i = tied_pair_dict.get([global_order[i]])
    tied_to_i = tied_pair_dict[global_order[i]].copy()
    tied_to_i.append(global_order[i])  # Include itself in the tied list
    for tied_name_i in tied_to_i:
        for j in range(i, len(global_order)):
            if i == j:
                continue
            # Create a pair (i, j) with respect to the global order
            preference_pairs.append((name_to_idx[tied_name_i], name_to_idx[global_order[j]], 0)) # 0 means i is preferred, 1 means j is preferred

            # Now create a pair between i and all tied to j
            if global_order[j] in tied_pair_dict:
                for tied_name_j in tied_pair_dict[global_order[j]]:
                    # Create a pair (i, tied_name) 
                    preference_pairs.append((name_to_idx[tied_name_i], name_to_idx[tied_name_j], 0)) # 0 means i is preferred, 1 means tied_name is preferred

    # Now create a pair between all tied to i and all tied to j, for these the label is 2
    for o in tied_to_i:
        for p in tied_to_i:
            if o == p:
                continue
            # Create a pair (o, p) with respect to the global order
            preference_pairs.append((name_to_idx[o], name_to_idx[p], 2))


# print(f"Global order (fnames): {global_order}")

# Check if we have duplicates, if so we raise errors if RAISE_ERRORS is True
seen_pairs = set()
for i, j, pref in preference_pairs:
    pair = (i, j, pref)
    if pair in seen_pairs:
        if False:
            raise ValueError(f"Duplicate preference pair found: {pair}")
        else:
            print(f"Warning: Duplicate preference pair found: {pair}, skipping it.")
            continue
    seen_pairs.add(pair)
# Print out the number of unique files in the pairwise comparisons and the total number of comparisons
unique_files = set()

for pair in preference_pairs:
    # if preference is 0, and the pair is (i, j), then i is preferred over j, so we add both to the pairwise_comparisons, left is always win
    if pair[2] == 0:
        pairwise_comparisons.append((idx_to_name[pair[0]], idx_to_name[pair[1]]))
    elif pair[2] == 1:
        pairwise_comparisons.append((idx_to_name[pair[1]], idx_to_name[pair[0]]))


for comp in pairwise_comparisons:
    unique_files.add(comp[0])
    unique_files.add(comp[1])
print(f"Number of unique files in comparisons: {len(unique_files)}")
print(f"Total number of pairwise comparisons: {len(pairwise_comparisons)}")



# Print the number of duplicate comparisons (where the same pair of files is compared multiple times)
comparison_counts = defaultdict(int)
for comp in pairwise_comparisons:
    pair = tuple(sorted((comp[0], comp[1])))
    comparison_counts[pair] += 1
num_duplicates = sum(count - 1 for count in comparison_counts.values() if count > 1)
print(f"Number of duplicate comparisons: {num_duplicates}")
# lastly print out the 5 most common files and how many times they appear in the comparisons
file_counts = defaultdict(int)
for comp in pairwise_comparisons:
    file_counts[comp[0]] += 1
    file_counts[comp[1]] += 1
most_common_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:5]
print("Most common files in comparisons:")
for file, count in most_common_files:
    print(f"{file}: {count} comparisons")
# Now we have the pairwise comparisons created by the vlm, we can run them against the ground truth model to see how well the preferences align with the actual reward values. We can compute the reward for each observation in the pair and check if the preference ranking matches the reward ranking.
correct_preferences = 0
total_comparisons = 0

# Shuffle the pairwise comparisons to avoid any order effects
pairwise_comparisons = list(set(pairwise_comparisons)) # Remove duplicates
# Shuffle
import random
random.shuffle(pairwise_comparisons)
pairwise_comparisons = pairwise_comparisons[:20000] # Limit to 100 comparisons for speed, can increase this later

for comp in pairwise_comparisons:
    file1, file2 = comp
    preference = 0 # TODO: check if this is actually the case
    obs1 = cached_observations[file1]
    obs2 = cached_observations[file2]
    # Compute rewards for each observation in the pair and average them
    rewards1 = []
    rewards2 = []
    for o1, o2 in zip(obs1, obs2):
        r1 = ground_truth_model(**o1)
        r2 = ground_truth_model(**o2)
        rewards1.append(r1.item())
        rewards2.append(r2.item())
    avg_reward1 = sum(rewards1) / len(rewards1)
    avg_reward2 = sum(rewards2) / len(rewards2)
    # Check if the preference matches the reward ranking
    if (preference == 0 and avg_reward1 > avg_reward2) or (preference == 1 and avg_reward2 > avg_reward1):
        correct_preferences += 1
    elif VERBOSE:
        print(f"Preference mismatch for pair ({file1}, {file2}): avg_reward1={avg_reward1:.4f}, avg_reward2={avg_reward2:.4f}")
    total_comparisons += 1
print(f"Preference accuracy: {correct_preferences / total_comparisons:.2f} ({correct_preferences}/{total_comparisons})")