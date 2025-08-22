# train_dynamic_reward.py

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

LOG_FAILURES = False
LOG_SUCCESS = False
VERBOSE_PARAMETER_TRACKING = False # If True, the parameters will be printed after each 10 epoch
TRACK_FAILURES = False # If True, the number of failures for each file will be tracked and printed at the end of training
FAILURE_TRACK_PROGRESS = defaultdict(list)

MAXIMIZE_LOSS = False # If True, the loss will be maximized instead of minimized
FLIP_LABELS = False # If True, the labels will be flipped (0 -> 1 and 1 -> 0) in the loss function
NOISE_INSERTION = 0.0

AUTOMATIC_TERMINATION = True # If True, the training process will automatically terminate if the validation loss does not improve for 10 epochs, best model parameters will be returned
BATCH_SIZE = 128 # Batch size for training, if set to None, the entire dataset will be used as a batch
RAISE_ERRORS = True # If True, errors will be raised during training, if False, errors will be caught and printed
MAX_ROLLOUT_LENGTH = 150 # Maximum length of a rollout, if set to None, the entire rollout will be used

# VALIDATION_RATIO = 0.2
VALIDATION_SIZE = 1024
USE_ONLY_ONE_BATCH = False # Minimize VLM queries to save time

SAVE_FINAL_MODEL = True # If True, the final residual NN model will be saved to a file

def return_env_vars(obs_buf: torch.Tensor, potentials: torch.Tensor=None, prev_potential: torch.Tensor=None, action: torch.Tensor=None, dof_vel: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
    if potentials is None:
        object_pos = obs_buf[72:75].unsqueeze(0)
        object_rot = obs_buf[75:79].unsqueeze(0)
        object_angvel = (obs_buf[82:85] / 0.2).unsqueeze(0) # Velocities are scaled by 0.2 -> this is a hardcoded environment constant
        goal_rot = obs_buf[88:92].unsqueeze(0)
        
        fingertip_start_index = 96
        fingertip_state_size = 13
        fingertip_data = []
        for i in range(5): # 5 fingertips
            idx = fingertip_start_index + i * fingertip_state_size
            fingertip_data.append(obs_buf[idx:idx + 3])
        
        # fingertip_pos = torch.tensor(fingertip_data, dtype=torch.float32).reshape(1, 5, 3)
        fingertip_pos = torch.stack(fingertip_data).unsqueeze(0)

        return {
            "object_pos": object_pos,
            "object_rot":object_rot, 
            "goal_rot":goal_rot, 
            "object_angvel":object_angvel, 
            "fingertip_pos":fingertip_pos
        }
    else:
        root_states = obs_buf
        # targets = [1000,0,0]
        targets = torch.tensor([1000, 0, 0], dtype=torch.float32, requires_grad=True)
        potentials = potentials
        prev_potential = prev_potential
        actions = action
        dof_vel = dof_vel
        # prev_potentials = potentials[:-1]
        dt = 0.0166  # Assuming a fixed timestep of 0.02 seconds
        return {
            "root_states": root_states.unsqueeze(0),  # Add batch dimension
            "targets": targets.unsqueeze(0),  # Add batch dimension
            "potentials": potentials.unsqueeze(0),  # Add batch dimension
            "prev_potentials": prev_potential.unsqueeze(0),  # Add batch dimension
            "actions": actions.unsqueeze(0),  # Add batch dimension
                # "dof_vel": dof_vel.unsqueeze(0), # Add batch dimension if not None # Disabled for now
            "dof_vel_scale": 0.2, # From Ant env file
            "dt": dt,
            "up_axis_idx": 2, # From Ant env file
        }

def get_reward_input_keys(model):
    method = model.compute_reward
    sig = inspect.signature(method)
    return list(sig.parameters.keys())[0:]  # exclude 'self'
    
def convert_file_length_to_rollout_length(file_length: int, task: str) -> int:
    if task == "ShadowHandBottleCap":
        # Score line is already subtracted, 
        return int((file_length - 16) / 16)
    elif task == "ShadowHandDoorOpenInward":
        return int((file_length - 20) / 20)
    elif task == "Ant":
        return int ((file_length - 4) / 4)

def get_preference_pairs(data_folder: str, task: str):
    filenames = [f for f in os.listdir(data_folder) if f.endswith(".txt")]
    if task == "ShadowHand":
        # First count lines in each file to determine rollout length
        rollout_lengths = {}
        rollout_scores = {}
        for i, filename in enumerate(filenames):
            with open(os.path.join(data_folder, filename), 'r') as f:
                # f.readline()  # Skip the score line
                # Count the remaining lines which represent the rollout length
                rollout_scores[i] = float(f.readline())
                rollout_lengths[i] = sum(1 for _ in f)
        
        preference_pairs = []
        for i in range(len(filenames)):
            for j in range(i,len(filenames)):
                # Seed is the first number before the first underscore in the filename, if seeds are not equal, skip the pair
                if filenames[i].split("_")[0] != filenames[j].split("_")[0]:
                    continue
                if i != j:
                    if True:
                        # Prefer the shorter rollout (0 means i is preferred, 1 means j is preferred)
                        if rollout_scores[i] == rollout_scores[j]:
                            # If scores are equal and 1, prefer the shorter rollout
                            # If the scores are equal and 0, prefer the longer rollout
                            # If the lengths are equal, prefer neither
                            if rollout_lengths[i] == rollout_lengths[j]:
                                continue
                            elif rollout_scores[i] == 2:
                                preference_pairs.append((i, j, 0 if rollout_lengths[i] < rollout_lengths[j] else 1))
                            else:
                                preference_pairs.append((i, j, 0 if rollout_lengths[i] > rollout_lengths[j] else 1))
                        elif rollout_scores[i] > rollout_scores[j]:
                            preference_pairs.append((i, j, 0))
                        elif rollout_scores[i] < rollout_scores[j]:
                            preference_pairs.append((i, j, 1))
                    else:
                        with open(os.path.join(data_folder, filenames[i]), 'r') as f1:
                            score_i = float(f1.readline())
                            file1_length = len(f1.readlines())
                        with open(os.path.join(data_folder, filenames[j]), 'r') as f2:
                            score_j = float(f2.readline())
                            file2_length = len(f2.readlines())
                        if score_i == score_j:
                            continue
                        # Check the length of both files and discard if they are not the same
                        if file1_length != file2_length:
                            continue
                        preference_pairs.append((i, j, 0 if score_i > score_j else 1))
        if FLIP_LABELS:
            preference_pairs = [(i, j, 1 - pref) for i, j, pref in preference_pairs]
        return filenames, torch.tensor(preference_pairs, dtype=torch.float32)
    elif task == "Ant":
        rollout_scores = {}
        for i, filename in enumerate(filenames):
            with open(os.path.join(data_folder, filename), 'r') as f:
                rollout_scores[i] = f.readline()
                # First line says Mean Success: <>
                rollout_scores[i] = float(rollout_scores[i].split(":")[1].strip())
        preference_pairs = []
        for i in range(len(filenames)):
            for j in range(i, len(filenames)):
                if filenames[i].split("_")[0] != filenames[j].split("_")[0]:
                    continue
                if i != j:
                    if rollout_scores[i] == rollout_scores[j]:
                        continue
                    elif rollout_scores[i] > rollout_scores[j]:
                        preference_pairs.append((i, j, 0))
                    else:
                        preference_pairs.append((i, j, 1))
        if FLIP_LABELS:
            preference_pairs = [(i, j, 1 - pref) for i, j, pref in preference_pairs]
        return filenames, torch.tensor(preference_pairs, dtype=torch.float32)
    elif task == "ShadowHandScissors":
        # First load the json file that contains previously processed pairs in the data_folder
        previous_pairs_path = os.path.join(data_folder, "preference_pairings.json")
        if os.path.exists(previous_pairs_path):
            with open(previous_pairs_path, 'r') as f:
                previous_pairs = json.load(f)
        else:
            previous_pairs = {}
        
        video_paths = []
        for filename in filenames:
            # Open the file and save the first line as the video path
            with open(os.path.join(data_folder, filename), 'r') as f:
                video_path = f.readline().strip()
                video_paths.append(video_path)

        preference_pairs = []
        for pair in previous_pairs:
            # Each pair is a string of the form '["filename1", "filename2"]'
            filenames_pair = eval(pair)
            i = filenames.index(filenames_pair[0])
            j = filenames.index(filenames_pair[1])
            preference = previous_pairs[pair]
            preference_pairs.append((i, j, preference))
        for i in range(len(filenames)):
            for j in range(i, len(filenames)):
                if USE_ONLY_ONE_BATCH and len(preference_pairs) > 2 * BATCH_SIZE:
                    return filenames, torch.tensor(preference_pairs, dtype=torch.float32)
                if filenames[i].split("_")[0] != filenames[j].split("_")[0]:
                    continue
                if i != j:
                    json_key = str([filenames[i], filenames[j]])
                    flipped_json_key = str([filenames[j], filenames[i]])
                    if json_key in previous_pairs or flipped_json_key in previous_pairs:
                        # Grab the preference from the previous_pairs
                        continue # Skip this pair if it has already been processed, in this case we're reusing all previous pairs
                        if json_key in previous_pairs:
                            preference = previous_pairs[json_key]
                        else:
                            # Flip the pair
                            preference = 1 - previous_pairs[flipped_json_key]
                        preference_pairs.append((i, j, preference))
                    else:
                        # If not in previous pairs, we will need to query the vlm and add them to the previous pairs
                        # VLM Runs in a different conda environment, so we need to start another subprocess to run it
                        # vlm_query_started = False
                        conda_environment_name = "vlm"
                        vlm_script_path = "./utils/vlm.py"
                        vlm_output_path = "./utils/vlm_response.txt"

                        vp1 = video_paths[i]
                        vp2 = video_paths[j]

                        # Delete the vlm_output_path if it exists
                        if os.path.exists(vlm_output_path):
                            os.remove(vlm_output_path)
                        # if not vlm_query_started:
                        try:
                            # import subprocess
                            subprocess.run(["conda", "run", "-n", conda_environment_name, "python", vlm_script_path, task, vp1, vp2], check=True)
                            # vlm_query_started = True
                        except Exception as e:
                            print(f"Error running VLM query: {e}")
                            if RAISE_ERRORS:
                                raise e
                            continue # Don't add this pair if there was an error running the VLM query
                        # Start a timer in case the VLM query takes too long
                        start_time = time.time()
                        while True and (time.time() - start_time < 60):  # Wait for up to 60 seconds
                            # Check if the vlm query has finished by checking if the subprocess has finished
                            if os.path.exists(vlm_output_path):
                                with open(vlm_output_path, 'r') as f:
                                    vlm_output = f.read().strip()
                                try:
                                    preference = float(vlm_output)
                                except:
                                    print(f"Error parsing VLM output: {vlm_output}")
                                    if RAISE_ERRORS:
                                        raise ValueError(f"Invalid VLM output: {vlm_output}")
                                    else:
                                        continue
                                if preference == 5:
                                    print(f"VLM couldn't returna preference for {vp1} and {vp2}, skipping pair.")
                                    continue
                                # Add the pair to the previous pairs
                                previous_pairs[json_key] = preference
                                # Save the previous pairs to the json file
                                with open(previous_pairs_path, 'w') as f:
                                    json.dump(previous_pairs, f)
                                preference_pairs.append((i, j, preference))
                                break
        return filenames, torch.tensor(preference_pairs, dtype=torch.float32)
    elif task == "ShadowHandBottleCap":
        # First count lines in each file to determine rollout length
        rollout_lengths = {}
        rollout_scores = {}

        # Open the liv_scores.json in the data_folder if it exists
        if os.path.exists(os.path.join(data_folder, "liv_scores.json")):
            with open(os.path.join(data_folder, "liv_scores.json"), 'r') as f:
                liv_scores = json.load(f)
        else:
            liv_scores = {}
        for i, filename in enumerate(filenames):
            with open(os.path.join(data_folder, filename), 'r') as f:
                # f.readline()  # Skip the score line
                # Count the remaining lines which represent the rollout length
                first_line = f.readline()
                if first_line.startswith("/"):
                    if filename in liv_scores:
                        # Use the cached score
                        rollout_scores[i] = 0
                        rollout_lengths[i] = convert_file_length_to_rollout_length(len(f.readlines()), task)
                        continue
                    # This is a video path, that means score was 0, get the LIV score instead using the video
                    conda_environment_name = "liv"
                    liv_script_path = "./utils/liv_score.py"
                    liv_output_path = "./utils/liv_response.txt"

                    # Delete the liv_output_path if it exists
                    if os.path.exists(liv_output_path):
                        os.remove(liv_output_path)
                    # Start the subprocess to run the LIV query
                    try:
                        subprocess.run(["conda", "run", "-n", conda_environment_name, "python", liv_script_path, task, first_line.strip()], check=True)
                    except Exception as e:
                        print(f"Error running LIV query: {e}")
                        if RAISE_ERRORS:
                            raise e
                        continue
                    # No timer needed for LIV, always returns
                    while True:
                        if os.path.exists(liv_output_path):
                            with open(liv_output_path, 'r') as ff:
                                liv_score = ff.read().strip()
                            try:
                                liv_scores[filename] = eval(liv_score)
                                # Update the liv_scores.json file
                                with open(os.path.join(data_folder, "liv_scores.json"), 'w') as fjson:
                                    json.dump(liv_scores, fjson)
                            except:
                                print(f"Error parsing LIV output: {liv_score}")
                                if RAISE_ERRORS:
                                    raise ValueError(f"Invalid LIV output: {liv_score}")
                                continue
                            break
                else:    
                    rollout_scores[i] = float(first_line)
                rollout_lengths[i] = sum(1 for _ in f)

        
        preference_pairs = [] 
        for i in range(len(filenames)):
            for j in range(i,len(filenames)):
                # Seed is the first number before the first underscore in the filename, if seeds are not equal, skip the pair
                if filenames[i].split("_")[0] != filenames[j].split("_")[0]:
                    continue
                if i != j:
                    if True:
                        # Prefer the shorter rollout (0 means i is preferred, 1 means j is preferred)
                        if rollout_scores[i] == rollout_scores[j]:
                            # If scores are equal and 1, prefer the shorter rollout
                            # If the scores are equal and 0, prefer the longer rollout
                            # If the lengths are equal, prefer neither
                            if rollout_lengths[i] == rollout_lengths[j]:
                                # In this case, we can use the LIV scores
                                i_score = sum(liv_scores.get(filenames[i]))
                                j_score = sum(liv_scores.get(filenames[j]))
                                # Higher score is worse, so we prefer the lower score
                                if i_score < j_score:
                                    preference_pairs.append((i, j, 0))
                                elif i_score > j_score:
                                    preference_pairs.append((i, j, 1))
                            elif rollout_scores[i] == 1:
                                preference_pairs.append((i, j, 0 if rollout_lengths[i] < rollout_lengths[j] else 1))
                            else:
                                preference_pairs.append((i, j, 0 if rollout_lengths[i] > rollout_lengths[j] else 1))
                        elif rollout_scores[i] > rollout_scores[j]:
                            preference_pairs.append((i, j, 0))
                        elif rollout_scores[i] < rollout_scores[j]:
                            preference_pairs.append((i, j, 1))
                    else:
                        with open(os.path.join(data_folder, filenames[i]), 'r') as f1:
                            score_i = float(f1.readline())
                            file1_length = len(f1.readlines())
                        with open(os.path.join(data_folder, filenames[j]), 'r') as f2:
                            score_j = float(f2.readline())
                            file2_length = len(f2.readlines())
                        if score_i == score_j:
                            continue
                        # Check the length of both files and discard if they are not the same
                        if file1_length != file2_length:
                            continue
                        preference_pairs.append((i, j, 0 if score_i > score_j else 1))
        if FLIP_LABELS:
            preference_pairs = [(i, j, 1 - pref) for i, j, pref in preference_pairs]
        return filenames, torch.tensor(preference_pairs, dtype=torch.float32)   
    elif task == "ShadowHandDoorOpenInward":
        preference_rankings = os.path.join(data_folder, "preference_rankings.txt")
        if os.path.exists(preference_rankings):
            with open(preference_rankings, 'r') as f:
                # First line is global order, Second line is tied pairs
                global_order = eval(f.readline().strip())
                tied_pairs = eval(f.readline().strip())
        else:
            global_order = []
            tied_pairs = []

        # First count lines in each file to determine rollout length
        rollout_lengths = {}
        rollout_scores = {}
        # video_paths = []
        video_paths = {}  # Use a dict to map index to video path
        for i, filename in enumerate(filenames):
            # Skip preference_rankings file
            if filename == "preference_rankings.txt":
                continue
            with open(os.path.join(data_folder, filename), 'r') as f:
                # f.readline()  # Skip the score line
                # Count the remaining lines which represent the rollout length
                # rollout_scores[i] = float(f.readline())
                # If the first line starts with a /, it is a video path, otherwise it is a score
                first_line = f.readline()
                if first_line.startswith("/"):
                    # video_paths.append(first_line.strip())
                    video_paths[i] = first_line.strip()
                    rollout_scores[i] = 0
                else:
                    rollout_scores[i] = float(first_line)
                rollout_lengths[i] = sum(1 for _ in f)
         
        name_to_idx = {fn: i for i, fn in enumerate(filenames)}
        idx_to_name = {i: fn for fn, i in name_to_idx.items()}


        def vlm_compare(idx_a, idx_b):
            conda_environment_name = "vlm"
            vlm_script_path = "./utils/vlm.py"
            vlm_output_path = "./utils/vlm_response.txt"

            vp1 = video_paths[idx_a]
            vp2 = video_paths[idx_b]

            print("Querying VLM for preference between", vp1, "and", vp2)
            # Delete the vlm_output_path if it exists
            if os.path.exists(vlm_output_path):
                os.remove(vlm_output_path)
            # Start the subprocess to run the VLM query
            try:
                subprocess.run(["conda", "run", "-n", conda_environment_name, "python", vlm_script_path, task, vp1, vp2], check=True)
            except Exception as e:
                print(f"Error running VLM query: {e}")
                if RAISE_ERRORS:
                    raise e
                return 5
            # Start a timer in case the VLM query takes too long
            start_time = time.time()
            while True and (time.time() - start_time < 60):
                # Check if the vlm query has finished by checking if the subprocess has finished
                if os.path.exists(vlm_output_path):
                    with open(vlm_output_path, 'r') as f:
                        vlm_output = f.read().strip()
                    try:
                        preference = float(vlm_output)
                    except:
                        print(f"Error parsing VLM output: {vlm_output}")
                        if RAISE_ERRORS:
                            raise ValueError(f"Invalid VLM output: {vlm_output}")
                        return 5
                    return preference


        # Single tiny helper: binary insert idx into global_order using our comparison rules.
        def bin_insert(idx):
            a_name = idx_to_name[idx]
            lo, hi = 0, len(global_order)
            while lo < hi:
                mid = (lo + hi) // 2
                # a, b = idx, global_order[mid]
                b_name = global_order[mid]

                if b_name not in name_to_idx:
                    print(f"Error: {b_name} not found in name_to_idx")
                    return
                
                a = idx
                b = name_to_idx[b_name]

                # Compare a vs b using your rules:
                # 1) higher score wins
                if rollout_scores[a] != rollout_scores[b]:
                    a_pref = rollout_scores[a] > rollout_scores[b]
                else:
                    # scores equal
                    if rollout_lengths[a] != rollout_lengths[b]:
                        if rollout_scores[a] == 1.0:
                            # prefer shorter
                            a_pref = rollout_lengths[a] < rollout_lengths[b]
                        else:
                            # prefer longer
                            a_pref = rollout_lengths[a] > rollout_lengths[b]
                    else:
                        # exact tie -> VLM # 1 => a wins, 2 => b wins, 5 => VLM failed, 0 => tie
                        lab = vlm_compare(a, b)  
                        if lab == 5:
                            print(f"VLM couldn't return a preference for {video_paths[a]} and {video_paths[b]}, skipping pair.")
                            return
                        if lab == 1:
                            a_pref = True
                        elif lab == 2:
                            a_pref = False
                        elif lab == 0:
                            # If VLM returns 0, we consider it a tie and let the two stay next to each other in the ranking
                            # We will then also store these pairs in the tied_pairs list
                            # tied_pairs.append((a, b))
                            tied_pairs.append((a_name, b_name))
                            return
                        else:
                            print(f"Unexpected VLM output: {lab}. Expected 1, 2, 0, or 5.")
                            if RAISE_ERRORS:
                                raise ValueError(f"Unexpected VLM output: {lab}")
                            return # Skip this pair if the VLM output is unexpected
                    


                # We maintain weak -> strong. If a is stronger than b, move right.
                if a_pref:
                    lo = mid + 1
                else:
                    hi = mid
                
            # insert if not duplicate
            # if lo >= len(global_order) or global_order[lo] != idx:
            if lo >= len(global_order) or global_order[lo] != a_name:
                global_order.insert(lo, a_name)

        # TEMP NUKE THIS
        # filenames = ["1.txt", "2.txt", "3.txt", "4.txt", "5.txt", "6.txt", "7.txt"]
        # video_paths = [
        #     "/home/avidavid/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0_copy_4.mp4",
        #     "/home/avidavid/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0_copy_5.mp4",
        #     "/home/avidavid/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0_copy_9.mp4",
        #     "/home/avidavid/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0_copy_2.mp4",
        #     "/home/avidavid/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0_copy_7.mp4",
        #     "/home/avidavid/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0_copy_3.mp4",
        #     "/home/avidavid/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0_copy_10.mp4",
        #     # "/home/avidavid/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0 copy 8.mp4",
        #     # "/home/avidavid/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0 copy.mp4",
        #     # "/home/avidavid/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0.mp4"
        # ]
        # rollout_scores = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # rollout_lengths = [140, 140, 140, 140, 140, 140, 140, 140, 140, 140]
        # name_to_idx = {fn: i for i, fn in enumerate(filenames)}
        # idx_to_name = {i: fn for fn, i in name_to_idx.items()}
        # TEMP NUKE THIS END


        preference_pairs = []
        seen_pairs = set()  # To avoid duplicates
        for i in range(len(filenames)):
            name = idx_to_name[i]
            if name == "preference_rankings.txt":
                continue
            if name not in global_order:
                for tie in tied_pairs:
                    if name in tie:
                        # If this name is already in placed, skip it
                        print(f"Skipping {name} as it is already in a tied pair: {tie}")
                        break
                else:
                    bin_insert(i)  # Insert i into global_order
                    with open(preference_rankings, 'w') as f:
                        f.write(str(global_order)) # Stores the list with respect to their filenames
                        f.write("\n")
                        f.write(str(tied_pairs)) # Stores the tied pairs as a list of tuples (i, j) where i and j are the filenames that are tied
                        
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
                if RAISE_ERRORS:
                    raise ValueError(f"Duplicate preference pair found: {pair}")
                else:
                    print(f"Warning: Duplicate preference pair found: {pair}, skipping it.")
                    continue
            seen_pairs.add(pair)

        # exit()
        return filenames, torch.tensor(preference_pairs, dtype=torch.float32)


def get_rollout_observations(rollout_path, task, required_keys=None, max_length=None, nn=False):
    if task == "Ant":
        # print("rollout_path:", rollout_path)
        with open(rollout_path, 'r') as f:
            f.readline()
            # Skip the line that says Root States:
            f.readline()
            # Read the rest of the lines as strings for now
            data = [line for line in f]
            # Find the line index that contains: "Potentials:"
            potentials_index = next(i for i, line in enumerate(data) if "Potentials:" in line)
            prev_potentials_index = next(i for i, line in enumerate(data) if "Previous Potentials:" in line)
            actions_index = next(i for i, line in enumerate(data) if "Actions:" in line)
                # dof_vel_index = next(i for i, line in enumerate(data) if "Dof Vel:" in line) Disabled for now
            # Lines 0-potentials_index are the root states
            root_states = [eval(data[i].strip())[0] for i in range(0, potentials_index)]
            # Lines potentials_index+1 to prev_potentials_index are the potentials            potentials_index = next(i for i, line in enumerate(data) if "Potentials:" in line)
            prev_potentials_index = next(i for i, line in enumerate(data) if "Previous Potentials:" in line)
            actions_index = next(i for i, line in enumerate(data) if "Actions:" in line)
                # dof_vel_index = next(i for i, line in enumerate(data) if "Dof Vel:" in line) Disabled for now
            # Lines 0-potentials_index are the root states
            root_states = [eval(data[i].strip())[0] for i in range(0, potentials_index)]
            # Lines potentials_index+1 to prev_potentials_index are the potentials
            potentials = [eval(data[i].strip())[0] for i in range(potentials_index + 1, prev_potentials_index)]
            # Lines prev_potentials_index+1 to actions_index are the previous potentials
            prev_potentials = [eval(data[i].strip())[0] for i in range(prev_potentials_index + 1, actions_index)]
            # Lines actions_index+1 to dof_vel_index are the actions
                # actions = [eval(data[i].strip())[0] for i in range(actions_index + 1, dof_vel_index)] For now we just go to end of file
            actions = [eval(data[i].strip())[0] for i in range(actions_index + 1, len(data))]
            # Lines dof_vel_index+1 to end are the dof velocities
                # dof_vels = [eval(data[i].strip())[0] for i in range(dof_vel_index + 1, len(data))] Disabled for now
            potentials = [eval(data[i].strip())[0] for i in range(potentials_index + 1, prev_potentials_index)]
            # Lines prev_potentials_index+1 to actions_index are the previous potentials
            prev_potentials = [eval(data[i].strip())[0] for i in range(prev_potentials_index + 1, actions_index)]
            # Lines actions_index+1 to dof_vel_index are the actions
            # actions = [eval(data[i].strip())[0] for i in range(actions_index + 1, dof_vel_index)] For now we just go to end of file
            actions = [eval(data[i].strip())[0] for i in range(actions_index + 1, len(data))]
            # Lines dof_vel_index+1 to end are the dof velocities
                # dof_vels = [eval(data[i].strip())[0] for i in range(dof_vel_index + 1, len(data))] Disabled for now
            

        input_dicts = []
        # print(len(root_states), len(potentials), len(prev_potentials), len(actions))
        usable_length = min(MAX_ROLLOUT_LENGTH, len(root_states), len(potentials), len(prev_potentials), len(actions)) #, len(dof_vels))
        # print(f"Usable length: {usable_length}")
        for i in range(usable_length): # Formerly len(root_states)
            root_state = torch.tensor(root_states[i], dtype=torch.float32, requires_grad=True)
            potential = torch.tensor(potentials[i], dtype=torch.float32, requires_grad=True)
            prev_potential = torch.tensor(prev_potentials[i], dtype=torch.float32, requires_grad=True)
            action = torch.tensor(actions[i], dtype=torch.float32, requires_grad=True)
                # dof_vel = torch.tensor(dof_vels[i], dtype=torch.float32, requires_grad=True) Disabled for now
            # prev_potential = torch.tensor(potentials[i - 1], dtype=torch.float32, requires_grad=True) if i > 0 else torch.zeros_like(potential)
            # Pad to make same length
            # prev_potential = torch.cat([prev_potential, torch.zeros_like(potential[len(prev_potential):])], dim=0)
            # Create a dictionary with the required keys
            full_vars = return_env_vars(root_state, potential, prev_potential, action)#, dof_vel)
            if not nn:
                filtered_vars = {k: full_vars[k] for k in required_keys}
            else:
                filtered_vars = full_vars
            input_dicts.append(filtered_vars)
        if nn:
                # If nn is True, we only want the obs_buf, in this case to save work just flatten each tensor into a single tensor we'll call obs_buf
                # obs_buf = torch.cat([v.squeeze(0) for v in input_dicts], dim=0)
                for i in range(len(input_dicts)):
                    # If there are some floats in the input_dicts, we need to convert them to tensors
                    # the floats are dof_vel_scale, dt, and up_axis_idx
                    for k in input_dicts[i]:
                        if isinstance(input_dicts[i][k], float) or isinstance(input_dicts[i][k], int):
                            input_dicts[i][k] = torch.tensor(input_dicts[i][k], dtype=torch.float32, requires_grad=True).unsqueeze(0).unsqueeze(0)
                        elif k == "potentials" or k == "prev_potentials":
                            # If the key is potention or prev_potention, we need to squeeze it to remove the first dimension
                            input_dicts[i][k] = input_dicts[i][k].unsqueeze(0)
                    obs_buf = torch.cat([input_dicts[i][k].squeeze(0) for k in input_dicts[i]], dim=0)
                    input_dicts[i] = {"obs_buf": obs_buf}  # Replace the input_dict with a single obs_buf
                # input_dicts = [{"obs_buf": obs_buf}]
        return input_dicts
    elif task == "ShadowHandScissors":
        with open(rollout_path, 'r') as f:
            f.readline() # Skip the video path stored on the first line
                # Tensors to Capture (Reference of code running in env):
                # print(f"Object Pos: {self.object_pos.tolist()}")
                # print(f"Object Rot: {self.object_rot.tolist()}")
                # print(f"Goal Pos: {self.goal_pos.tolist()}")
                # print(f"Goal Rot: {self.goal_rot.tolist()}")
                # print(f"Scissors Right Handle Pos: {self.scissors_right_handle_pos.tolist()}")
                # print(f"Scissors Left Handle Pos: {self.scissors_left_handle_pos.tolist()}")
                # print(f"Object Dof Pos: {self.object_dof_pos.tolist()}")
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
            f.readline()  # Skip the line that says Object Pos:
            data = [line for line in f]
            # Find the line index that contains: "Object Rot:"
            object_rot_index = next(i for i, line in enumerate(data) if "Object Rot:" in line)
            goal_pos_index = next(i for i, line in enumerate(data) if "Goal Pos:" in line)
            goal_rot_index = next(i for i, line in enumerate(data) if "Goal Rot:" in line)
            scissors_right_handle_pos_index = next(i for i, line in enumerate(data) if "Scissors Right Handle Pos:" in line)
            scissors_left_handle_pos_index = next(i for i, line in enumerate(data) if "Scissors Left Handle Pos:" in line)
            object_dof_pos_index = next(i for i, line in enumerate(data) if "Object Dof Pos:" in line)
            left_hand_pos_index = next(i for i, line in enumerate(data) if "Left Hand Pos:" in line)
            right_hand_pos_index = next(i for i, line in enumerate(data) if "Right Hand Pos:" in line)
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
            # Lines 0-object_pos_index are the object pos
            object_pos = [eval(data[i].strip())[0] for i in range(0, object_rot_index)]
            object_rot = [eval(data[i].strip())[0] for i in range(object_rot_index+1, goal_pos_index)]
            goal_pos = [eval(data[i].strip())[0] for i in range(goal_pos_index + 1, goal_rot_index)]
            goal_rot = [eval(data[i].strip())[0] for i in range(goal_rot_index + 1, scissors_right_handle_pos_index)]
            scissors_right_handle_pos = [eval(data[i].strip())[0] for i in range(scissors_right_handle_pos_index + 1, scissors_left_handle_pos_index)]
            scissors_left_handle_pos = [eval(data[i].strip())[0] for i in range(scissors_left_handle_pos_index + 1, object_dof_pos_index)]
            object_dof_pos = [eval(data[i].strip())[0] for i in range(object_dof_pos_index + 1, left_hand_pos_index)]
            left_hand_pos = [eval(data[i].strip())[0] for i in range(left_hand_pos_index + 1, right_hand_pos_index)]
            right_hand_pos = [eval(data[i].strip())[0] for i in range(right_hand_pos_index + 1, right_hand_ff_pos_index)]
            right_hand_ff_pos = [eval(data[i].strip())[0] for i in range(right_hand_ff_pos_index + 1, right_hand_mf_pos_index)]
            right_hand_mf_pos = [eval(data[i].strip())[0] for i in range(right_hand_mf_pos_index + 1, right_hand_rf_pos_index)]
            right_hand_rf_pos = [eval(data[i].strip())[0] for i in range(right_hand_rf_pos_index + 1, right_hand_lf_pos_index)]
            right_hand_lf_pos = [eval(data[i].strip())[0] for i in range(right_hand_lf_pos_index + 1, right_hand_th_pos_index)]
            right_hand_th_pos = [eval(data[i].strip())[0] for i in range(right_hand_th_pos_index + 1, left_hand_ff_pos_index)]
            left_hand_ff_pos = [eval(data[i].strip())[0] for i in range(left_hand_ff_pos_index + 1, left_hand_mf_pos_index)]
            left_hand_mf_pos = [eval(data[i].strip())[0] for i in range(left_hand_mf_pos_index + 1, left_hand_rf_pos_index)]
            left_hand_rf_pos = [eval(data[i].strip())[0] for i in range(left_hand_rf_pos_index + 1, left_hand_lf_pos_index)]
            left_hand_lf_pos = [eval(data[i].strip())[0] for i in range(left_hand_lf_pos_index + 1, left_hand_th_pos_index)]
            left_hand_th_pos = [eval(data[i].strip())[0] for i in range(left_hand_th_pos_index + 1, len(data))]

            # # No need to return_env_vars here, just return the filtered variables
            # full_vars = {
            #     "object_pos": torch.tensor(object_pos, dtype=torch.float32, requires_grad=True).unsqueeze(0),
            #     "object_rot": torch.tensor(object_rot, dtype=torch.float32, requires_grad=True).unsqueeze(0),
            #     "goal_pos": torch.tensor(goal_pos, dtype=torch.float32, requires_grad=True).unsqueeze(0),
            #     "goal_rot": torch.tensor(goal_rot, dtype=torch.float32, requires_grad=True).unsqueeze(0),
            #     "scissors_right_handle_pos": torch.tensor(scissors_right_handle_pos, dtype=torch.float32, requires_grad=True).unsqueeze(0),
            #     "scissors_left_handle_pos": torch.tensor(scissors_left_handle_pos, dtype=torch.float32, requires_grad=True).unsqueeze(0),
            #     "object_dof_pos": torch.tensor(object_dof_pos, dtype=torch.float32, requires_grad=True).unsqueeze(0),
            #     "left_hand_pos": torch.tensor(left_hand_pos, dtype=torch.float32, requires_grad=True).unsqueeze(0),
            #     "right_hand_pos": torch.tensor(right_hand_pos, dtype=torch.float32, requires_grad=True).unsqueeze(0),
            #     "right_hand_ff_pos": torch.tensor(right_hand_ff_pos, dtype=torch.float32, requires_grad=True).unsqueeze(0),
            #     "right_hand_mf_pos": torch.tensor(right_hand_mf_pos, dtype=torch.float32, requires_grad=True).unsqueeze(0),
            #     "right_hand_rf_pos": torch.tensor(right_hand_rf_pos, dtype=torch.float32, requires_grad=True).unsqueeze(0),
            #     "right_hand_lf_pos": torch.tensor(right_hand_lf_pos, dtype=torch.float32, requires_grad=True).unsqueeze(0),
            #     "right_hand_th_pos": torch.tensor(right_hand_th_pos, dtype=torch.float32, requires_grad=True).unsqueeze(0),
            #     "left_hand_ff_pos": torch.tensor(left_hand_ff_pos, dtype=torch.float32, requires_grad=True).unsqueeze(0),
            #     "left_hand_mf_pos": torch.tensor(left_hand_mf_pos, dtype=torch.float32, requires_grad=True).unsqueeze(0),
            #     "left_hand_rf_pos": torch.tensor(left_hand_rf_pos, dtype=torch.float32, requires_grad=True).unsqueeze(0),
            #     "left_hand_lf_pos": torch.tensor(left_hand_lf_pos, dtype=torch.float32, requires_grad=True).unsqueeze(0),
            #     "left_hand_th_pos": torch.tensor(left_hand_th_pos, dtype=torch.float32, requires_grad=True).unsqueeze(0),
            # }
            # filtered_vars = {k: full_vars[k] for k in required_keys}
            # input_dicts.append(filtered_vars)
            input_dicts = []
            usable_length = min(MAX_ROLLOUT_LENGTH, len(object_pos), len(object_rot), len(goal_pos), len(goal_rot),
                                len(scissors_right_handle_pos), len(scissors_left_handle_pos), len(object_dof_pos),
                                len(left_hand_pos), len(right_hand_pos), len(right_hand_ff_pos), len(right_hand_mf_pos),
                                len(right_hand_rf_pos), len(right_hand_lf_pos), len(right_hand_th_pos),
                                len(left_hand_ff_pos), len(left_hand_mf_pos), len(left_hand_rf_pos),
                                len(left_hand_lf_pos), len(left_hand_th_pos))
            for i in range(usable_length):
                full_vars = {
                    "object_pos": torch.tensor(object_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                    "object_rot": torch.tensor(object_rot[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                    "goal_pos": torch.tensor(goal_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                    "goal_rot": torch.tensor(goal_rot[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                    "scissors_right_handle_pos": torch.tensor(scissors_right_handle_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                    "scissors_left_handle_pos": torch.tensor(scissors_left_handle_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                    "object_dof_pos": torch.tensor(object_dof_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                    "left_hand_pos": torch.tensor(left_hand_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                    "right_hand_pos": torch.tensor(right_hand_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
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
                }
                if not nn:
                    filtered_vars = {k: full_vars[k] for k in required_keys}
                else:
                    filtered_vars = full_vars # If nn is True, we want all the variables
                input_dicts.append(filtered_vars)
            if nn:
                # If nn is True, we only want the obs_buf, in this case to save work just flatten each tensor into a single tensor we'll call obs_buf
                # obs_buf = torch.cat([v.squeeze(0) for v in input_dicts], dim=0)
                for i in range(len(input_dicts)):
                    obs_buf = torch.cat([input_dicts[i][k].squeeze(0) for k in input_dicts[i]], dim=0)
                    input_dicts[i] = {"obs_buf": obs_buf}  # Replace the input_dict with a single obs_buf
                # input_dicts = [{"obs_buf": obs_buf}]
                return input_dicts
        

            return input_dicts

    elif task == "ShadowHandBottleCap":
        with open(rollout_path, 'r') as f:
            f.readline() # Skip video path or score line
            f.readline()  # Skip the line that says Object Pos:
            data = [line for line in f]
            # Tesnors to Capture (Reference of code running in env):
            # print(f"Object Pos: {self.object_pos.tolist()}")
            # print(f"Object Rot: {self.object_rot.tolist()}")
            # print(f"Goal Pos: {self.goal_pos.tolist()}")
            # print(f"Goal Rot: {self.goal_rot.tolist()}")
            # print(f"Bottle Cap Pos: {self.bottle_cap_pos.tolist()}")
            # print(f"Bottle Pos: {self.bottle_pos.tolist()}")
            # print(f"Bottle Cap Up: {self.bottle_cap_up.tolist()}")
            # print(f"Left Hand Pos: {self.left_hand_pos.tolist()}")
            # print(f"Right Hand Pos: {self.right_hand_pos.tolist()}")
            # print(f"Right Hand Ff Pos: {self.right_hand_ff_pos.tolist()}")
            # print(f"Right Hand Mf Pos: {self.right_hand_mf_pos.tolist()}")
            # print(f"Right Hand Rf Pos: {self.right_hand_rf_pos.tolist()}")
            # print(f"Right Hand Lf Pos: {self.right_hand_lf_pos.tolist()}")
            # print(f"Right Hand Th Pos: {self.right_hand_th_pos.tolist()}")
            # print(f"Actions: {actions.tolist()}")
            # print(f"Obs buf: {self.obs_buf.tolist()}")
            
            object_rot_index = next(i for i, line in enumerate(data) if "Object Rot:" in line)
            goal_pos_index = next(i for i, line in enumerate(data) if "Goal Pos:" in line)
            goal_rot_index = next(i for i, line in enumerate(data) if "Goal Rot:" in line)
            bottle_cap_pos_index = next(i for i, line in enumerate(data) if "Bottle Cap Pos:" in line)
            bottle_pos_index = next(i for i, line in enumerate(data) if "Bottle Pos:" in line)
            bottle_cap_up_index = next(i for i, line in enumerate(data) if "Bottle Cap Up:" in line)
            left_hand_pos_index = next(i for i, line in enumerate(data) if "Left Hand Pos:" in line)
            right_hand_pos_index = next(i for i, line in enumerate(data) if "Right Hand Pos:" in line)
            right_hand_ff_pos_index = next(i for i, line in enumerate(data) if "Right Hand Ff Pos:" in line)
            right_hand_mf_pos_index = next(i for i, line in enumerate(data) if "Right Hand Mf Pos:" in line)
            right_hand_rf_pos_index = next(i for i, line in enumerate(data) if "Right Hand Rf Pos:" in line)
            right_hand_lf_pos_index = next(i for i, line in enumerate(data) if "Right Hand Lf Pos:" in line)
            right_hand_th_pos_index = next(i for i, line in enumerate(data) if "Right Hand Th Pos:" in line)
            actions_index = next(i for i, line in enumerate(data) if "Actions:" in line)
            obs_buf_index = next(i for i, line in enumerate(data) if "Obs buf:" in line)
            # Lines 0-object_rot_index are the object pos
            if not nn:
                object_pos = [eval(data[i].strip())[0] for i in range(0, object_rot_index)]
                object_rot = [eval(data[i].strip())[0] for i in range(object_rot_index + 1, goal_pos_index)]
                goal_pos = [eval(data[i].strip())[0] for i in range(goal_pos_index + 1, goal_rot_index)]
                goal_rot = [eval(data[i].strip())[0] for i in range(goal_rot_index + 1, bottle_cap_pos_index)]
                bottle_cap_pos = [eval(data[i].strip())[0] for i in range(bottle_cap_pos_index + 1, bottle_pos_index)]
                bottle_pos = [eval(data[i].strip())[0] for i in range(bottle_pos_index + 1, bottle_cap_up_index)]
                bottle_cap_up = [eval(data[i].strip())[0] for i in range(bottle_cap_up_index + 1, left_hand_pos_index)]
                left_hand_pos = [eval(data[i].strip())[0] for i in range(left_hand_pos_index + 1, right_hand_pos_index)]
                right_hand_pos = [eval(data[i].strip())[0] for i in range(right_hand_pos_index + 1, right_hand_ff_pos_index)]
                right_hand_ff_pos = [eval(data[i].strip())[0] for i in range(right_hand_ff_pos_index + 1, right_hand_mf_pos_index)]
                right_hand_mf_pos = [eval(data[i].strip())[0] for i in range(right_hand_mf_pos_index + 1, right_hand_rf_pos_index)]
                right_hand_rf_pos = [eval(data[i].strip())[0] for i in range(right_hand_rf_pos_index + 1, right_hand_lf_pos_index)]
                right_hand_lf_pos = [eval(data[i].strip())[0] for i in range(right_hand_lf_pos_index + 1, right_hand_th_pos_index)]
                right_hand_th_pos = [eval(data[i].strip())[0] for i in range(right_hand_th_pos_index + 1, actions_index)]
                actions = [eval(data[i].strip())[0] for i in range(actions_index + 1, obs_buf_index)]

                input_dicts = []
                usable_length = min(MAX_ROLLOUT_LENGTH, len(object_pos), len(object_rot), len(goal_pos), len(goal_rot),
                                    len(bottle_cap_pos), len(bottle_pos), len(bottle_cap_up), len(left_hand_pos),
                                    len(right_hand_pos), len(right_hand_ff_pos), len(right_hand_mf_pos),
                                    len(right_hand_rf_pos), len(right_hand_lf_pos), len(right_hand_th_pos),
                                    len(actions))
                for i in range(usable_length):
                    full_vars = {
                        "object_pos": torch.tensor(object_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "object_rot": torch.tensor(object_rot[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "goal_pos": torch.tensor(goal_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "goal_rot": torch.tensor(goal_rot[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "bottle_cap_pos": torch.tensor(bottle_cap_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "bottle_pos": torch.tensor(bottle_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "bottle_cap_up": torch.tensor(bottle_cap_up[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "left_hand_pos": torch.tensor(left_hand_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "right_hand_pos": torch.tensor(right_hand_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "right_hand_ff_pos": torch.tensor(right_hand_ff_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "right_hand_mf_pos": torch.tensor(right_hand_mf_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "right_hand_rf_pos": torch.tensor(right_hand_rf_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "right_hand_lf_pos": torch.tensor(right_hand_lf_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "right_hand_th_pos": torch.tensor(right_hand_th_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "actions": torch.tensor(actions[i], dtype=torch.float32, requires_grad=True).unsqueeze(0)
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
    elif task == "ShadowHandDoorOpenInward": #Similar to bottlecap setup
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
            object_rot_index = next(i for i, line in enumerate(data) if "Object Rot:" in line)
            goal_pos_index = next(i for i, line in enumerate(data) if "Goal Pos:" in line)
            goal_rot_index = next(i for i, line in enumerate(data) if "Goal Rot:" in line)
            door_left_handle_pos_index = next(i for i, line in enumerate(data) if "Door Left Handle Pos:" in line)
            door_right_handle_pos_index = next(i for i, line in enumerate(data) if "Door Right Handle Pos:" in line)
            left_hand_pos_index = next(i for i, line in enumerate(data) if "Left Hand Pos:" in line)
            right_hand_pos_index = next(i for i, line in enumerate(data) if "Right Hand Pos:" in line)
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
                object_pos = [eval(data[i].strip())[0] for i in range(0, object_rot_index)]
                object_rot = [eval(data[i].strip())[0] for i in range(object_rot_index + 1, goal_pos_index)]
                goal_pos = [eval(data[i].strip())[0] for i in range(goal_pos_index + 1, goal_rot_index)]
                goal_rot = [eval(data[i].strip())[0] for i in range(goal_rot_index + 1, door_left_handle_pos_index)]
                door_left_handle_pos = [eval(data[i].strip())[0] for i in range(door_left_handle_pos_index + 1, door_right_handle_pos_index)]
                door_right_handle_pos = [eval(data[i].strip())[0] for i in range(door_right_handle_pos_index + 1, left_hand_pos_index)]
                left_hand_pos = [eval(data[i].strip())[0] for i in range(left_hand_pos_index + 1, right_hand_pos_index)]
                right_hand_pos = [eval(data[i].strip())[0] for i in range(right_hand_pos_index + 1, right_hand_ff_pos_index)]
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
                actions = [eval(data[i].strip())[0] for i in range(actions_index + 1, obs_buf_index)]
                
                input_dicts = []
                usable_length = min(MAX_ROLLOUT_LENGTH, len(object_pos), len(object_rot), len(goal_pos), len(goal_rot),
                                    len(door_left_handle_pos), len(door_right_handle_pos), len(left_hand_pos),
                                    len(right_hand_pos), len(right_hand_ff_pos), len(right_hand_mf_pos),
                                    len(right_hand_rf_pos), len(right_hand_lf_pos), len(right_hand_th_pos),
                                    len(left_hand_ff_pos), len(left_hand_mf_pos), len(left_hand_rf_pos),
                                    len(left_hand_lf_pos), len(left_hand_th_pos), len(actions))
                for i in range(usable_length):
                    full_vars = {
                        "object_pos": torch.tensor(object_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "object_rot": torch.tensor(object_rot[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "goal_pos": torch.tensor(goal_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "goal_rot": torch.tensor(goal_rot[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "door_left_handle_pos": torch.tensor(door_left_handle_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "door_right_handle_pos": torch.tensor(door_right_handle_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "left_hand_pos": torch.tensor(left_hand_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
                        "right_hand_pos": torch.tensor(right_hand_pos[i], dtype=torch.float32, requires_grad=True).unsqueeze(0),
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
                        "actions": torch.tensor(actions[i], dtype=torch.float32, requires_grad=True).unsqueeze(0)
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


    else:
        with open(rollout_path, 'r') as f:
            f.readline()  # Skip score line
            data = [eval(line) for line in f]
    
    # If max_length is specified, truncate the data
    if max_length is not None:
        data = data[:max_length]
        
    data = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    # object_rot_list, goal_rot_list = [], []
    input_dicts = []
    for i in range(data.shape[0]):
        full_vars = return_env_vars(data[i])
        filtered_vars = {k: full_vars[k] for k in required_keys}
        input_dicts.append(filtered_vars)
        # object_rot_list.append(object_rot)
        # goal_rot_list.append(goal_rot)
    # return object_rot_list, goal_rot_list
    return input_dicts


def bradley_terry_loss(model, comparisons, task, filenames, data_folder, verbose_accururacy=False):
    loss_fn = nn.CrossEntropyLoss()
    input_keys = get_reward_input_keys(model)
    
    # First load all rollout data
    rollout_data_full = {}
    for i, path in enumerate(filenames):
        with open(os.path.join(data_folder, path), 'r') as f:
            f.readline()  # Skip score line
            rollout_data_full[i] = len([line for line in f])
            rollout_data_full[i] = convert_file_length_to_rollout_length(rollout_data_full[i], task)
    
    rollout_rewards = {}
    # for idx in range(len(comparisons)):
    #     i, j = int(comparisons[idx, 0]), int(comparisons[idx, 1])
        
    #     # Determine the length of the shorter rollout
    #     min_length = min(rollout_data_full[i], rollout_data_full[j])
        
    #     # Get observations for both rollouts up to the shorter length
    #     key_i = (i, min_length)
    #     if key_i not in rollout_rewards:
    #         inputs_i = get_rollout_observations(os.path.join(data_folder, filenames[i]), input_keys, min_length)
    #         total_reward_i = torch.tensor(0.0, requires_grad=True)
    #         for inp in inputs_i:
    #             reward, _ = model(**inp)
    #             total_reward_i = total_reward_i + reward
    #         rollout_rewards[key_i] = total_reward_i

    #     key_j = (j, min_length)
    #     if key_j not in rollout_rewards:
    #         inputs_j = get_rollout_observations(os.path.join(data_folder, filenames[j]), input_keys, min_length)
    #         total_reward_j = torch.tensor(0.0, requires_grad=True)
    #         for inp in inputs_j:
    #             reward, _ = model(**inp)
    #             total_reward_j = total_reward_j + reward
    #         rollout_rewards[key_j] = total_reward_j
    cached_observations = {}
    for idx in range(len(comparisons)):
        i, j = int(comparisons[idx, 0]), int(comparisons[idx, 1])
        min_length = min(rollout_data_full[i], rollout_data_full[j])

        for k in [i, j]:
            if k not in cached_observations:
                # Cache the full observation sequence
                try:
                    cached_observations[k] = get_rollout_observations(os.path.join(data_folder, filenames[k]), task, input_keys)
                except Exception as e:
                    print(f"Error loading observations for {filenames[k]}: {e}")
                    # cached_observations[k] = []
                    continue

            
            key = (k, min_length)
            if key not in rollout_rewards:
                inputs = cached_observations[k][:min_length]
                total_reward = torch.tensor(0.0, requires_grad=True)
                for inp in inputs:
                    reward, _ = model(**inp) # tanh
                    total_reward = total_reward + reward
                rollout_rewards[key] = total_reward

    # left = torch.stack([rollout_rewards[int(i)].squeeze() for i in comparisons[:, 0]])
    # right = torch.stack([rollout_rewards[int(i)].squeeze() for i in comparisons[:, 1]])
    # left = torch.stack([rollout_rewards[(int(i), min(rollout_data_full[int(i)], rollout_data_full[int(j)]))].squeeze() for i, j in comparisons])
    # right = torch.stack([rollout_rewards[(int(j), min(rollout_data_full[int(i)], rollout_data_full[int(j)]))].squeeze() for i, j in comparisons])
    left = torch.stack([
        rollout_rewards[(int(row[0]), min(rollout_data_full[int(row[0])], rollout_data_full[int(row[1])]))].squeeze()
        for row in comparisons
    ])
    right = torch.stack([
        rollout_rewards[(int(row[1]), min(rollout_data_full[int(row[0])], rollout_data_full[int(row[1])]))].squeeze()
        for row in comparisons
    ])

    logits = torch.stack([left, right], dim=1)
    targets = comparisons[:, -1].long()

    # with torch.no_grad():
    #     acc = (torch.argmax(logits, dim=1) == targets).float().mean()
    #     print(f"Pairwise accuracy: {acc.item():.2f}")

    # if verbose_accururacy:
    #     if TRACK_FAILURES:
    #         failure_per_idx = defaultdict(int)

    #     for i in range(len(comparisons)):
    #         left_idx = int(comparisons[i, 0])
    #         right_idx = int(comparisons[i, 1])
    #         preference = int(comparisons[i, 2])

    #         model_rewards = torch.stack([left[i], right[i]])
    #         if preference == 0:
    #             if model_rewards[0] > model_rewards[1]:
    #                 if LOG_SUCCESS:
    #                     print(f"Correct: {filenames[left_idx]} ({model_rewards[0]:.4f}) > {filenames[right_idx]} ({model_rewards[1]:.4f})")
    #             else:
    #                 if LOG_FAILURES:
    #                     print(f"Incorrect: {filenames[left_idx]} ({model_rewards[0]:.4f}) < {filenames[right_idx]} ({model_rewards[1]:.4f})")   
    #                 if TRACK_FAILURES:
    #                     failure_per_idx[left_idx] += 1
    #         else:
    #             if model_rewards[0] < model_rewards[1]:
    #                 if LOG_SUCCESS:
    #                     print(f"Correct: {filenames[left_idx]} ({model_rewards[0]:.4f}) < {filenames[right_idx]} ({model_rewards[1]:.4f})")
    #             else:
    #                 if LOG_FAILURES:
    #                     print(f"Incorrect: {filenames[left_idx]} ({model_rewards[0]:.4f}) > {filenames[right_idx]} ({model_rewards[1]:.4f})")
    #                 if TRACK_FAILURES:
    #                     failure_per_idx[right_idx] += 1
    #     if TRACK_FAILURES:
    #         # Iterate over all the files and add the failures for those that failed
    #         for i in range(len(filenames)):
    #             FAILURE_TRACK_PROGRESS[i].append(failure_per_idx[i])
    #         print("Failure tracking:")
    #         for i in FAILURE_TRACK_PROGRESS:
    #             print(f"{filenames[i]}: {FAILURE_TRACK_PROGRESS[i]}")


    # return loss_fn(logits, targets), acc

    # Handle tri-state targets: 0/1 => CE as before; 2 => MSE(left, right)
    device = left.device
    mask_tie = (targets == 2)
    mask_ce = ~mask_tie

    ce_loss = torch.tensor(0.0, device=device)
    mse_loss = torch.tensor(0.0, device=device)
    n_ce = int(mask_ce.sum().item())
    n_mse = int(mask_tie.sum().item())

    if n_ce > 0:
        ce_loss = loss_fn(logits[mask_ce], targets[mask_ce])  # mean over 0/1 samples
    if n_mse > 0:
        mse_loss = F.mse_loss(left[mask_tie], right[mask_tie], reduction='mean')  # mean over tie samples

    if (n_ce + n_mse) > 0:
        base_loss = (ce_loss * n_ce + mse_loss * n_mse) / (n_ce + n_mse)
    else:
        base_loss = torch.tensor(0.0, device=device)

    total_loss = base_loss

    with torch.no_grad():
        if n_ce > 0:
            predictions = torch.argmax(logits[mask_ce], dim=1)
            acc = (predictions == targets[mask_ce]).float().mean()
            print(f"Pairwise accuracy: {acc.item():.2f}, Base loss: {base_loss.item():.4f}")
        else:
            print(f"Pairwise accuracy: N/A (only ties), Base loss: {base_loss.item():.4f}")

    if verbose_accururacy:
        if TRACK_FAILURES:
            failure_per_idx = defaultdict(int)

        for i in range(len(comparisons)):
            left_idx = int(comparisons[i, 0])
            right_idx = int(comparisons[i, 1])
            preference = int(comparisons[i, 2])

            model_rewards = torch.stack([left[i], right[i]])
            if preference == 0:
                if model_rewards[0] > model_rewards[1]:
                    if LOG_SUCCESS:
                        print(f"Correct: {filenames[left_idx]} ({model_rewards[0]:.4f}) > {filenames[right_idx]} ({model_rewards[1]:.4f})")
                else:
                    if LOG_FAILURES:
                        print(f"Incorrect: {filenames[left_idx]} ({model_rewards[0]:.4f}) < {filenames[right_idx]} ({model_rewards[1]:.4f})")   
                    if TRACK_FAILURES:
                        failure_per_idx[left_idx] += 1
            elif preference == 1:
                if model_rewards[0] < model_rewards[1]:
                    if LOG_SUCCESS:
                        print(f"Correct: {filenames[left_idx]} ({model_rewards[0]:.4f}) < {filenames[right_idx]} ({model_rewards[1]:.4f})")
                else:
                    if LOG_FAILURES:
                        print(f"Incorrect: {filenames[left_idx]} ({model_rewards[0]:.4f}) > {filenames[right_idx]} ({model_rewards[1]:.4f})")
                    if TRACK_FAILURES:
                        failure_per_idx[right_idx] += 1
            else:
                # preference == 2 (tie): no failure counting; optional logging only
                if LOG_SUCCESS:
                    print(f"Tie: {filenames[left_idx]} ({model_rewards[0]:.4f}) ~ {filenames[right_idx]} ({model_rewards[1]:.4f})")

        if TRACK_FAILURES:
            # Iterate over all the files and add the failures for those that failed
            for i in range(len(filenames)):
                FAILURE_TRACK_PROGRESS[i].append(failure_per_idx[i])
            print("Failure tracking:")
            for i in FAILURE_TRACK_PROGRESS:
                print(f"{filenames[i]}: {FAILURE_TRACK_PROGRESS[i]}")

    return total_loss, acc

def nn_bradley_terry_loss(model, python_model, comparisons, task, filenames, data_folder, verbose_accururacy=False):
    loss_fn = nn.CrossEntropyLoss()
    input_keys = get_reward_input_keys(python_model)
    
    # First load all rollout data
    rollout_data_full = {}
    for i, path in enumerate(filenames):
        with open(os.path.join(data_folder, path), 'r') as f:
            f.readline()  # Skip score line
            rollout_data_full[i] = len([line for line in f])
            rollout_data_full[i] = convert_file_length_to_rollout_length(rollout_data_full[i], task)
    
    rollout_rewards = {}

    cached_nn_observations = {}
    cached_observations = {}
    for idx in range(len(comparisons)):
        i, j = int(comparisons[idx, 0]), int(comparisons[idx, 1])
        min_length = min(rollout_data_full[i], rollout_data_full[j])

        for k in [i, j]:
            if k not in cached_nn_observations:
                # Cache the full observation sequence
                try:
                    cached_nn_observations[k] = get_rollout_observations(os.path.join(data_folder, filenames[k]), task, nn=True)
                    cached_observations[k] = get_rollout_observations(os.path.join(data_folder, filenames[k]), task, input_keys)
                except Exception as e:
                    print(f"Error loading observations for {filenames[k]}: {e}")
                    # cached_nn_observations[k] = []
                    continue

            
            key = (k, min_length)
            if key not in rollout_rewards:
                nn_inputs = cached_nn_observations[k][:min_length]
                py_inputs = cached_observations[k][:min_length]
                total_reward = torch.tensor(0.0, requires_grad=True)
                for indx in range(len(nn_inputs)):
                    nn_inp = nn_inputs[indx]
                    py_inp = py_inputs[indx]
                    nn_reward = model(nn_inp["obs_buf"]) # consider tanh
                    py_reward, _ = python_model(**py_inp) # tanh
                    total_reward = total_reward + nn_reward + py_reward
                # rollout_rewards[key] = total_reward
                rollout_rewards[key] = total_reward / len(nn_inputs)  # Average over the sequence length to prevent mode

    # left = torch.stack([rollout_rewards[int(i)].squeeze() for i in comparisons[:, 0]])
    # right = torch.stack([rollout_rewards[int(i)].squeeze() for i in comparisons[:, 1]])
    # left = torch.stack([rollout_rewards[(int(i), min(rollout_data_full[int(i)], rollout_data_full[int(j)]))].squeeze() for i, j in comparisons])
    # right = torch.stack([rollout_rewards[(int(j), min(rollout_data_full[int(i)], rollout_data_full[int(j)]))].squeeze() for i, j in comparisons])
    left = torch.stack([
        rollout_rewards[(int(row[0]), min(rollout_data_full[int(row[0])], rollout_data_full[int(row[1])]))].squeeze()
        for row in comparisons
    ])
    right = torch.stack([
        rollout_rewards[(int(row[1]), min(rollout_data_full[int(row[0])], rollout_data_full[int(row[1])]))].squeeze()
        for row in comparisons
    ])

    logits = torch.stack([left, right], dim=1)
    targets = comparisons[:, -1].long()

    with torch.no_grad():
        acc = (torch.argmax(logits, dim=1) == targets).float().mean()
        print(f"Pairwise accuracy: {acc.item():.2f}")

    if verbose_accururacy:
        if TRACK_FAILURES:
            failure_per_idx = defaultdict(int)

        for i in range(len(comparisons)):
            left_idx = int(comparisons[i, 0])
            right_idx = int(comparisons[i, 1])
            preference = int(comparisons[i, 2])

            model_rewards = torch.stack([left[i], right[i]])
            if preference == 0:
                if model_rewards[0] > model_rewards[1]:
                    if LOG_SUCCESS:
                        print(f"Correct: {filenames[left_idx]} ({model_rewards[0]:.4f}) > {filenames[right_idx]} ({model_rewards[1]:.4f})")
                else:
                    if LOG_FAILURES:
                        print(f"Incorrect: {filenames[left_idx]} ({model_rewards[0]:.4f}) < {filenames[right_idx]} ({model_rewards[1]:.4f})")   
                    if TRACK_FAILURES:
                        failure_per_idx[left_idx] += 1
            else:
                if model_rewards[0] < model_rewards[1]:
                    if LOG_SUCCESS:
                        print(f"Correct: {filenames[left_idx]} ({model_rewards[0]:.4f}) < {filenames[right_idx]} ({model_rewards[1]:.4f})")
                else:
                    if LOG_FAILURES:
                        print(f"Incorrect: {filenames[left_idx]} ({model_rewards[0]:.4f}) > {filenames[right_idx]} ({model_rewards[1]:.4f})")
                    if TRACK_FAILURES:
                        failure_per_idx[right_idx] += 1
        if TRACK_FAILURES:
            # Iterate over all the files and add the failures for those that failed
            for i in range(len(filenames)):
                FAILURE_TRACK_PROGRESS[i].append(failure_per_idx[i])
            print("Failure tracking:")
            for i in FAILURE_TRACK_PROGRESS:
                print(f"{filenames[i]}: {FAILURE_TRACK_PROGRESS[i]}")


    return loss_fn(logits, targets), acc

    # # Handle tri-state targets: 0/1 => CE as before; 2 => MSE(left, right)
    # device = left.device
    # mask_tie = (targets == 2)
    # mask_ce = ~mask_tie

    # ce_loss = torch.tensor(0.0, device=device)
    # mse_loss = torch.tensor(0.0, device=device)
    # n_ce = int(mask_ce.sum().item())
    # n_mse = int(mask_tie.sum().item())

    # if n_ce > 0:
    #     ce_loss = loss_fn(logits[mask_ce], targets[mask_ce])  # mean over 0/1 samples
    # if n_mse > 0:
    #     mse_loss = F.mse_loss(left[mask_tie], right[mask_tie], reduction='mean')  # mean over tie samples

    # if (n_ce + n_mse) > 0:
    #     base_loss = (ce_loss * n_ce + mse_loss * n_mse) / (n_ce + n_mse)
    # else:
    #     base_loss = torch.tensor(0.0, device=device)

    # total_loss = base_loss

    # with torch.no_grad():
    #     if n_ce > 0:
    #         predictions = torch.argmax(logits[mask_ce], dim=1)
    #         acc = (predictions == targets[mask_ce]).float().mean()
    #         print(f"Pairwise accuracy: {acc.item():.2f}, Base loss: {base_loss.item():.4f}")
    #     else:
    #         print(f"Pairwise accuracy: N/A (only ties), Base loss: {base_loss.item():.4f}")

    # if verbose_accururacy:
    #     if TRACK_FAILURES:
    #         failure_per_idx = defaultdict(int)

    #     for i in range(len(comparisons)):
    #         left_idx = int(comparisons[i, 0])
    #         right_idx = int(comparisons[i, 1])
    #         preference = int(comparisons[i, 2])

    #         model_rewards = torch.stack([left[i], right[i]])
    #         if preference == 0:
    #             if model_rewards[0] > model_rewards[1]:
    #                 if LOG_SUCCESS:
    #                     print(f"Correct: {filenames[left_idx]} ({model_rewards[0]:.4f}) > {filenames[right_idx]} ({model_rewards[1]:.4f})")
    #             else:
    #                 if LOG_FAILURES:
    #                     print(f"Incorrect: {filenames[left_idx]} ({model_rewards[0]:.4f}) < {filenames[right_idx]} ({model_rewards[1]:.4f})")   
    #                 if TRACK_FAILURES:
    #                     failure_per_idx[left_idx] += 1
    #         elif preference == 1:
    #             if model_rewards[0] < model_rewards[1]:
    #                 if LOG_SUCCESS:
    #                     print(f"Correct: {filenames[left_idx]} ({model_rewards[0]:.4f}) < {filenames[right_idx]} ({model_rewards[1]:.4f})")
    #             else:
    #                 if LOG_FAILURES:
    #                     print(f"Incorrect: {filenames[left_idx]} ({model_rewards[0]:.4f}) > {filenames[right_idx]} ({model_rewards[1]:.4f})")
    #                 if TRACK_FAILURES:
    #                     failure_per_idx[right_idx] += 1
    #         else:
    #             # preference == 2 (tie): no failure counting; optional logging only
    #             if LOG_SUCCESS:
    #                 print(f"Tie: {filenames[left_idx]} ({model_rewards[0]:.4f}) ~ {filenames[right_idx]} ({model_rewards[1]:.4f})")

    #     if TRACK_FAILURES:
    #         # Iterate over all the files and add the failures for those that failed
    #         for i in range(len(filenames)):
    #             FAILURE_TRACK_PROGRESS[i].append(failure_per_idx[i])
    #         print("Failure tracking:")
    #         for i in FAILURE_TRACK_PROGRESS:
    #             print(f"{filenames[i]}: {FAILURE_TRACK_PROGRESS[i]}")

    # return total_loss, acc


def wrap_reward_module(code_string: str, param_names: dict, module_name="DynamicReward"):
    import_lines = []
    method_lines = []

    for line in code_string.strip().splitlines():
        if line.strip().startswith("import") or line.strip().startswith("from"):
            import_lines.append(line.strip())
        else:
            method_lines.append(line.rstrip())

    # Find the method definition (e.g., def compute_reward(self, ...))
    method_header_index = next(
        (i for i, line in enumerate(method_lines) if line.strip().startswith("def compute_reward")), -1
    )
    if method_header_index == -1:
        raise ValueError("Expected a method named `compute_reward`.")

    method_def = method_lines[method_header_index]
    method_def_line = method_lines[method_header_index].strip()

    method_args = method_def_line[
        method_def_line.index("(") + 1:method_def_line.index(")")
    ].split(",")
    method_args = [arg.strip() for arg in method_args if arg.strip() and arg.strip() != "self"]

    method_body = method_lines[method_header_index + 1:]
    indented_method = [f"    {line}" if line.strip() else "" for line in [method_def] + method_body]

    param_init = "\n".join(
        f"        self.{k} = nn.Parameter(torch.tensor({v}, dtype=torch.float32))"
        for k, v in param_names.items()
    )

    for i in range(len(method_args)):
        # If it has a colon remove it
        if ":" in method_args[i]:
            method_args[i] = method_args[i].split(":")[0].strip()

    # Create a string where it says arg=inputs[arg] for each arg
    arg_assignments = [f"{arg}= inputs['{arg}']" for arg in method_args]
    # Convert arg_assignments to a string with no surrounding brackets
    arg_assignments = ", ".join(arg_assignments)
    class_code = f"""
import torch
import torch.nn as nn
{chr(10).join(import_lines)}

class {module_name}(nn.Module):
    def __init__(self):
        super().__init__()
{param_init}

{chr(10).join(indented_method)}

    def forward(self, **inputs):
        return self.compute_reward({arg_assignments})
"""
    return class_code





def create_model_from_code(code_str: str, param_defaults: dict):
    class_code = wrap_reward_module(code_str, param_defaults)

    # DEBUG: Optional - save generated module
    print(class_code)

    # Use a shared dictionary for globals so imports persist
    exec_scope = {}
    exec(class_code, exec_scope)
    return exec_scope["DynamicReward"]()

def train_nn_model(python_model, filenames, comparisons, task: str, code_str: str, param_defaults: dict, data_folder: str, epochs=20, lr=5e-2, logger=None):
    # Now that the python reward function isn't improving anymore we add a neural network term to augment the reward function (added)
    # From this point we don't want to modify the python reward function anymore, we just want to train the neural network to augment it
    torch.manual_seed(12312)  # Set seed for reproducibility

    # Initialize the nn model
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
    
    task_to_obs_dim = {
        "ShadowHandScissors": 57, 
        "ShadowHandBottleCap": 420, 
        "ShadowHandDoorOpenInward": 417,
        "Ant": 29,
    }
    NN_Reward = nn_reward_model(obs_dim=task_to_obs_dim[task])
    optimizer = optim.Adam(NN_Reward.parameters(), lr=lr/10)
    # Shuffle the comparisons
    comparisons = comparisons[torch.randperm(comparisons.size(0))]
    # Split off val_ratio of the comparisons for validation
    # val_ratio = VALIDATION_RATIO
    # if USE_ONLY_ONE_BATCH:
    #     val_ratio = 0.5
    # validation_comparisons = comparisons[:int(len(comparisons) * val_ratio)]
    # comparisons = comparisons[int(len(comparisons) * val_ratio):]
    # input_keys = get_reward_input_keys(python_model)

    # Split off VALIDATION_SIZE of the comparisons for validation
    validation_comparisons = comparisons[:VALIDATION_SIZE]
    comparisons = comparisons[VALIDATION_SIZE:]


    if AUTOMATIC_TERMINATION:
        original_state = NN_Reward.state_dict()
        original_validation_loss = float('inf')
        original_validation_accuracy = 0.0
        best_validation_loss = float('inf')
        best_validation_accuracy = 0.0
        best_model_state = original_state
        epochs_without_improvement = 0
    if BATCH_SIZE is not None:
        # Split off a validation set to use for all epochs with BATCH_SIZE
        # if len(validation_comparisons) < BATCH_SIZE:
        #     print("Not enough validation comparisons for batch size, using all comparisons.")
        #     batch_validation_comparisons = validation_comparisons
        # else:
        #     indices = torch.randperm(len(validation_comparisons))[:BATCH_SIZE]
        #     batch_validation_comparisons = validation_comparisons[indices]

        batch_validation_comparisons = validation_comparisons

    # Temporary overfit testing
    # batch_initialized = False
    for i in range(epochs):
        if BATCH_SIZE is not None:
            # Split off BATCH_SIZE data points from comparisons and use that for the next epoch
            if len(comparisons) < BATCH_SIZE:
                print("Not enough comparisons for batch size, using all comparisons.")
                batch_comparisons = comparisons
            else:
                # if not batch_initialized:
                    indices = torch.randperm(len(comparisons))[:BATCH_SIZE]
                    batch_comparisons = comparisons[indices]
                    batch_initialized = True
        
        optimizer.zero_grad()
        loss, accuracy = nn_bradley_terry_loss(NN_Reward, python_model, batch_comparisons, task, filenames, data_folder, verbose_accururacy=(i % 10 == 0))

        with torch.no_grad():
            val_loss, val_accuracy = nn_bradley_terry_loss(NN_Reward, python_model, batch_validation_comparisons, task, filenames, data_folder)
        if i == 0 and AUTOMATIC_TERMINATION:
            original_validation_loss = val_loss.item()
            original_validation_accuracy = val_accuracy
            # print(f"Original Validation Loss: {original_validation_loss:.4f}, Original Validation Accuracy: {original_validation_accuracy:.4f}")
            # print(f"Validation Loss: {val_loss.item():.4f}")

        print(f"Epoch {i+1}/{epochs}, Train Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

        loss.backward()
        optimizer.step()

        # Check for best validation loss
        if AUTOMATIC_TERMINATION and val_loss.item() < best_validation_loss:
            best_validation_loss = val_loss.item()
            best_validation_accuracy = val_accuracy

            best_model_state = NN_Reward.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 10:
                print("Early stopping triggered due to no improvement in validation loss.")
                break

    if AUTOMATIC_TERMINATION:
        if best_model_state is not None:
            NN_Reward.load_state_dict(best_model_state)
            print(f"Loaded best model state with validation loss: {best_validation_loss:.4f}")
        else:
            NN_Reward.load_state_dict(original_state)
            print("No improvement in validation loss, using original NN.") # We shouldn't use any NN in this case
            
    if (logger is not None) and AUTOMATIC_TERMINATION:
        logger.info(f"Original Validation Loss: {original_validation_loss:.4f}, Original Validation Accuracy: {original_validation_accuracy:.4f}")
        logger.info(f"Final Validation Loss: {best_validation_loss:.4f}, Final Validation Accuracy: {best_validation_accuracy:.4f}")
    elif AUTOMATIC_TERMINATION:
        print(f"Original Validation Loss: {original_validation_loss:.4f}, Original Validation Accuracy: {original_validation_accuracy:.4f}")
        print(f"Final Validation Loss: {best_validation_loss:.4f}, Final Validation Accuracy: {best_validation_accuracy:.4f}")

    return NN_Reward

def train_python_model(model, filenames, comparisons, task: str, code_str: str, param_defaults: dict, data_folder: str, epochs=20, lr=5e-2, logger=None):
    try:
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # raise ValueError("This is a test error to check the error handling in the training function.")

        # Set torch randperm seed for reproducibility
        torch.manual_seed(0)
        # Shuffle the comparisons
        comparisons = comparisons[torch.randperm(comparisons.size(0))]
        # Split off 20% of the comparisons for validation
        # val_ratio = VALIDATION_RATIO
        # if USE_ONLY_ONE_BATCH:
        #     val_ratio = 0.5
        # validation_comparisons = comparisons[:int(len(comparisons) * val_ratio)]
        # comparisons = comparisons[int(len(comparisons) * val_ratio):]

        # Split off VALIDATION_SIZE of the comparisons for validation
        validation_comparisons = comparisons[:VALIDATION_SIZE]
        comparisons = comparisons[VALIDATION_SIZE:]

        # input_keys = get_reward_input_keys(model)
        # rollout_data = {
        #     i: get_rollout_observations(os.path.join(data_folder, path), input_keys)
        #     for i, path in enumerate(filenames)
        # }
        
        # print(f"Initial Loss: {bradley_terry_loss(model, comparisons, filenames, data_folder, verbose_accururacy=True)}")
        if AUTOMATIC_TERMINATION:
            original_state = model.state_dict()
            original_validation_loss = float('inf')
            original_validation_accuracy = 0.0
            best_validation_loss = float('inf')
            best_validation_accuracy = 0.0
            best_model_state = original_state
            epochs_without_improvement = 0

        if BATCH_SIZE is not None:
            # # Split off a validation set to use for all epochs with BATCH_SIZE
            # if len(validation_comparisons) < BATCH_SIZE:
            #     print("Not enough validation comparisons for batch size, using all comparisons.")
            #     batch_validation_comparisons = validation_comparisons
            # else:
            #     indices = torch.randperm(len(validation_comparisons))[:BATCH_SIZE]
            #     batch_validation_comparisons = validation_comparisons[indices]

            batch_validation_comparisons = validation_comparisons

        for i in range(epochs):

            if BATCH_SIZE is not None:
                # Split off BATCH_SIZE data points from comparisons and use that for the next epoch
                if len(comparisons) < BATCH_SIZE:
                    print("Not enough comparisons for batch size, using all comparisons.")
                    batch_comparisons = comparisons
                else:
                    indices = torch.randperm(len(comparisons))[:BATCH_SIZE]
                    batch_comparisons = comparisons[indices]



            optimizer.zero_grad()
            loss, accuracy = bradley_terry_loss(model, batch_comparisons, task, filenames, data_folder, verbose_accururacy=(i % 10 == 0))
            
            
            # If MAXIMIZE_LOSS is True, we need to negate the loss
            if MAXIMIZE_LOSS:
                loss = -loss

            # Calculate the validation loss
            with torch.no_grad():
                val_loss, val_accuracy = bradley_terry_loss(model, batch_validation_comparisons, task, filenames, data_folder)
            
            if i == 0 and AUTOMATIC_TERMINATION:
                original_validation_loss = val_loss.item()
                original_validation_accuracy = val_accuracy
                # print(f"Original Validation Loss: {original_validation_loss:.4f}, Original Validation Accuracy: {original_validation_accuracy:.4f}")
                # print(f"Validation Loss: {val_loss.item():.4f}")
            print(f"Epoch {i+1}/{epochs}, Train Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

            loss.backward()
            optimizer.step()

            # Check for best validation loss
            if AUTOMATIC_TERMINATION and val_loss.item() < best_validation_loss:
                best_validation_loss = val_loss.item()
                best_validation_accuracy = val_accuracy

                best_model_state = model.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 10:
                    print("Early stopping triggered due to no improvement in validation loss.")
                    break

            if VERBOSE_PARAMETER_TRACKING:
                if i % 10 == 0:
                    print("Learned parameters:")
                    for name, param in model.named_parameters():
                        print(f"{name}: {param.item()}")

        if AUTOMATIC_TERMINATION:
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print(f"Loaded best model state with validation loss: {best_validation_loss:.4f}")
            else:
                model.load_state_dict(original_state)
                print("No improvement in validation loss, using original model state.")

        print("Learned parameters:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.item()}")

        if (logger is not None) and AUTOMATIC_TERMINATION:
            logger.info(f"Original Validation Loss: {original_validation_loss:.4f}, Original Validation Accuracy: {original_validation_accuracy:.4f}")
            logger.info(f"Final Validation Loss: {best_validation_loss:.4f}, Final Validation Accuracy: {best_validation_accuracy:.4f}")
        elif AUTOMATIC_TERMINATION:
            print(f"Original Validation Loss: {original_validation_loss:.4f}, Original Validation Accuracy: {original_validation_accuracy:.4f}")
            print(f"Final Validation Loss: {best_validation_loss:.4f}, Final Validation Accuracy: {best_validation_accuracy:.4f}")

        return model
    except Exception as e:
        print(f"An error occurred during tuning: {e}")
        if RAISE_ERRORS:
            raise e
        print("Using the original model state.")
        # raise e
        # Create a class with param_defaults as the attributes and return that as tensors
        return_class = type("DynamicReward", (nn.Module,), {})
        for key, value in param_defaults.items():
            setattr(return_class, key, torch.tensor(value, dtype=torch.float32, requires_grad=True))
        
        return return_class()
    
def train_reward_model(task: str, code_str: str, param_defaults: dict, data_folder: str, epochs=20, lr=5e-2, logger=None):
    try:
        code_str = code_str.replace("-> Tuple[torch.Tensor, Dict[str, torch.Tensor]]","")
        code_str = code_str.replace("compute_reward(", "compute_reward(self,")
        model = create_model_from_code(code_str, param_defaults)
        filenames, comparisons = get_preference_pairs(data_folder, task)
    except Exception as e:
        print(f"Error creating model from code: {e}")
        if RAISE_ERRORS:
            raise e
        # If there is an error here then there's nothing we can do, so we just return a model with the default parameters
        model = type("DynamicReward", (nn.Module,), {})
        for key, value in param_defaults.items():
            setattr(model, key, torch.tensor(value, dtype=torch.float32, requires_grad=True))
        return model
    
    # Noise insertion experiment
    if NOISE_INSERTION is not None:
        # Add noise to the comparisons
        print(f"Adding noise to the comparisons with a scale of {NOISE_INSERTION}")
        noise_scale = NOISE_INSERTION
        for comparison in comparisons:
            flip_label = torch.rand(1).item() < noise_scale
            if flip_label:
                # Flip the label
                if comparison[2] == 0:
                    comparison[2] = 1
                elif comparison[2] == 1:
                    comparison[2] = 0
                # else: # Tie, do nothing

    # Train python rw func
    python_model = train_python_model(model=model, filenames=filenames, comparisons=comparisons, task=task, code_str=code_str, param_defaults=param_defaults, data_folder=data_folder, epochs=epochs, lr=lr, logger=logger)
    # python_model = model
    # Train nn rw func on top of the python rw func
    nn_model = train_nn_model(python_model=python_model, filenames=filenames, comparisons=comparisons, task=task, code_str=code_str, param_defaults=param_defaults, data_folder=data_folder, epochs=epochs, lr=lr, logger=logger)

    # Save the final nn model's .pt file (all info not just the weights)
    if SAVE_FINAL_MODEL:
        # model_path = os.path.join(data_folder, f"{task}_reward_model.pt")
        # torch.save(nn_model.state_dict(), model_path)
        # print(f"Saved final model to {model_path}")
#         model.load_state_dict(state_dict, strict=True)   # strict=False if you *really* want to ignore extras
    # model.eval()

    # # ---- script/trace ----
    # ts_model = (
    #     torch.jit.trace(model, example_input) if use_trace
    #     else torch.jit.script(model)
    # )

    # # ---- save ----
    # ts_model.save(pt_path)
    # print(f"TorchScript model saved ➜  {pt_path}")
        model.eval()
        task_to_obs_dim = {
            "ShadowHandScissors": 57,
            "ShadowHandBottleCap": 420,
            "ShadowHandDoorOpenInward": 417,
            "Ant" :29,
        }
        ts_model = torch.jit.trace(nn_model, torch.randn(1, task_to_obs_dim[task]))  # Assuming the input is a tensor of shape (1, 57) for ShadowHandScissors
        model_path = os.path.join(data_folder, f"{task}_reward_model.ptt")
        ts_model.save(model_path)
        print(f"Saved final model to {model_path}")

# Example when running script directly
if __name__ == "__main__":
#     reward_code = '''
# import torch
# from typing import Dict, Tuple

# def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, rot_diff_temp: torch.Tensor, success_threshold: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#     rot_diff = torch.sum((object_rot - goal_rot) ** 2, dim=-1)
#     rot_diff_reward = torch.exp(-rot_diff_temp * rot_diff)
#     success_reward = (rot_diff < success_threshold).float()
#     total_reward = rot_diff_reward + success_reward
#     return total_reward, {"rot_diff_reward": rot_diff_reward, "success_reward": success_reward}
# '''

    reward_code = '''
def compute_reward(object_rot, goal_rot):
    # compute the cosine similarity between object's current orientation and the target orientation
    dist_penalty_scaler = self.dist_penalty_scaler
    reward_temp = self.reward_temp
    garbage_term_scaler = self.garbage_term_scaler
    survival_scaler = self.survival_scaler

    similarity = torch.nn.functional.cosine_similarity(object_rot, goal_rot, dim=-1)

    # transform similarity to a distance-like metric
    distance = 1 - similarity

    # dist_penalty_scaler = self.dist_penalty_scaler # This should learn to be negative if the algo works

    # larger reward the smaller the rotation difference
    reward = dist_penalty_scaler * distance #LLM had -1 instead of the dist_penalty_scaler

    # temperature parameter adjusted for reward scaling
    # reward_temp = reward_temp # LLM set this to 1
    reward_temp = torch.clamp(reward_temp, min=0.5, max=15) # prevent explosion

    # scale the raw reward using an exponential function
    # scaled_reward = torch.exp(torch.clamp(reward / reward_temp, min=-50, max=50)) # prevent gradient explosion NECESSARY
    # safe_reward = reward / reward_temp
    # safe_reward = safe_reward.clone().detach().requires_grad_(True)  # Prevent in-place issues
    # scaled_reward = torch.nn.functional.softplus(torch.clamp(safe_reward, min=-10, max=10)) # torch.exp had grad issues
    safe_reward = reward / reward_temp  # this keeps gradient flow
    safe_reward = torch.clamp(safe_reward, min=-10, max=10)
    scaled_reward = torch.nn.functional.softplus(safe_reward)

    # Survival Reward, just adds a constant equal to the parameter, encourages longer rollouts
    survival_reward = self.survival_scaler
    # for now try with a pure noise reward
    # noise_reward = torch.randn(1) * garbage_term_scaler
    scaled_reward += survival_reward
    scaled_reward += garbage_term_scaler * torch.randn_like(scaled_reward) # Ideally this gets silenced as well?

    return scaled_reward, {}
'''

    reward_code = '''
def compute_reward(object_rot: torch. Tensor, goal_rot: torch. Tensor, object_angvel: torch. Tensor, object_pos: torch. Tensor, fingertip_pos: torch.Tensor):
    
    rot_diff = torch.abs(torch.sum(object_rot * goal_rot, dim=1) - 1) / 2
    rotation_reward_temp = self.rotation_reward_temp
    rotation_reward = torch.exp(-rotation_reward_temp * rot_diff)

    # Angular velocity penalty
    angvel_norm = torch.norm(object_angvel, dim=1)
    angvel_threshold = self.angvel_threshold
    angvel_penalty_temp = self.angvel_penalty_temp
    angular_velocity_penalty = torch.where(angvel_norm > angvel_threshold, torch.exp(-angvel_penalty_temp * (angvel_norm - angvel_threshold)), torch.zeros_like(angvel_norm))
    
    # Distance reward
    min_distance_temp = self.min_distance_temp
    min_distance = torch.min(torch.norm(fingertip_pos - object_pos[:, None], dim=2), dim=1).values
    uncapped_distance_reward = torch.exp(-min_distance_temp * min_distance) 
    distance_reward = torch.clamp(uncapped_distance_reward, 0.0, 1.0)

    total_reward = rotation_reward - angular_velocity_penalty + distance_reward

    reward_components = {
        "rotation_reward": rotation_reward,
        "angular_velocity_penalty": angular_velocity_penalty, 
        "distance_reward": distance_reward
    }
    return total_reward, reward_components'''

    param_defaults = {
        "dist_penalty_scaler": -0.9,
        "reward_temp": 1.0,
        "garbage_term_scaler": 0.001,
        "survival_scaler": 0.1,
    }

    param_defaults = {
        "rotation_reward_temp": 20.0,
        "angvel_threshold": 2.0,
        "angvel_penalty_temp": 2.0,
        "min_distance_temp": 10.0,
    }
    
    param_defaults = {
        "rotation_reward_temp": 40.47,
        "angvel_threshold": -3.98,
        "angvel_penalty_temp": 9.21,
        "min_distance_temp": -5.69,
    }

    # model = train_reward_model(
    #     code_str=reward_code,
    #     param_defaults=param_defaults,
    #     data_folder="./preference_data",
    #     epochs=45,
    #     lr=0.5
    # )
    # exit()

    reward_code = '''
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, potentials: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute distance between ant's current position and its forward target
    torso_position = root_states[:, 0:3]
    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    # Compute progress towards the forward target
    prev_potentials_new = potentials.clone()
    progress = -torch.norm(to_target, p=2, dim=1) / dt

    # Calculate the step reward for forward progress (negative distance to target)
    forward_reward = progress - prev_potentials_new
    forward_reward_temperature = self.forward_reward_temperature  # Added temperature for forward_reward scaling
    forward_normalized_reward = torch.exp(forward_reward / forward_reward_temperature)
    
    # print("progress:", progress)
    # print("prev_potentials_new:", prev_potentials_new)
    # print("forward_reward:", forward_reward)
    # print("forward_normalized_reward:", forward_normalized_reward)

    # Compute a reward component for the current velocity
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    forward_velocity_temperature = self.forward_velocity_temperature  # Adjusted temperature for velocity_reward scaling
    forward_velocity_normalized_reward = torch.exp(forward_velocity / forward_velocity_temperature)

    # Add a penalty term for the agent's body height deviation from the target height
    target_height = self.target_height
    height_penalty = torch.abs(torso_position[:, 2] - target_height)
    height_penalty_temperature = self.height_penalty_temperature  # Adjusted temperature for height_penalty scaling
    height_normalized_penalty = torch.exp(-height_penalty / height_penalty_temperature)

    # Compute total reward and individual reward components
    reward = forward_normalized_reward * forward_velocity_normalized_reward * height_normalized_penalty
    reward_components = {
        "forward_reward": forward_normalized_reward,
        "velocity_reward": forward_velocity_normalized_reward,
        "height_penalty": height_normalized_penalty
    }
    # print(f"Reward Components: {reward_components}")
    return reward, reward_components'''

    reward_code = '''
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Scalar weights and parameters (these will become trainable)
    speed_weight = self.speed_weight   # Increase weight for speed as it's most important
    direction_weight = self.direction_weight # Weight for direction
    speed_temp = self.speed_temp  # Temperature parameter for speed sensitivity
    direction_temp = self.direction_temp  # Temperature parameter for direction sensitivity
    distance_threshold = self.distance_threshold  # Success threshold for progressing forward distance

    # Get the velocity of the ant
    velocity = root_states[:, 7:10]  
    ant_forward_velocity = velocity[:, 1] 

    # Computation of speed reward 
    speed_reward = torch.exp(-speed_temp * (1.0 - ant_forward_velocity))

    # Computation of direction reward (reward forward progress)
    forward_progress = potentials - prev_potentials
    direction_reward = (forward_progress > distance_threshold).float()

    # Increase the weights of forward direction
    direction_reward *= direction_weight

    # Combine the rewards components with corresponding weights
    total_reward = speed_weight * speed_reward + direction_weight * direction_reward

    # Return total reward and individual reward components in a dictionary
    rewards_dict = {'speed_reward': speed_reward, 'direction_reward': direction_reward}
    return total_reward, rewards_dict
'''


    # param_defaults = {
    #     "forward_reward_temperature": 5.0, # Started as 0.1, Passed as 10.0
    #     "forward_velocity_temperature": 10.0, # Started as 1.0, Passed as 10.0
    #     "target_height": -0.4, # Started as 0.4, Passed as 0.4
    #     "height_penalty_temperature": -0.1, # Started as 0.1, Passed as 0.1
    # }
    
    param_defaults = {
        "speed_weight": 2.0, 
        "direction_weight": 1.0, 
        "speed_temp": 0.05, 
        "direction_temp": 0.1, 
        "distance_threshold": 0.1
    }

#     reward_code = '''
# def compute_reward(scissors_right_handle_pos: torch.Tensor, scissors_left_handle_pos: torch.Tensor, right_hand_pos: torch.Tensor, left_hand_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

#     target_opened_distance = self.target_opened_distance  # Set the desired opened distance between the scissors" handles when opened
#     opened_reward_temp = self.opened_reward_temp  # Change the value to adjust the sensitivity of the opened scissors reward component

#     # Calculate the distance between the right and left handles of the scissors
#     handle_distance = torch.norm(scissors_right_handle_pos - scissors_left_handle_pos, dim=-1)

#     # Calculate the reward based on the opened distance of the scissors
#     opened_reward = torch.exp(opened_reward_temp * (handle_distance - target_opened_distance))

#     # Calculate the distance between the hands and the corresponding handles of the scissors
#     right_hand_to_handle_dist = torch.norm(right_hand_pos - scissors_right_handle_pos, dim=-1)
#     left_hand_to_handle_dist = torch.norm(left_hand_pos - scissors_left_handle_pos, dim=-1)

#     # Penalize the agent if the hands are too far from the handles
#     handle_reaching_penalty = 0.5 * (right_hand_to_handle_dist + left_hand_to_handle_dist)

#     # Calculate the total reward
#     total_reward = opened_reward - handle_reaching_penalty

#     # Log individual rewards for debugging
#     reward_info = {
#         "opened_reward": opened_reward,
#         "handle_reaching_penalty": handle_reaching_penalty
#     }

#     return total_reward, reward_info'''

#     param_defaults = {
#         "target_opened_distance": 0.3,
#         "opened_reward_temp": 5.0,
#     }

############ Fully Flipped???

#     reward_code = '''
# def compute_reward(left_hand_pos: torch.Tensor, right_hand_rf_pos: torch.Tensor, bottle_pos: torch.Tensor, bottle_cap_pos: torch.Tensor, bottle_cap_up: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#     # Scalar weights and parameters
#     left_hand_bottle_weight = self.left_hand_bottle_weight
#     right_fingertip_cap_weight = self.right_fingertip_cap_weight
#     cap_orientation_weight = self.cap_orientation_weight

#     # Squared distances between the hand and bottle, and right fingertip and cap. Less is better.
#     left_hand_bottle_dist = torch.sum((left_hand_pos - bottle_pos)**2, dim=-1)
#     right_hand_fingertip_cap_dist = torch.sum((right_hand_rf_pos - bottle_cap_pos)**2, dim=-1)

#     # Reward based on the vertical orientation of the cap. We want up direction to align with world's up direction (0, 0, 1)
#     cap_orientation = bottle_cap_up @ torch.tensor([0, 0, 1], device=bottle_cap_up.device, dtype=bottle_cap_up.dtype)

#     # It's good if left hand is is near to bottle, and right hand fingertip is near to cap,
#     # and the cap orientation is aligned with the world's up direction.
#     reward = (left_hand_bottle_weight * torch.exp(-left_hand_bottle_dist) + 
#               right_fingertip_cap_weight * torch.exp(-right_hand_fingertip_cap_dist) + 
#               cap_orientation_weight * cap_orientation)

#     components = {"left_hand_bottle_reward": torch.exp(-left_hand_bottle_dist),
#                   "right_hand_fingertip_cap_reward": torch.exp(-right_hand_fingertip_cap_dist),
#                   "cap_orientation_reward": cap_orientation}

#     return reward, components
# '''

#     param_defaults = {
#         "left_hand_bottle_weight": 1.0,
#         "right_fingertip_cap_weight": 1.0,
#         "cap_orientation_weight": 1.0,
#     }

#     reward_code = '''
# def compute_reward(bottle_cap_pos: torch.Tensor, bottle_pos: torch.Tensor, right_hand_pos: torch.Tensor, left_hand_pos: torch.Tensor, goal_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#     device = bottle_cap_pos.device

#     # Distance between the right hand and bottle cap
#     dist_right_hand_to_cap = torch.norm(right_hand_pos - bottle_cap_pos, dim=1)

#     # Distance between the left hand and bottle
#     dist_left_hand_to_bottle = torch.norm(left_hand_pos - bottle_pos, dim=1)

#     # Distance between the bottle cap and goal position
#     dist_cap_to_goal = torch.norm(bottle_cap_pos - goal_pos, dim=1)

#     # Penalize large distances between the hands
#     handdistance_reward_raw = -dist_right_hand_to_cap - dist_left_hand_to_bottle

#     # Apply transformation to handdistance_reward_raw
#     hand_distance_temperature = self.hand_distance_temperature
#     hand_distance_transformed_reward = torch.exp(handdistance_reward_raw / hand_distance_temperature)

#     # Penalize large distances between the bottle cap and goal position
#     cap_goal_distance_reward = -dist_cap_to_goal

#     # Combine individual reward components
#     total_reward = hand_distance_transformed_reward + cap_goal_distance_reward

#     # Create a dictionary to store individual reward components
#     rewards_dict = {
#         "hand_distance_transformed_reward": hand_distance_transformed_reward,
#         "cap_goal_distance_reward": cap_goal_distance_reward,
#     }

#     return total_reward, rewards_dict'''

#     param_defaults = {
#         "hand_distance_temperature": 50.0,
#     }

#     reward_code = '''
# def compute_reward(goal_pos: torch.Tensor, door_left_handle_pos: torch.Tensor, door_right_handle_pos: torch.Tensor, right_hand_pos: torch.Tensor, left_hand_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#     # Define scalar constants
#     reaching_weight = 1.0  # Reward for reaching the door handle
#     grasping_weight = 1.0  # Reward for grasping the door handle
#     goal_weight = 2.0      # Reward for moving the handle towards the goal
    
#     reaching_temp = 0.05   # Temperature for reaching reward sensitivity
#     grasping_temp = 0.05   # Temperature for grasping reward sensitivity
#     goal_temp = 0.02       # Temperature for goal reward sensitivity
    
#     grasping_threshold = 0.02  # Threshold for successful grasp
#     reaching_threshold = 0.05  # Threshold for successfully reaching the handle
#     goal_distance_threshold = 0.05  # Success threshold for the goal
    
#     # Calculate distance from the handles to hands and goal
#     handle_hand_dist = torch.min(
#         torch.norm(door_left_handle_pos - left_hand_pos, dim=-1),
#         torch.norm(door_right_handle_pos - right_hand_pos, dim=-1)
#     )
#     goal_distance = torch.norm(goal_pos - door_left_handle_pos - door_right_handle_pos, dim=-1)

#     # Calculate rewards for reaching the handle and moving it towards the goal
#     reaching_reward = torch.exp(-reaching_temp * handle_hand_dist)
#     goal_reward = torch.exp(-goal_temp * goal_distance)

#     # Calculate reward for grasping the handle
#     grasping_reward = torch.where(handle_hand_dist < grasping_threshold, 1.0, 0.0)

#     # Combine rewards, giving higher weight to moving the handle towards the goal
#     total_reward = reaching_weight * reaching_reward + grasping_weight * grasping_reward + goal_weight * goal_reward

#     rewards_dict = {'reaching_reward': reaching_reward, 'grasping_reward': grasping_reward, 'goal_reward': goal_reward}

#     return total_reward, rewards_dict
# '''

#     param_defaults = {
#         "reaching_weight": 1.0,
#         "grasping_weight": 1.0,
#         "goal_weight": 2.0,
#         "reaching_temp": 0.05,
#         "grasping_temp": 0.05,
#         "goal_temp": 0.02,
#         "grasping_threshold": 0.02,
#         "reaching_threshold": 0.05,
#         "goal_distance_threshold": 0.05
#     }

#     reward_code = '''
# def compute_reward() -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#     reward = 0.0
#     reward_components = {}
#     return reward, reward_components'''

#     param_defaults = {}

    reward_code = '''
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Scalar weights and parameters
    velocity_weight = 3.522552 # weight for velocity reward component
    velocity_temp = 0.657764 # temperature parameter for velocity sensitivity
    velocity_threshold = 3.274507 # success threshold for desired velocity
    inactivity_threshold = 0.169638 # penalty threshold for inactivity

    # Compute velocity from the root_states
    velocity = root_states[:, 7:10]

    # Compute the velocity in the forward direction
    to_target = targets - root_states[:, 0:3]
    to_target_norm = torch.norm(to_target, p=2, dim=-1, keepdim=True)
    to_target_normalized = to_target / to_target_norm
    forward_velocity = torch.sum(velocity * to_target_normalized, dim=-1)

    # Compute velocity reward and inactivity penalty
    velocity_reward = torch.sigmoid(velocity_temp * (forward_velocity - velocity_threshold))
    inactivity_penalty = torch.sigmoid(-velocity_temp * (forward_velocity - inactivity_threshold))

    # Compute total reward
    total_reward = velocity_weight * velocity_reward - inactivity_penalty
    
    # Return total reward and individual reward components
    return total_reward, {"velocity_reward": velocity_reward, "inactivity_penalty": inactivity_penalty}
'''

    param_defaults = {"empty_param": 0.0}

    model = train_reward_model(
        task="Ant",
        # task="ShadowHandScissors",
        # task="ShadowHandBottleCap",
        # task="ShadowHandDoorOpenInward",
        code_str=reward_code,
        param_defaults=param_defaults,
        # data_folder="./preference_data_ant",
        # data_folder="./auto_preference_data",
        data_folder="./ant_data_body",
        # data_folder="./auto_preference_data_exp13_scissor_test",
        epochs=45,
        lr=0.1
    )
    print("Done")
