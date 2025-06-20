# train_dynamic_reward.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
import inspect
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
AUTOMATIC_TERMINATION = True # If True, the training process will automatically terminate if the validation loss does not improve for 10 epochs, best model parameters will be returned
BATCH_SIZE = 64 # Batch size for training, if set to None, the entire dataset will be used as a batch
RAISE_ERRORS = False # If True, errors will be raised during training, if False, errors will be caught and printed
MAX_ROLLOUT_LENGTH = 100 # Maximum length of a rollout, if set to None, the entire rollout will be used

def return_env_vars(obs_buf: torch.Tensor, potentials: torch.Tensor=None, prev_potential: torch.Tensor=None, action: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
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
        # prev_potentials = potentials[:-1]
        dt = 0.0166  # Assuming a fixed timestep of 0.02 seconds
        return {
            "root_states": root_states.unsqueeze(0),  # Add batch dimension
            "targets": targets.unsqueeze(0),  # Add batch dimension
            "potentials": potentials.unsqueeze(0),  # Add batch dimension
            "prev_potentials": prev_potential.unsqueeze(0),  # Add batch dimension
            "actions": actions.unsqueeze(0),  # Add batch dimension
            "dt": dt
        }

def get_reward_input_keys(model):
    method = model.compute_reward
    sig = inspect.signature(method)
    return list(sig.parameters.keys())[0:]  # exclude 'self'
    

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


def get_rollout_observations(rollout_path, task, required_keys, max_length=None):
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
            # Lines 0-potentials_index are the root states
            root_states = [eval(data[i].strip())[0] for i in range(0, potentials_index)]
            # Lines potentials_index+1 to prev_potentials_index are the potentials
            potentials = [eval(data[i].strip())[0] for i in range(potentials_index + 1, prev_potentials_index)]
            # Lines prev_potentials_index+1 to actions_index are the previous potentials
            prev_potentials = [eval(data[i].strip())[0] for i in range(prev_potentials_index + 1, actions_index)]
            # Lines actions_index+1 to end are the actions
            actions = [eval(data[i].strip())[0] for i in range(actions_index + 1, len(data))]
            
        input_dicts = []
        # print(len(root_states), len(potentials), len(prev_potentials), len(actions))
        usable_length = min(MAX_ROLLOUT_LENGTH, len(root_states), len(potentials), len(prev_potentials), len(actions))
        # print(f"Usable length: {usable_length}")
        for i in range(usable_length): # Formerly len(root_states)
            root_state = torch.tensor(root_states[i], dtype=torch.float32, requires_grad=True)
            potential = torch.tensor(potentials[i], dtype=torch.float32, requires_grad=True)
            prev_potential = torch.tensor(prev_potentials[i], dtype=torch.float32, requires_grad=True)
            action = torch.tensor(actions[i], dtype=torch.float32, requires_grad=True)
            # prev_potential = torch.tensor(potentials[i - 1], dtype=torch.float32, requires_grad=True) if i > 0 else torch.zeros_like(potential)
            # Pad to make same length
            # prev_potential = torch.cat([prev_potential, torch.zeros_like(potential[len(prev_potential):])], dim=0)
            # Create a dictionary with the required keys
            full_vars = return_env_vars(root_state, potential, prev_potential, action)
            filtered_vars = {k: full_vars[k] for k in required_keys}
            input_dicts.append(filtered_vars)
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



def train_reward_model(task: str, code_str: str, param_defaults: dict, data_folder: str, epochs=20, lr=5e-2, logger=None):
    try:
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        code_str = code_str.replace("-> Tuple[torch.Tensor, Dict[str, torch.Tensor]]","")
        code_str = code_str.replace("compute_reward(", "compute_reward(self,")
        model = create_model_from_code(code_str, param_defaults)
        filenames, comparisons = get_preference_pairs(data_folder, task)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # raise ValueError("This is a test error to check the error handling in the training function.")

        # Set torch randperm seed for reproducibility
        torch.manual_seed(0)
        # Shuffle the comparisons
        comparisons = comparisons[torch.randperm(comparisons.size(0))]
        # Split off 20% of the comparisons for validation
        validation_comparisons = comparisons[:int(len(comparisons) * 0.2)]
        comparisons = comparisons[int(len(comparisons) * 0.2):]

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
            best_model_state = None
            epochs_without_improvement = 0

        if BATCH_SIZE is not None:
            # Split off a validation set to use for all epochs with BATCH_SIZE
            if len(validation_comparisons) < BATCH_SIZE:
                print("Not enough validation comparisons for batch size, using all comparisons.")
                batch_validation_comparisons = validation_comparisons
            else:
                indices = torch.randperm(len(validation_comparisons))[:BATCH_SIZE]
                batch_validation_comparisons = validation_comparisons[indices]

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
            
            if i == 0 and AUTOMATIC_TERMINATION:
                original_validation_loss = loss.item()
                original_validation_accuracy = accuracy
                # print(f"Original Validation Loss: {original_validation_loss:.4f}, Original Validation Accuracy: {original_validation_accuracy:.4f}")
            
            # If MAXIMIZE_LOSS is True, we need to negate the loss
            if MAXIMIZE_LOSS:
                loss = -loss
            # Calculate the validation loss
            with torch.no_grad():
                val_loss, val_accuracy = bradley_terry_loss(model, batch_validation_comparisons, task, filenames, data_folder)
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

#     reward_code = '''
# def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#     # Scalar weights and parameters (these will become trainable)
#     speed_weight = self.speed_weight   # Increase weight for speed as it's most important
#     direction_weight = self.direction_weight # Weight for direction
#     speed_temp = self.speed_temp  # Temperature parameter for speed sensitivity
#     direction_temp = self.direction_temp  # Temperature parameter for direction sensitivity
#     distance_threshold = self.distance_threshold  # Success threshold for progressing forward distance

#     # Get the velocity of the ant
#     velocity = root_states[:, 7:10]  
#     ant_forward_velocity = velocity[:, 1] 

#     # Computation of speed reward 
#     speed_reward = torch.exp(-speed_temp * (1.0 - ant_forward_velocity))

#     # Computation of direction reward (reward forward progress)
#     forward_progress = potentials - prev_potentials
#     direction_reward = (forward_progress > distance_threshold).float()

#     # Increase the weights of forward direction
#     direction_reward *= direction_weight

#     # Combine the rewards components with corresponding weights
#     total_reward = speed_weight * speed_reward + direction_weight * direction_reward

#     # Return total reward and individual reward components in a dictionary
#     rewards_dict = {'speed_reward': speed_reward, 'direction_reward': direction_reward}
#     return total_reward, rewards_dict
# '''


    param_defaults = {
        "forward_reward_temperature": 5.0, # Started as 0.1, Passed as 10.0
        "forward_velocity_temperature": 10.0, # Started as 1.0, Passed as 10.0
        "target_height": -0.4, # Started as 0.4, Passed as 0.4
        "height_penalty_temperature": -0.1, # Started as 0.1, Passed as 0.1
    }
    
    # param_defaults = {
    #     "speed_weight": 2.0, 
    #     "direction_weight": 1.0, 
    #     "speed_temp": 0.05, 
    #     "direction_temp": 0.1, 
    #     "distance_threshold": 0.1
    # }

    model = train_reward_model(
        task="Ant",
        code_str=reward_code,
        param_defaults=param_defaults,
        data_folder="./preference_data_ant",
        # data_folder="./auto_preference_data",
        epochs=45,
        lr=0.1
    )
    print("Done")
