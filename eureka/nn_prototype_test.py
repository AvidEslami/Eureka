# import torch
# import torch.nn as nn
# import torch.optim as optim

# # Preference data
# data_points = [
#     "ShadowHand_2025-03-07_01-39-39.txt",
#     "ShadowHand_2025-03-07_01-40-05.txt",
#     "ShadowHand_2025-03-07_01-40-51.txt",
#     "ShadowHand_2025-03-07_01-41-18.txt",
#     "ShadowHand_2025-03-07_01-41-44.txt",
# ]

# # Construct preference pairs based on prepended scores
# preference_pairs = []
# for i in range(len(data_points)):
#     for j in range(len(data_points)):
#         if i != j:
#             with open(f"./preference_data/{data_points[i]}", 'r') as f1, \
#                  open(f"./preference_data/{data_points[j]}", 'r') as f2:
#                 score_i = float(f1.readline())
#                 score_j = float(f2.readline())
#             preference_pairs.append((i, j, 0 if score_i > score_j else 1))

# comparisons = torch.tensor(preference_pairs, dtype=torch.float32)

# # Load rollout trajectories into running memory
# def load_rollouts(paths):
#     rollouts = {}
#     for i, path in enumerate(paths):
#         with open(f"./preference_data/{path}", 'r') as f:
#             f.readline()  # Skip score
#             obs = [eval(line) for line in f]
#             obs_tensor = torch.tensor(obs, dtype=torch.float32)
#             rollouts[i] = obs_tensor
#     return rollouts

# rollout_data = load_rollouts(data_points)

# # Simple MLP reward function (made simpler to prevent instant preference convergence)
# class MLPReward(nn.Module):
#     def __init__(self, obs_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(obs_dim, 8),
#             nn.ReLU(),
#             nn.Linear(8, 4),
#             nn.ReLU(),
#             nn.Linear(4, 1)
#         )

#     def forward(self, obs_seq):
#         rewards = self.net(obs_seq)     
#         return rewards.sum() 

# # Bradley-Terry loss
# def bradley_terry_loss(model, comparisons, rollout_data):
#     rewards = {}
#     for i in rollout_data:
#         rollout = rollout_data[i]
#         rollout.requires_grad_(True)
#         total_reward = model(rollout)
#         rewards[i] = total_reward

#     left = torch.stack([rewards[int(i)].squeeze() for i in comparisons[:, 0]])
#     right = torch.stack([rewards[int(i)].squeeze() for i in comparisons[:, 1]])
#     scores = torch.stack([left, right], dim=1)

#     targets = comparisons[:, -1].to(torch.long).squeeze()

#     with torch.no_grad():
#         pred = torch.argmax(scores, dim=1)
#         acc = (pred == targets).float().mean()
#         print(f"Pairwise accuracy: {acc.item():.2f}")

#     return nn.CrossEntropyLoss()(scores, targets)

# # Training loop
# def train(model, rollout_data, comparisons, epochs=20, lr=5e-3):
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         loss = bradley_terry_loss(model, comparisons, rollout_data)
#         print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")
#         loss.backward()
#         optimizer.step()

#     print("Final learned parameters:")
#     for name, param in model.named_parameters():
#         print(name, param.data.norm().item())


# obs_dim = rollout_data[0].shape[1]
# model = MLPReward(obs_dim)
# train(model, rollout_data, comparisons)

# # Save model state dict
# torch.save(model.state_dict(), "mlp_reward.pt")
# print("Model Saved as mlp_reward.pt")
# # Wrap and script the model for deployment? Might works in the torch.jit setup
# scripted_model = torch.jit.script(model)
# scripted_model.save("mlp_reward_scripted.pt")
# print("Model Jit Saved as mlp_reward_scripted.pt")

# train_dynamic_reward.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import inspect
from typing import Dict, Tuple
from collections import defaultdict
torch.autograd.set_detect_anomaly(True)

LOG_FAILURES = False
LOG_SUCCESS = False
TRACK_FAILURES = True
FAILURE_TRACK_PROGRESS = defaultdict(list)
MAXIMIZE_LOSS = False # If True, the loss will be maximized instead of minimized

def return_env_vars(obs_buf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

def get_reward_input_keys(model):
    method = model.compute_reward
    sig = inspect.signature(method)
    return list(sig.parameters.keys())[0:]  # exclude 'self'
    

def get_preference_pairs(data_folder: str):
    filenames = [f for f in os.listdir(data_folder) if f.endswith(".txt")]
    
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
    return filenames, torch.tensor(preference_pairs, dtype=torch.float32)


def get_rollout_observations(rollout_path, required_keys, max_length=None):
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


def bradley_terry_loss(model, comparisons, filenames, data_folder, verbose_accururacy=False):
    loss_fn = nn.CrossEntropyLoss()
    # input_keys = get_reward_input_keys(model)
    
    # First load all rollout data
    rollout_data_full = {}
    for i, path in enumerate(filenames):
        with open(os.path.join(data_folder, path), 'r') as f:
            f.readline()  # Skip score line
            rollout_data_full[i] = [line for line in f]
    
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
        min_length = min(len(rollout_data_full[i]), len(rollout_data_full[j]))

        for k in [i, j]:
            if k not in cached_observations:
                # Cache the full observation sequence
                # cached_observations[k] = get_rollout_observations(os.path.join(data_folder, filenames[k]), input_keys)
                # MLP takes in full observation sequence as input
                cached_observations[k] = rollout_data_full[k]
            
            key = (k, min_length)
            if key not in rollout_rewards:
                inputs = cached_observations[k][:min_length]
                total_reward = torch.tensor(0.0, requires_grad=True)
                for inp in inputs:
                    inp # inp is a stringified list of observations
                    # Convert string to tensor
                    inp = eval(inp)
                    inp = torch.tensor(inp, dtype=torch.float32, requires_grad=True)
                    reward = model(inp)
                    total_reward = total_reward + reward
                rollout_rewards[key] = total_reward

    # left = torch.stack([rollout_rewards[int(i)].squeeze() for i in comparisons[:, 0]])
    # right = torch.stack([rollout_rewards[int(i)].squeeze() for i in comparisons[:, 1]])
    # left = torch.stack([rollout_rewards[(int(i), min(rollout_data_full[int(i)], rollout_data_full[int(j)]))].squeeze() for i, j in comparisons])
    # right = torch.stack([rollout_rewards[(int(j), min(rollout_data_full[int(i)], rollout_data_full[int(j)]))].squeeze() for i, j in comparisons])
    left = torch.stack([
        rollout_rewards[(int(row[0]), min(len(rollout_data_full[int(row[0])]), len(rollout_data_full[int(row[1])])))].squeeze()
        for row in comparisons
    ])
    right = torch.stack([
        rollout_rewards[(int(row[1]), min(len(rollout_data_full[int(row[0])]), len(rollout_data_full[int(row[1])])))].squeeze()
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


    return loss_fn(logits, targets)


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



def train_reward_model(data_folder: str, epochs=20, lr=5e-2):
    # code_str = code_str.replace("-> Tuple[torch.Tensor, Dict[str, torch.Tensor]]","")
    # code_str = code_str.replace("compute_reward(", "compute_reward(self,")
    # model = create_model_from_code(code_str, param_defaults)
    class MLPReward(nn.Module):
        def __init__(self, obs_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 8),
                nn.ReLU(),
                nn.Linear(8, 4),
                nn.ReLU(),
                nn.Linear(4, 1)
            )

        def forward(self, obs_seq):
            rewards = self.net(obs_seq)     
            # return rewards.sum() 
            return rewards
    model = MLPReward(obs_dim=211) # 96 is the observation dimension for the Shadow Hand environment
    filenames, comparisons = get_preference_pairs(data_folder)
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
    for i in range(epochs):
        optimizer.zero_grad()
        loss = bradley_terry_loss(model, comparisons, filenames, data_folder, verbose_accururacy=(i % 10 == 0))
        # If MAXIMIZE_LOSS is True, we need to negate the loss
        if MAXIMIZE_LOSS:
            loss = -loss
        loss.backward()
        optimizer.step()
        # Calculate the validation loss
        with torch.no_grad():
            val_loss = bradley_terry_loss(model, validation_comparisons, filenames, data_folder)
            # print(f"Validation Loss: {val_loss.item():.4f}")
        print(f"Epoch {i+1}/{epochs}, Train Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

    #     if i % 10 == 0:
    #         print("Learned parameters:")
    #         for name, param in model.named_parameters():
    #             print(f"{name}: {param.item()}")

    # print("Learned parameters:")
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.item()}")

    return model


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

    model = train_reward_model(
        data_folder="./preference_data",
        epochs=45,
        lr=0.5
    )
    print("Done")
