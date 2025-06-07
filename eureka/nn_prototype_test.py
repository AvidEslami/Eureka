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
import matplotlib.pyplot as plt
import numpy as np
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



def train_reward_model(data_folder: str, epochs=20, lr=5e-2, checkpoint_dir="./checkpoints", save_every=10, resume_from=None):
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
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
            return rewards
            
    model = MLPReward(obs_dim=211)
    filenames, comparisons = get_preference_pairs(data_folder)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Variables for tracking best model
    best_val_loss = float('inf')
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming training from {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best validation loss: {best_val_loss:.4f}")

    # Set torch randperm seed for reproducibility
    torch.manual_seed(0)
    # Shuffle the comparisons
    comparisons = comparisons[torch.randperm(comparisons.size(0))]
    # Split off 20% of the comparisons for validation
    validation_comparisons = comparisons[:int(len(comparisons) * 0.2)]
    comparisons = comparisons[int(len(comparisons) * 0.2):]
    epochs = start_epoch + epochs
    print("start_epoch", start_epoch, "epochs", epochs)
    for i in range(start_epoch, epochs):
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
            
        print(f"Epoch {i+1}/{epochs}, Train Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = {
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': loss.item(),
                'val_loss': val_loss.item(),
                'best_val_loss': best_val_loss
            }
            torch.save(best_checkpoint, os.path.join(checkpoint_dir, 'best_model.pt'))
            print(f"New best model saved with validation loss: {val_loss.item():.4f}")
        
        # Save checkpoint at regular intervals
        if (i + 1) % save_every == 0:
            checkpoint = {
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': loss.item(),
                'val_loss': val_loss.item(),
                'best_val_loss': best_val_loss
            }
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{i+1}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_checkpoint = {
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': loss.item(),
        'val_loss': val_loss.item(),
        'best_val_loss': best_val_loss
    }
    torch.save(final_checkpoint, os.path.join(checkpoint_dir, 'final_model.pt'))
    print(f"Final model saved: {os.path.join(checkpoint_dir, 'final_model.pt')}")
    
    # Also save just the model state dict for easy loading
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model_weights.pt'))
    print(f"Model weights saved: {os.path.join(checkpoint_dir, 'model_weights.pt')}")

    return model

def load_model_from_checkpoint(checkpoint_path: str, obs_dim: int = 211):
    """Load a model from a checkpoint file"""
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
            return rewards
    
    model = MLPReward(obs_dim)
    
    if checkpoint_path.endswith('model_weights.pt'):
        # Load just the state dict
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        # Load full checkpoint
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']} with validation loss: {checkpoint['val_loss']:.4f}")
    
    return model

def inference_example(sample_file: str):
    
    # Load the trained model from checkpoint
    model = load_model_from_checkpoint("./checkpoints/best_model.pt")
    model.eval()  # Set to evaluation mode
    
    with open(sample_file, 'r') as f:
        score = float(f.readline())  # Skip the score line
        observations = [eval(line.strip()) for line in f if line.strip()]  # Load observations
    
    print(f"File: {sample_file}")
    print(f"Ground truth score: {score}")
    print(f"Number of observations: {len(observations)}")
    
    # Convert observations to tensors and compute rewards
    total_reward = 0.0
    individual_rewards = []
    cumulative_rewards = []
    
    with torch.no_grad():  # No gradient computation for inference
        for i, obs in enumerate(observations):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            reward = model(obs_tensor)
            individual_rewards.append(reward.item())
            total_reward += reward.item()
            cumulative_rewards.append(total_reward)
            print(f"Step {i}: Reward = {reward.item():.4f}, Cumulative = {total_reward:.4f}")
    
    print(f"Total reward: {total_reward:.4f}")
    print(f"Average reward per step: {total_reward/len(observations):.4f}")
    
    # Create plots
    time_steps = np.arange(len(observations))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot individual rewards over time
    ax1.plot(time_steps, individual_rewards, 'b-', linewidth=1.5, alpha=0.7, label='Individual Rewards')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Individual Reward')
    ax1.set_title('Individual Rewards Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot cumulative rewards over time
    ax2.plot(time_steps, cumulative_rewards, 'r-', linewidth=2, label='Cumulative Reward')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title('Cumulative Rewards Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add text box with summary statistics
    textstr = f'Ground Truth Score: {score}\nTotal Reward: {total_reward:.4f}\nAvg Reward/Step: {total_reward/len(observations):.4f}\nSteps: {len(observations)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()
    
    return total_reward, individual_rewards, cumulative_rewards

def compute_reward_with_params(object_rot, goal_rot, object_angvel, object_pos, fingertip_pos, 
                              rotation_reward_temp, angvel_threshold, angvel_penalty_temp, min_distance_temp):
    """Compute reward with specific parameter values"""
    rot_diff = torch.abs(torch.sum(object_rot * goal_rot, dim=1) - 1) / 2
    rotation_reward = torch.exp(-rotation_reward_temp * rot_diff)

    # Angular velocity penalty
    angvel_norm = torch.norm(object_angvel, dim=1)
    angular_velocity_penalty = torch.where(angvel_norm > angvel_threshold, 
                                         torch.exp(-angvel_penalty_temp * (angvel_norm - angvel_threshold)), 
                                         torch.zeros_like(angvel_norm))
    
    # Distance reward
    min_distance = torch.min(torch.norm(fingertip_pos - object_pos[:, None], dim=2), dim=1).values
    uncapped_distance_reward = torch.exp(-min_distance_temp * min_distance) 
    distance_reward = torch.clamp(uncapped_distance_reward, 0.0, 1.0)

    total_reward = rotation_reward - angular_velocity_penalty + distance_reward
    
    reward_components = {
        "rotation_reward": rotation_reward,
        "angular_velocity_penalty": angular_velocity_penalty, 
        "distance_reward": distance_reward
    }
    
    return total_reward, reward_components

def inference_example_analytical(sample_file: str):
    """
    Compare two parameter sets for the analytical reward function plus neural network
    """
    # Load observation data from a preference file
    
    with open(sample_file, 'r') as f:
        score = float(f.readline())  # Skip the score line
        observations = [eval(line.strip()) for line in f if line.strip()]  # Load observations
    
    print(f"File: {sample_file}")
    print(f"Ground truth score: {score}")
    print(f"Number of observations: {len(observations)}")
    
    # Define three parameter sets
    params_set1 = {
        'rotation_reward_temp': 20.0,
        'angvel_threshold': 2.0,
        'angvel_penalty_temp': 2.0,
        'min_distance_temp': 10.0
    }

    params_set2 = {
        'rotation_reward_temp': 38.814964294433594,
        'angvel_threshold': 3.906949758529663,
        'angvel_penalty_temp': 4.602588653564453,
        'min_distance_temp': -5.336449146270752
    }
    
    params_set3 = {
        'rotation_reward_temp': 30.20393180847168,
        'angvel_threshold': -3.286231756210327,
        'angvel_penalty_temp': 8.288484573364258,
        'min_distance_temp': -1.0389528274536133
    }
    
    params_set4 = {
        'rotation_reward_temp': 13.19585132598877,
        'angvel_threshold': 8.594115257263184,
        'angvel_penalty_temp': 10.15329647064209,
        'min_distance_temp': 3.814347267150879
    }
    
    # Load neural network model
    model = load_model_from_checkpoint("./checkpoints/checkpoint_epoch_200.pt")
    model.eval()
    
    # Calculate rewards for all three methods
    results = {}
    
    # Analytical reward functions
    for params_name, params in [("Original Parameters", params_set1), ("2 Clip Final", params_set2), ("Flipped", params_set3), ("Normalized", params_set4)]:
        total_reward = 0.0
        individual_rewards = []
        cumulative_rewards = []
        rotation_rewards = []
        angular_velocity_penalties = []
        distance_rewards = []
        
        with torch.no_grad():
            for i, obs in enumerate(observations):
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                env_vars = return_env_vars(obs_tensor)
                
                reward, reward_components = compute_reward_with_params(
                    object_rot=env_vars["object_rot"],
                    goal_rot=env_vars["goal_rot"],
                    object_angvel=env_vars["object_angvel"],
                    object_pos=env_vars["object_pos"],
                    fingertip_pos=env_vars["fingertip_pos"],
                    **params
                )
                
                reward_value = reward.item()
                individual_rewards.append(reward_value)
                total_reward += reward_value
                cumulative_rewards.append(total_reward)
                
                # Store individual components
                rotation_rewards.append(reward_components["rotation_reward"].item())
                angular_velocity_penalties.append(reward_components["angular_velocity_penalty"].item())
                distance_rewards.append(reward_components["distance_reward"].item())
        
        results[params_name] = {
            'total_reward': total_reward,
            'individual_rewards': individual_rewards,
            'cumulative_rewards': cumulative_rewards,
            'avg_reward': total_reward / len(observations),
            'rotation_rewards': rotation_rewards,
            'angular_velocity_penalties': angular_velocity_penalties,
            'distance_rewards': distance_rewards
        }
        
        print(f"\n{params_name}:")
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Average reward per step: {total_reward/len(observations):.4f}")
    
    # Neural network rewards
    total_reward_nn = 0.0
    individual_rewards_nn = []
    cumulative_rewards_nn = []
    
    with torch.no_grad():
        for i, obs in enumerate(observations):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            reward = model(obs_tensor)
            reward_value = reward.item()
            individual_rewards_nn.append(reward_value)
            total_reward_nn += reward_value
            cumulative_rewards_nn.append(total_reward_nn)
    
    results["Neural Network"] = {
        'total_reward': total_reward_nn,
        'individual_rewards': individual_rewards_nn,
        'cumulative_rewards': cumulative_rewards_nn,
        'avg_reward': total_reward_nn / len(observations)
    }
    
    print(f"\nNeural Network:")
    print(f"  Total reward: {total_reward_nn:.4f}")
    print(f"  Average reward per step: {total_reward_nn/len(observations):.4f}")
    
    # Create comparison plots
    time_steps = np.arange(len(observations))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot individual rewards comparison
    axes[0, 0].plot(time_steps, results["Original Parameters"]['individual_rewards'], 
             'b-', linewidth=1.5, alpha=0.8, label='Original Parameters')
    axes[0, 0].plot(time_steps, results["2 Clip Final"]['individual_rewards'], 
             'r-', linewidth=1.5, alpha=0.8, label='2 Clip Final')
    axes[0, 0].plot(time_steps, results["Flipped"]['individual_rewards'], 
             'orange', linewidth=1.5, alpha=0.8, label='Flipped')
    axes[0, 0].plot(time_steps, results["Normalized"]['individual_rewards'], 
             'purple', linewidth=1.5, alpha=0.8, label='Normalized')
    axes[0, 0].plot(time_steps, results["Neural Network"]['individual_rewards'], 
             'g-', linewidth=1.5, alpha=0.8, label='Neural Network')
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Individual Reward')
    axes[0, 0].set_title('Total Rewards per Step')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    
    # Plot cumulative rewards comparison
    axes[0, 1].plot(time_steps, results["Original Parameters"]['cumulative_rewards'], 
             'b-', linewidth=2, label='Original Parameters')
    axes[0, 1].plot(time_steps, results["2 Clip Final"]['cumulative_rewards'], 
             'r-', linewidth=2, label='2 Clip Final')
    axes[0, 1].plot(time_steps, results["Flipped"]['cumulative_rewards'], 
             'orange', linewidth=2, label='Flipped')
    axes[0, 1].plot(time_steps, results["Normalized"]['cumulative_rewards'], 
             'purple', linewidth=2, label='Normalized')
    axes[0, 1].plot(time_steps, results["Neural Network"]['cumulative_rewards'], 
             'g-', linewidth=2, label='Neural Network')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Cumulative Reward')
    axes[0, 1].set_title('Cumulative Rewards')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)
    
    # Plot rotation reward components
    for params_name, color in [("Original Parameters", 'b'), ("2 Clip Final", 'r'), 
                              ("Flipped", 'orange'), ("Normalized", 'purple')]:
        axes[1, 0].plot(time_steps, results[params_name]['rotation_rewards'], 
                 color, linewidth=1.5, alpha=0.8, label=params_name)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Rotation Reward')
    axes[1, 0].set_title('Rotation Reward Components')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)
    
    # Plot angular velocity penalty components
    for params_name, color in [("Original Parameters", 'b'), ("2 Clip Final", 'r'), 
                              ("Flipped", 'orange'), ("Normalized", 'purple')]:
        axes[1, 1].plot(time_steps, [-x for x in results[params_name]['angular_velocity_penalties']], 
                 color, linewidth=1.5, alpha=0.8, label=params_name)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Angular Velocity Penalty')
    axes[1, 1].set_title('Angular Velocity Penalties')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)
    
    # Plot distance reward components
    for params_name, color in [("Original Parameters", 'b'), ("2 Clip Final", 'r'), 
                              ("Flipped", 'orange'), ("Normalized", 'purple')]:
        axes[1, 2].plot(time_steps, results[params_name]['distance_rewards'], 
                 color, linewidth=1.5, alpha=0.8, label=params_name)
    axes[1, 2].set_xlabel('Time Step')
    axes[1, 2].set_ylabel('Distance Reward')
    axes[1, 2].set_title('Distance Reward Components')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend(fontsize=8)
    
    # Add comparison statistics in top-right corner
    orig_total = results["Original Parameters"]['total_reward']
    opt_total = results["2 Clip Final"]['total_reward']
    latest_total = results["Flipped"]['total_reward']
    newest_total = results["Normalized"]['total_reward']
    nn_total = results["Neural Network"]['total_reward']
    
    improvement_opt = ((opt_total - orig_total) / abs(orig_total)) * 100 if orig_total != 0 else 0
    improvement_latest = ((latest_total - orig_total) / abs(orig_total)) * 100 if orig_total != 0 else 0
    improvement_newest = ((newest_total - orig_total) / abs(orig_total)) * 100 if orig_total != 0 else 0
    improvement_nn = ((nn_total - orig_total) / abs(orig_total)) * 100 if orig_total != 0 else 0
    
    textstr = (f'Ground Truth: {score}\n'
               f'Original: {orig_total:.3f}\n'
               f'2 Clip Final: {opt_total:.3f} ({improvement_opt:+.1f}%)\n'
               f'Flipped: {latest_total:.3f} ({improvement_latest:+.1f}%)\n'
               f'Normalized: {newest_total:.3f} ({improvement_newest:+.1f}%)\n'
               f'Neural Net: {nn_total:.3f} ({improvement_nn:+.1f}%)\n'
               f'Steps: {len(observations)}')
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    axes[0, 2].text(0.02, 0.98, textstr, transform=axes[0, 2].transAxes, fontsize=7,
             verticalalignment='top', bbox=props)
    axes[0, 2].axis('off')  # Hide the axes for the stats panel
    
    plt.tight_layout()
    plt.show()
    
    return results

def compare_rollouts(file1_path: str, file2_path: str, checkpoint_path: str = "./checkpoints/best_model.pt"):
    """Compare two rollouts using the trained model"""
    
    # Load the trained model
    model = load_model_from_checkpoint(checkpoint_path)
    model.eval()
    
    rollout_rewards = {}
    
    for file_path in [file1_path, file2_path]:
        with open(file_path, 'r') as f:
            score = float(f.readline())
            observations = [eval(line.strip()) for line in f if line.strip()]
        
        total_reward = 0.0
        with torch.no_grad():
            for obs in observations:
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                reward = model(obs_tensor)
                total_reward += reward.item()
        
        rollout_rewards[file_path] = {
            'total_reward': total_reward,
            'ground_truth_score': score,
            'num_steps': len(observations)
        }
    
    print("Rollout Comparison:")
    for file_path, info in rollout_rewards.items():
        print(f"{file_path}:")
        print(f"  Model total reward: {info['total_reward']:.4f}")
        print(f"  Ground truth score: {info['ground_truth_score']}")
        print(f"  Number of steps: {info['num_steps']}")
    
    # Determine which rollout the model prefers
    file1_reward = rollout_rewards[file1_path]['total_reward']
    file2_reward = rollout_rewards[file2_path]['total_reward']
    
    if file1_reward > file2_reward:
        print(f"\nModel prefers: {file1_path}")
    elif file2_reward > file1_reward:
        print(f"\nModel prefers: {file2_path}")
    else:
        print("\nModel is indifferent between the two rollouts")
    
    return rollout_rewards

def evaluate_on_test_dataset(test_data_folder: str, checkpoint_path: str = "./checkpoints/best_model.pt", verbose: bool = True):
    """
    Evaluate the trained model on a test dataset
    
    Args:
        test_data_folder: Path to the test dataset folder
        checkpoint_path: Path to the model checkpoint
        verbose: Whether to print detailed results
    
    Returns:
        Dictionary containing evaluation metrics
    """
    
    # Load the trained model
    model = load_model_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Get test preference pairs
    test_filenames, test_comparisons = get_preference_pairs(test_data_folder)
    
    if len(test_comparisons) == 0:
        print("No valid preference pairs found in test dataset!")
        return None
    
    print(f"Test dataset: {len(test_filenames)} files, {len(test_comparisons)} preference pairs")
    
    # Evaluate using Bradley-Terry loss
    with torch.no_grad():
        test_loss = bradley_terry_loss(model, test_comparisons, test_filenames, test_data_folder, verbose_accururacy=verbose)
        
        # Calculate detailed metrics
        rollout_rewards = {}
        rollout_data_full = {}
        
        # Load all rollout data
        for i, filename in enumerate(test_filenames):
            with open(os.path.join(test_data_folder, filename), 'r') as f:
                f.readline()  # Skip score line
                rollout_data_full[i] = [line for line in f]
        
        # Calculate rewards for each comparison
        correct_predictions = 0
        total_predictions = len(test_comparisons)
        prediction_details = []
        
        for idx in range(len(test_comparisons)):
            i, j = int(test_comparisons[idx, 0]), int(test_comparisons[idx, 1])
            preference = int(test_comparisons[idx, 2])
            
            min_length = min(len(rollout_data_full[i]), len(rollout_data_full[j]))
            
            # Calculate rewards for both rollouts
            for k in [i, j]:
                key = (k, min_length)
                if key not in rollout_rewards:
                    inputs = rollout_data_full[k][:min_length]
                    total_reward = torch.tensor(0.0)
                    for inp in inputs:
                        inp = eval(inp)
                        inp = torch.tensor(inp, dtype=torch.float32)
                        reward = model(inp)
                        total_reward = total_reward + reward
                    rollout_rewards[key] = total_reward
            
            reward_i = rollout_rewards[(i, min_length)]
            reward_j = rollout_rewards[(j, min_length)]
            
            # Check if prediction is correct
            if preference == 0:  # i is preferred
                predicted_correctly = reward_i > reward_j
            else:  # j is preferred
                predicted_correctly = reward_j > reward_i
            
            if predicted_correctly:
                correct_predictions += 1
            
            prediction_details.append({
                'file_i': test_filenames[i],
                'file_j': test_filenames[j],
                'reward_i': reward_i.item(),
                'reward_j': reward_j.item(),
                'true_preference': 'i' if preference == 0 else 'j',
                'predicted_preference': 'i' if reward_i > reward_j else 'j',
                'correct': predicted_correctly
            })
    
    # Calculate metrics
    accuracy = correct_predictions / total_predictions
    
    # Group predictions by correctness
    correct_predictions_list = [p for p in prediction_details if p['correct']]
    incorrect_predictions_list = [p for p in prediction_details if not p['correct']]
    
    results = {
        'test_loss': test_loss.item(),
        'accuracy': accuracy,
        'correct_predictions': correct_predictions,
        'total_predictions': total_predictions,
        'prediction_details': prediction_details
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"TEST DATASET EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Test Loss: {test_loss.item():.4f}")
        print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
        print(f"Correct Predictions: {len(correct_predictions_list)}")
        print(f"Incorrect Predictions: {len(incorrect_predictions_list)}")
        
        if len(incorrect_predictions_list) > 0:
            print(f"\nIncorrect Predictions:")
            for pred in incorrect_predictions_list[:5]:  # Show first 5 incorrect predictions
                print(f"  {pred['file_i']} (reward: {pred['reward_i']:.4f}) vs {pred['file_j']} (reward: {pred['reward_j']:.4f})")
                print(f"    True preference: {pred['true_preference']}, Predicted: {pred['predicted_preference']}")
        
        if len(correct_predictions_list) > 0:
            print(f"\nSample Correct Predictions:")
            for pred in correct_predictions_list[:3]:  # Show first 3 correct predictions
                print(f"  {pred['file_i']} (reward: {pred['reward_i']:.4f}) vs {pred['file_j']} (reward: {pred['reward_j']:.4f})")
                print(f"    True preference: {pred['true_preference']}, Predicted: {pred['predicted_preference']}")
    
    return results

def compute_reward(object_rot: torch. Tensor, goal_rot: torch. Tensor, object_angvel: torch. Tensor, object_pos: torch. Tensor, fingertip_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str,torch.Tensor]]:
    rotation_reward_temp = 20.0
    angvel_threshold = 2.0
    angvel_penalty_temp = 2.0
    min_distance_temp = 10.0

    angvel_penalty_temp = 2.6188
    angvel_threshold = 3.7946
    min_distance_temp = 1.2229
    rotation_reward_temp = 28.1642
    
    rotation_reward_temp = 38.814964294433594
    angvel_threshold = 3.906949758529663
    angvel_penalty_temp = 4.602588653564453
    min_distance_temp = -5.336449146270752

    rotation_reward_temp = 33.89820861816406
    angvel_threshold = 3.8580095767974854
    angvel_penalty_temp = 3.8610970973968506
    min_distance_temp = -3.908681631088257
    rot_diff = torch.abs(torch.sum(object_rot * goal_rot, dim=1) - 1) / 2
    rotation_reward = torch.exp(-rotation_reward_temp * rot_diff)

    # Angular velocity penalty
    angvel_norm = torch.norm(object_angvel, dim=1)
    angular_velocity_penalty = torch.where(angvel_norm > angvel_threshold, torch.exp(-angvel_penalty_temp * (angvel_norm - angvel_threshold)), torch.zeros_like(angvel_norm))
    
    # Distance reward
    min_distance = torch.min(torch.norm(fingertip_pos - object_pos[:, None], dim=2), dim=1).values
    uncapped_distance_reward = torch.exp(-min_distance_temp * min_distance) 
    distance_reward = torch.clamp(uncapped_distance_reward, 0.0, 1.0)

    total_reward = rotation_reward - angular_velocity_penalty + distance_reward

    reward_components = {
        "rotation_reward": rotation_reward,
        "angular_velocity_penalty": angular_velocity_penalty, 
        "distance_reward": distance_reward
    }
    return total_reward, reward_components
# Example when running script directly
if __name__ == "__main__":
    # model = train_reward_model(
    #     data_folder="./preference_data",
    #     epochs=200,
    #     lr=0.0001,
    #     checkpoint_dir="./checkpoints_full_length",
    #     save_every=10,
    #     resume_from="./checkpoints_full_length/checkpoint_epoch_20.pt"
    # )
    # print("Training Done")
    # exit()
    
    # Test analytical reward function vs neural network
    print("=== Comparing Analytical vs Neural Network Rewards ===")
    inference_example_analytical("./preference_data/1_ShadowHand_2025-06-03_03-42-06.txt")
    
    # Single rollout inference with MLP
    # print("\n=== Testing MLP Reward Model ===")
    # inference_example("./preference_data/1_ShadowHand_2025-06-03_03-31-46.txt")
    
    # Compare two rollouts
    # compare_rollouts(
    #     "./preference_data/1_ShadowHand_2025-06-03_03-31-46.txt",
    #     "./preference_data/1_ShadowHand_2025-06-03_03-39-46.txt" 
    # )
    # test_results = evaluate_on_test_dataset("./preference_data", "./checkpoints_full_length/checkpoint_epoch_20.pt")

    