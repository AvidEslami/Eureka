# train_dynamic_reward.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import inspect
from typing import Dict, Tuple

torch.autograd.set_detect_anomaly(True)


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
    preference_pairs = []
    for i in range(len(filenames)):
        for j in range(len(filenames)):
            if i != j:
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


def get_rollout_observations(rollout_path, required_keys):
    with open(rollout_path, 'r') as f:
        f.readline()  # Skip score line
        data = [eval(line) for line in f]
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
    input_keys = get_reward_input_keys(model)
    rollout_data = {
        i: get_rollout_observations(os.path.join(data_folder, path), input_keys)
        for i, path in enumerate(filenames)
    }

    rollout_rewards = {}
    for i, inputs in rollout_data.items():
        total_reward = torch.tensor(0.0, requires_grad=True)
        for inp in inputs:
            reward, _ = model(**inp)
            total_reward = total_reward + reward
        rollout_rewards[i] = total_reward

    left = torch.stack([rollout_rewards[int(i)].squeeze() for i in comparisons[:, 0]])
    right = torch.stack([rollout_rewards[int(i)].squeeze() for i in comparisons[:, 1]])
    logits = torch.stack([left, right], dim=1)
    targets = comparisons[:, -1].long()

    with torch.no_grad():
        acc = (torch.argmax(logits, dim=1) == targets).float().mean()
        print(f"Pairwise accuracy: {acc.item():.2f}")

    if verbose_accururacy:
        for i in range(len(comparisons)):
            left_idx = int(comparisons[i, 0])
            right_idx = int(comparisons[i, 1])
            # Also print the success score of the comparison (first line inside the file)
            with open(os.path.join(data_folder, filenames[left_idx]), 'r') as f1:
                left_success = float(f1.readline())
            with open(os.path.join(data_folder, filenames[right_idx]), 'r') as f2:
                right_success = float(f2.readline())
            if targets[i] == 0:
                print(f"Left: {filenames[left_idx]}, Right: {filenames[right_idx]} - Correct ({left_success} vs {right_success})")
            else:
                print(f"Left: {filenames[left_idx]}, Right: {filenames[right_idx]} - Incorrect ({left_success} vs {right_success})")

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



def train_reward_model(code_str: str, param_defaults: dict, data_folder: str, epochs=20, lr=5e-2):
    code_str = code_str.replace("-> Tuple[torch.Tensor, Dict[str, torch.Tensor]]","")
    code_str = code_str.replace("compute_reward(", "compute_reward(self,")
    model = create_model_from_code(code_str, param_defaults)
    filenames, comparisons = get_preference_pairs(data_folder)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # input_keys = get_reward_input_keys(model)
    # rollout_data = {
    #     i: get_rollout_observations(os.path.join(data_folder, path), input_keys)
    #     for i, path in enumerate(filenames)
    # }
    

    print(f"Initial Loss: {bradley_terry_loss(model, comparisons, filenames, data_folder, verbose_accururacy=True)}")
    for i in range(epochs):
        optimizer.zero_grad()
        loss = bradley_terry_loss(model, comparisons, filenames, data_folder)
        loss.backward()
        optimizer.step()
        print(f"Epoch {i+1}/{epochs}, Loss: {loss.item():.4f}")

    print("Learned parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.item()}")

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

    model = train_reward_model(
        code_str=reward_code,
        param_defaults=param_defaults,
        data_folder="./preference_data",
        epochs=50,
        lr=0.5
    )
    print("Done")
