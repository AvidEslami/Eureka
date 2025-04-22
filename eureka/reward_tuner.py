# train_dynamic_reward.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
from typing import Dict, Tuple

torch.autograd.set_detect_anomaly(True)


def return_env_vars(obs_buf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    object_rot = obs_buf[75:79]
    object_angvel = obs_buf[82:85] / 0.2 # Velocities are scaled by 0.2 -> this is a hardcoded environment constant
    goal_rot = obs_buf[88:92]
    return object_rot, goal_rot


def get_preference_pairs(data_folder: str):
    filenames = [f for f in os.listdir(data_folder) if f.endswith(".txt")]
    preference_pairs = []
    for i in range(len(filenames)):
        for j in range(len(filenames)):
            if i != j:
                with open(os.path.join(data_folder, filenames[i]), 'r') as f1:
                    score_i = float(f1.readline())
                with open(os.path.join(data_folder, filenames[j]), 'r') as f2:
                    score_j = float(f2.readline())
                preference_pairs.append((i, j, 0 if score_i > score_j else 1))
    return filenames, torch.tensor(preference_pairs, dtype=torch.float32)


def get_rollout_observations(rollout_path):
    with open(rollout_path, 'r') as f:
        f.readline()  # Skip score line
        data = [eval(line) for line in f]
    data = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    object_rot_list, goal_rot_list = [], []
    for i in range(data.shape[0]):
        object_rot, goal_rot = return_env_vars(data[i])
        object_rot_list.append(object_rot)
        goal_rot_list.append(goal_rot)
    return object_rot_list, goal_rot_list


def bradley_terry_loss(model, comparisons, filenames, data_folder):
    loss_fn = nn.CrossEntropyLoss()
    rollout_data = {
        i: get_rollout_observations(os.path.join(data_folder, path))
        for i, path in enumerate(filenames)
    }

    rollout_rewards = {}
    for i, (object_rots, goal_rots) in rollout_data.items():
        total_reward = torch.tensor(0.0, requires_grad=True)
        for obj_rot, goal_rot in zip(object_rots, goal_rots):
            reward, _ = model(obj_rot, goal_rot)
            total_reward = total_reward + reward
        rollout_rewards[i] = total_reward

    left = torch.stack([rollout_rewards[int(i)].squeeze() for i in comparisons[:, 0]])
    right = torch.stack([rollout_rewards[int(i)].squeeze() for i in comparisons[:, 1]])
    logits = torch.stack([left, right], dim=1)
    targets = comparisons[:, -1].long()

    with torch.no_grad():
        acc = (torch.argmax(logits, dim=1) == targets).float().mean()
        print(f"Pairwise accuracy: {acc.item():.2f}")

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
    method_body = method_lines[method_header_index + 1:]
    indented_method = [f"    {line}" if line.strip() else "" for line in [method_def] + method_body]

    param_init = "\n".join(
        f"        self.{k} = nn.Parameter(torch.tensor({v}, dtype=torch.float32))"
        for k, v in param_names.items()
    )

    class_code = f"""
import torch
import torch.nn as nn
{chr(10).join(import_lines)}

class {module_name}(nn.Module):
    def __init__(self):
        super().__init__()
{param_init}

{chr(10).join(indented_method)}

    def forward(self, object_rot, goal_rot):
        return self.compute_reward(object_rot, goal_rot)
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

    print(f"Initial Loss: {bradley_terry_loss(model, comparisons, filenames, data_folder)}")
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

    param_defaults = {
        "dist_penalty_scaler": -0.9,
        "reward_temp": 1.0,
        "garbage_term_scaler": 0.001,
        "survival_scaler": 0.1,
    }

    model = train_reward_model(
        code_str=reward_code,
        param_defaults=param_defaults,
        data_folder="./preference_data",
        epochs=1,
        lr=0.05
    )
    print("Done")
