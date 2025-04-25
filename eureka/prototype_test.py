from isaacgym.torch_utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.optimize import minimize

torch.autograd.set_detect_anomaly(True)

def return_env_vars(obs_buf: torch.Tensor) -> torch.Tensor:
    ### Reconstruct state variables object_rot and goal_rot from obs_buf

    # obs_buf[:, 38:42] = quat_mul(object_rot, quat_conjugate(goal_rot))

    object_rot = obs_buf[75:79]
    object_rot = torch.tensor([object_rot[0], object_rot[1], object_rot[2], object_rot[3]])
    goal_rot = obs_buf[88:92]
    goal_rot = torch.tensor([goal_rot[0], goal_rot[1], goal_rot[2], goal_rot[3]])

    return object_rot, goal_rot

# the folder ./preference_data contains 5 rollouts, we'd like to create 5*5 preference pairs
# The first line of each file is the score, use this to determine preference
# Each index should be [rollout_1,rollout_2,0 if rollout_1 is better, 1 if rollout_2 is better]
# The files have arbitrary names
data_points = [
    "ShadowHand_2025-03-07_01-39-39.txt",
    "ShadowHand_2025-03-07_01-40-05.txt",
    "ShadowHand_2025-03-07_01-40-51.txt",
    "ShadowHand_2025-03-07_01-41-18.txt",
    "ShadowHand_2025-03-07_01-41-44.txt",
]

preference_pairs = []
for i in range(len(data_points)):
    for j in range(len(data_points)):
        if i != j:
            with open(f"./preference_data/{data_points[i]}", 'r') as f:
                score_i = float(f.readline())
            with open(f"./preference_data/{data_points[j]}", 'r') as f:
                score_j = float(f.readline())
            preference_pairs.append((i, j, 0 if score_i > score_j else 1))

print("Preference pairs:")
for pair in preference_pairs:
    print(pair)

# LLM's reward function, but I added some garbage that will hopefully get silenced
# LLM's reward function transformed to nn.Module
class RewardFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.rotation_weight = nn.Parameter(torch.tensor([17.282897], requires_grad=True))
        self.rotation_temp = nn.Parameter(torch.tensor([19.708775], requires_grad=True))
        self.distance_threshold = nn.Parameter(torch.tensor([18.531124], requires_grad=True))

    def forward(self, object_rot, goal_rot):
        # Scalar weights and parameters (these will become trainable)
    
        # Convert quaternions to euler for easier comparison
        object_rot_euler = quat_to_euler(object_rot)
        goal_rot_euler = quat_to_euler(goal_rot)
    
        # Calculate the distance between current rotation and target rotation
        rotation_distance = torch.linalg.norm(object_rot_euler- goal_rot_euler)
    
        # Create a reward for getting close to the target rotation
        # The reward should increase as the rotation gets closer to the goal
        rotation_reward = self.rotation_weight * torch.exp(-self.rotation_temp * rotation_distance)
    
        # Create a bonus reward when the rotation is within the threshold
        success_reward = (rotation_distance < self.distance_threshold).float()
    
        # Combine the rewards
        total_reward = rotation_reward + success_reward
    
        # Create a dictionary of the reward components
        individual_reward_components = {
            'rotation_reward': rotation_reward,
            'success_reward': success_reward,
        }
    
        return total_reward, individual_reward_components

comparisons = torch.tensor(preference_pairs, dtype=torch.float32)
num_items = int(comparisons[:, :2].max().item()) + 1


def get_rollout_observations(rollout_path):
    with open(rollout_path, 'r') as f:
        f.readline()  # skip first line (reward score)
        data = [eval(line) for line in f]
    
    data = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    object_rot_list = []
    goal_rot_list = []

    for i in range(data.shape[0]):
        object_rot, goal_rot = return_env_vars(data[i])
        object_rot_list.append(object_rot)
        goal_rot_list.append(goal_rot)

    return object_rot_list, goal_rot_list


def bradley_terry_loss(model, comparisons):
    loss_fn = torch.nn.CrossEntropyLoss()

    # Load rollout data and compute scalar reward per trajectory
    rollout_data = {
        i: get_rollout_observations(f"./preference_data/{path}")
        for i, path in enumerate(data_points)
    }

    rollout_rewards = {}
    for i, (object_rots, goal_rots) in rollout_data.items():
        total_reward = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        for obj_rot, goal_rot in zip(object_rots, goal_rots):
            total_reward = total_reward + model(obj_rot, goal_rot)  # No in-place operation
        rollout_rewards[i] = total_reward

    left = torch.stack([rollout_rewards[int(i)].squeeze() for i in comparisons[:, 0]])
    right = torch.stack([rollout_rewards[int(i)].squeeze() for i in comparisons[:, 1]])
    rewards = torch.stack([left, right], dim=1)  # Ensure shape [batch_size, 2]

    targets = comparisons[:, -1].to(torch.long).squeeze()  # Ensure shape [batch_size]

    with torch.no_grad():
        pred = torch.argmax(rewards, dim=1)
        acc = (pred == targets).float().mean()
        print(f"Pairwise accuracy: {acc.item():.2f}")

    # print(f"Targets shape: {targets.shape}")
    # print(f"Rewards shape: {rewards.shape}")
    # exit()
    return loss_fn(rewards, targets)





# trainable_params = torch.nn.Parameter(torch.tensor([12.0, -1.0, 4.0], dtype=torch.float32, requires_grad=True))
# optimizer = optim.LBFGS([trainable_params])

# Optimize using Bradley-Terry loss
def train_model(model):
    optimizer = optim.Adam(model.parameters(), lr=5e-2)
    print(f"Loss Before Update: {bradley_terry_loss(model, comparisons)}")

    print(f"Loss: {bradley_terry_loss(model,comparisons)}")

    
    for i in range(20):
        optimizer.zero_grad()
        loss = bradley_terry_loss(model, comparisons)
        loss.backward()

        # print(model.dist_penalty_scaler.grad)
        # print(model.reward_temp.grad)
        # print(model.garbage_term_scaler.grad)
        # print(model.survival_scaler.grad)

        # if i%500==0:
        print(f"Loss: {bradley_terry_loss(model,comparisons)}")
        optimizer.step()
        # optimizer.step(closure)
    # Check rewards after training

    print("\nLearned parameters:")
    for name, param in model.named_parameters():
        print(name, param.item())
    # print("\nGood example reward (after training):", model(good_example_x).item())
    # print("Bad example reward (after training):", model(bad_example_x).item())
    # Calculate the loss after training
    print(f"Loss After Update: {bradley_terry_loss(model, comparisons)}")

# def closure():
#     optimizer.zero_grad()
#     loss = bradley_terry_loss(trainable_params)
#     loss.backward()  # Compute gradients
#     print(f"Gradients: {trainable_params.grad}")
#     return loss

print("Before training:")
score = 0
max_score = len(comparisons)

# print(f"{score}/{max_score}")

# print("Training...")
# for _ in range(3):  # Adjust number of optimization steps
#     optimizer.step(closure)


# learned_params = trainable_params.detach().numpy()  # Convert back to NumPy
# print("Final Parameters:", learned_params)

train_model(RewardFunction())