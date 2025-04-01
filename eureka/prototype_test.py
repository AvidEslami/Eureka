import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.optimize import minimize

torch.autograd.set_detect_anomaly(True)

# Algorithm 1 EUREKA
# 1: Require: Task description l, environment code M ,
# coding LLM LLM, fitness function F , initial prompt prompt
# 2: Hyperparameters: search iteration N , iteration batch size K
# 3: for N iterations do
# 4: // Sample K reward code from LLM
# 5: R1, ..., Rk ∼ LLM(l, M, prompt)
# -> We Come Here: Tune K Reward Codes Using previous preference data
# 6: // Evaluate TUNED reward candidates
# 7: s1 = F (R1), ..., sK = F (RK )
# 8: // Reward reflection
# 9: prompt := prompt : Reflection(Rn
# best, sn
# best),
# where best = arg maxk s1, ..., sK
# 10: // Update Eureka reward
# 11: REureka, sEureka = (Rn
# best, sn
# best), if sn
# best > sEureka
# 12: Output: Eureka


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
class RewardFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.dist_penalty_scaler = nn.Parameter(torch.tensor([0.9], requires_grad=True))
        self.reward_temp = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.garbage_term_scaler = nn.Parameter(torch.tensor([0.001], requires_grad=True))
        self.survival_scaler = nn.Parameter(torch.tensor([0.1], requires_grad=True))
    def forward(self, object_rot, goal_rot):
        # compute the cosine similarity between object's current orientation and the target orientation
        similarity = torch.nn.functional.cosine_similarity(object_rot, goal_rot, dim=-1)

        # transform similarity to a distance-like metric
        distance = 1 - similarity
        
        # dist_penalty_scaler = self.dist_penalty_scaler # This should learn to be negative if the algo works

        # larger reward the smaller the rotation difference
        reward = self.dist_penalty_scaler * distance #LLM had -1 instead of the dist_penalty_scaler

        # temperature parameter adjusted for reward scaling
        # reward_temp = self.reward_temp # LLM set this to 1
        reward_temp = torch.clamp(self.reward_temp, min=0.5, max=15) # prevent explosion

        # scale the raw reward using an exponential function
        # scaled_reward = torch.exp(torch.clamp(reward / reward_temp, min=-50, max=50)) # prevent gradient explosion NECESSARY
        # safe_reward = reward / reward_temp
        # safe_reward = safe_reward.clone().detach().requires_grad_(True)  # Prevent in-place issues
        # scaled_reward = torch.nn.functional.softplus(torch.clamp(safe_reward, min=-10, max=10)) # torch.exp had grad issues
        safe_reward = reward / reward_temp  # this keeps gradient flow
        safe_reward = torch.clamp(safe_reward, min=-10, max=10)
        scaled_reward = torch.nn.functional.softplus(safe_reward)


        # GARBAGE TERM
        # garbage_term_scaler = self.garbage_term_scaler # We expect the algorithm to silence this term since it shouldn't help
        # I can't think of a bad reward term that isn't worth flipping

        # Syrvuvak Reward, just adds a constant equal to the parameter, encourages longer rollouts
        survival_reward = self.survival_scaler
        # for now try with a pure noise reward
        # noise_reward = torch.randn(1) * garbage_term_scaler
        scaled_reward += survival_reward
        scaled_reward += self.garbage_term_scaler * torch.randn_like(scaled_reward) # Ideally this gets silenced as well?

        return scaled_reward

comparisons = torch.tensor(preference_pairs, dtype=torch.float32)
num_items = int(comparisons[:, :2].max().item()) + 1

# def get_rollout_reward(rollout_path, params):
#     # Skips the first line (success score), then repeatedly calls return_env_Vars and reward_function
#     # reward_sum = 0
#     # with open(rollout_path, 'r') as f:
#     #     f.readline()
#     #     for line in f:
#     #         line = eval(line)
#     #         obs_buf = torch.tensor(line)
#     #         object_rot, goal_rot = return_env_vars(obs_buf)
#     #         reward_sum += reward_function(params, object_rot, goal_rot)
#     # return reward_sum
#     # Note: FILE IO IS NOT DIFFERENTIABLE?

#     with open(rollout_path, 'r') as f:
#         f.readline()
#         data = [eval(line) for line in f]
#         data = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    
#     reward_sum = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) # fails if not forced to be a float
#     for i in range(data.shape[0]):
#         object_rot, goal_rot = return_env_vars(data[i])
#         reward_sum = reward_sum + reward_function(params, object_rot, goal_rot)
#     return reward_sum

# def get_rollout_observations(rollout_path):
#     # Note: FILE IO MIGHT NOT BE DIFFERENTIABLE?

#     with open(rollout_path, 'r') as f:
#         f.readline()
#         data = [eval(line) for line in f]
#         data = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    
#     reward_sum = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) # fails if not forced to be a float
#     object_rot_list, goal_rot_list = [], []
#     for i in range(data.shape[0]):
#         object_rot, goal_rot = return_env_vars(data[i])
#         object_rot_list.append(object_rot)
#         goal_rot_list.append(goal_rot)
#     return object_rot_list, goal_rot_list

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


# def bradley_terry_loss(trainable_params):
#     # scores = np.exp(np.clip([reward_func3(trainable_params, x) for x in range(num_items)], -50, 50))  # Prevent overflow
#     scores = {}
#     for i,x in enumerate(data_points):
#         clamped_reward = torch.clamp(get_rollout_reward(f"./preference_data/{x}", trainable_params), min=-500, max=500) / 10
#         # clamped_reward = clamped_reward / 10
#         scores[i] = torch.exp(clamped_reward)
#     loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
#     for item1, item2, outcome in comparisons:
#         prob = scores[item1] / (scores[item1] + scores[item2])
#         prob = torch.clamp(prob, min=1e-6, max=1 - 1e-6)   # Prevent log(0) issues
#         loss = loss - (outcome * torch.log(prob) + (1 - outcome) * torch.log(1 - prob))
#     return loss

# def bradley_terry_loss_trajectory(model):
#     rollout_data = {i: get_rollout_observations(f"./preference_data/{x}") for i, x in enumerate(data_points)}
#     # Call model on every entry in rollout_data
#     # scores = {x: torch.exp(torch.clamp(model(rollout_data[x]), -50, 50)) for x in rollout_data}
#     scores = {}
#     for rollout in rollout_data:
#         scores[rollout] = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
#         for i in range(len(rollout_data[rollout][0])):
#             object_rot, goal_rot = rollout_data[rollout][0][i], rollout_data[rollout][1][i]
#             scores[rollout] = scores[rollout] + torch.exp(torch.clamp(model(object_rot, goal_rot), -50, 50))
#     loss = 0
#     epsilon = 1e-7
#     for item1, item2, outcome in comparisons:
#         prob = scores[int(item1)] / (scores[int(item1)] + scores[int(item2)])
#         prob = torch.clamp(prob, epsilon, 1 - epsilon)
#         loss -= outcome * torch.log(prob) + (1 - outcome) * torch.log(1 - prob)
#     return loss

# def bradley_terry_loss(model, comparisons):
#     # scores = torch.exp(torch.clamp(torch.stack([model(torch.tensor(x, dtype=torch.float32)) for x in range(num_items)]), -50, 50))
#     # scores = {}
#     loss = torch.nn.CrossEntropyLoss()
#     # Get rollout data and compute reward for each data point
#     rollout_data = {i: get_rollout_observations(f"./preference_data/{x}") for i, x in enumerate(data_points)}
#     # Call model on every entry in rollout_data
#     # scores = {x: torch.exp(torch.clamp(model(rollout_data[x]), -50, 50)) for x in rollout_data}
#     rollout_rewards = {}
#     for rollout in rollout_data:
#         # rollout_rewards[rollout] = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
#         rollout_rewards[rollout] = torch.tensor(0.0, dtype=torch.float32)
#         for i in range(len(rollout_data[rollout][0])):
#             object_rot, goal_rot = rollout_data[rollout][0][i], rollout_data[rollout][1][i]
#             rollout_rewards[rollout] = rollout_rewards[rollout] + model(object_rot, goal_rot)

#     targets = comparisons[:,-1]
#     # Replace the first two columns with the rollout reward for the number at the value
#     left = torch.stack([rollout_rewards[int(x)] for x in comparisons[:, 0]])
#     right = torch.stack([rollout_rewards[int(x)] for x in comparisons[:, 1]])
#     rewards = torch.stack([left, right], dim=1).squeeze(-1)  # Shape: [batch_size, 2]
#     targets = comparisons[:,-1]
#     # targets = torch._cast_Int(targets)
#     targets = targets.to(torch.long)
#     loss_values = loss(rewards, targets)
#     return loss_values.mean()


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