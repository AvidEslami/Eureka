import torch
import numpy as np
from scipy.optimize import minimize
import torch.optim as optim

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
def reward_function(parameters, object_rot, goal_rot):
    # compute the cosine similarity between object's current orientation and the target orientation
    similarity = torch.nn.functional.cosine_similarity(object_rot, goal_rot, dim=-1)

    # transform similarity to a distance-like metric
    distance = 1 - similarity
    
    dist_penalty_scaler = parameters[0] # This should learn to be negative if the algo works

    # larger reward the smaller the rotation difference
    reward = dist_penalty_scaler * distance #LLM had -1 instead of the dist_penalty_scaler

    # temperature parameter adjusted for reward scaling
    # reward_temp = parameters[1] # LLM set this to 1
    reward_temp = torch.clamp(parameters[1], min=0.5, max=15) # prevent explosion

    # scale the raw reward using an exponential function
    # scaled_reward = torch.exp(torch.clamp(reward / reward_temp, min=-50, max=50)) # prevent gradient explosion NECESSARY
    safe_reward = reward / reward_temp
    safe_reward = safe_reward.clone().detach().requires_grad_(True)  # Prevent in-place issues
    scaled_reward = torch.nn.functional.softplus(torch.clamp(safe_reward, min=-10, max=10)) # torch.exp had grad issues


    # GARBAGE TERM
    garbage_term_scaler = parameters[2] # We expect the algorithm to silence this term since it shouldn't help
    # I can't think of a bad reward term that isn't worth flipping

    # for now try with a pure noise reward
    # noise_reward = torch.randn(1) * garbage_term_scaler
    scaled_reward += garbage_term_scaler * torch.randn_like(scaled_reward) # Ideally this gets silenced as well?

    return scaled_reward

comparisons = np.array(preference_pairs)

def get_rollout_reward(rollout_path, params):
    # Skips the first line (success score), then repeatedly calls return_env_Vars and reward_function
    # reward_sum = 0
    # with open(rollout_path, 'r') as f:
    #     f.readline()
    #     for line in f:
    #         line = eval(line)
    #         obs_buf = torch.tensor(line)
    #         object_rot, goal_rot = return_env_vars(obs_buf)
    #         reward_sum += reward_function(params, object_rot, goal_rot)
    # return reward_sum
    # Note: FILE IO IS NOT DIFFERENTIABLE?

    with open(rollout_path, 'r') as f:
        f.readline()
        data = [eval(line) for line in f]
        data = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    
    reward_sum = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) # fails if not forced to be a float
    for i in range(data.shape[0]):
        object_rot, goal_rot = return_env_vars(data[i])
        reward_sum = reward_sum + reward_function(params, object_rot, goal_rot)
    return reward_sum

def bradley_terry_loss(trainable_params):
    # scores = np.exp(np.clip([reward_func3(trainable_params, x) for x in range(num_items)], -50, 50))  # Prevent overflow
    scores = {}
    for i,x in enumerate(data_points):
        clamped_reward = torch.clamp(get_rollout_reward(f"./preference_data/{x}", trainable_params), min=-500, max=500) / 10
        # clamped_reward = clamped_reward / 10
        scores[i] = torch.exp(clamped_reward)
    loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    for item1, item2, outcome in comparisons:
        prob = scores[item1] / (scores[item1] + scores[item2])
        prob = torch.clamp(prob, min=1e-6, max=1 - 1e-6)   # Prevent log(0) issues
        loss = loss - (outcome * torch.log(prob) + (1 - outcome) * torch.log(1 - prob))
    return loss

trainable_params = torch.nn.Parameter(torch.tensor([12.0, -1.0, 4.0], dtype=torch.float32, requires_grad=True))
optimizer = optim.LBFGS([trainable_params])

def closure():
    optimizer.zero_grad()
    loss = bradley_terry_loss(trainable_params)
    loss.backward()  # Compute gradients
    print(f"Gradients: {trainable_params.grad}")
    return loss

print("Before training:")
score = 0
max_score = len(comparisons)

# print(f"{score}/{max_score}")

print("Training...")
for _ in range(3):  # Adjust number of optimization steps
    optimizer.step(closure)


learned_params = trainable_params.detach().numpy()  # Convert back to NumPy

print("Final Parameters:", learned_params)