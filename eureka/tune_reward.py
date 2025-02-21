import numpy as np
from scipy.optimize import minimize

good_example_x = 3
bad_example_x = 6

# Updated paired comparison data to ensure meaningful ranking constraints
# (smaller x values should be preferred over larger ones, for now x is our fitness)
comparisons = np.array([
    (good_example_x, bad_example_x, 1),
    (2, 5, 1),  # Example: 2 should be preferred over 5
    (1, 4, 1),  # Example: 1 should be preferred over 4
    (0, 3, 1),  # Example: 0 should be preferred over 3
    (1, 5, 1),  # Example: 1 should be preferred over 5
    (2, 4, 1)   # Example: 2 should be preferred over 4
])

num_items = np.max(comparisons[:, :2]) + 1  # Number of items

# Linear reward function
def reward_func1(trainable_param1, x):
    return trainable_param1 * x

# Quadratic reward function with 3 trainable parameters
def reward_func2(params, x):
    a, b, c = params
    return a * x**2 + b * x + c  # Nonlinear function

### Bradley-Terry Loss for reward_func1
def bradley_terry_loss1(trainable_param):
    scores = np.exp(np.clip([reward_func1(trainable_param, x) for x in range(num_items)], -50, 50))  # Prevent overflow
    loss = 0
    for item1, item2, outcome in comparisons:
        prob = scores[item1] / (scores[item1] + scores[item2])
        prob = np.clip(prob, 1e-10, 1 - 1e-10)  # Prevent log(0) or division by 0 issues
        loss -= outcome * np.log(prob) + (1 - outcome) * np.log(1 - prob)
    return loss

# Initial parameter for training
initial_param = np.array([1.0])

print("EXPERIMENT 1: Linear reward function")
print("Good example reward (before training):", reward_func1(initial_param, good_example_x))
print("Bad example reward (before training):", reward_func1(initial_param, bad_example_x))

# Optimize using Bradley-Terry loss
result = minimize(bradley_terry_loss1, initial_param, method="L-BFGS-B")
learned_param = result.x[0]

print("\nLearned parameter:", learned_param)
print("Good example reward (after training):", reward_func1(learned_param, good_example_x))
print("Bad example reward (after training):", reward_func1(learned_param, bad_example_x),"\n")

### Bradley-Terry Loss for reward_func2
def bradley_terry_loss2(trainable_params):
    scores = np.exp(np.clip([reward_func2(trainable_params, x) for x in range(num_items)], -50, 50))  # Prevent overflow
    loss = 0
    for item1, item2, outcome in comparisons:
        prob = scores[item1] / (scores[item1] + scores[item2])
        prob = np.clip(prob, 1e-10, 1 - 1e-10)  # Prevent log(0) issues
        loss -= outcome * np.log(prob) + (1 - outcome) * np.log(1 - prob)
    return loss

# Initial parameters for training (random initialization)
initial_params = [1, -2, -3]  # a, b, c

# Check rewards before training
good_reward = reward_func2(initial_params, good_example_x)
bad_reward = reward_func2(initial_params, bad_example_x)
print("EXPERIMENT 2: Quadratic reward function")
print("Good example reward (before training):", good_reward)
print("Bad example reward (before training):", bad_reward)

# Optimize using Bradley-Terry loss
result = minimize(bradley_terry_loss2, initial_params, method="L-BFGS-B")
learned_params = result.x

print("\nLearned parameters (a, b, c):", learned_params)

# Check rewards after training
good_reward = reward_func2(learned_params, good_example_x)
bad_reward = reward_func2(learned_params, bad_example_x)

print("Good example reward (after training):", good_reward)
print("Bad example reward (after training):", bad_reward)


# Harder example:
comparisons = np.array([
    (2, 5, 0),  # Prefer 5 over 2
    (4, 7, 0),  # Prefer 7 over 4
    (6, 8, 1)   # Prefer 6 over 8 (inverted preference)
])

num_items = np.max(comparisons[:, :2]) + 1  # Number of items

# Quadratic reward function with 3 trainable parameters
def reward_func3(params, x):
    a, b, c = params
    return a * x**2 + b * x + c 

### Bradley-Terry Loss for reward_func3
def bradley_terry_loss3(trainable_params):
    scores = np.exp(np.clip([reward_func3(trainable_params, x) for x in range(num_items)], -50, 50))  # Prevent overflow
    loss = 0
    for item1, item2, outcome in comparisons:
        prob = scores[item1] / (scores[item1] + scores[item2])
        prob = np.clip(prob, 1e-10, 1 - 1e-10)  # Prevent log(0) issues
        loss -= outcome * np.log(prob) + (1 - outcome) * np.log(1 - prob)
    return loss

# Initial parameters for training (random initialization)
initial_params = [-1, 1, 1]  # a, b, c

print("EXPERIMENT 3: Quadratic Harder Preferences")

print("Before training:")
for trial in comparisons:
    print("Preference between", trial[0], "and", trial[1], ":", reward_func3(initial_params, trial[0]), reward_func3(initial_params, trial[1]))
    if trial[2] == 1:
        success = "Success" if reward_func3(initial_params, trial[0]) > reward_func3(initial_params, trial[1]) else "Fail"
    else:
        success = "Success" if reward_func3(initial_params, trial[0]) < reward_func3(initial_params, trial[1]) else "Fail"
    print(success)

# Optimize using Bradley-Terry loss
result = minimize(bradley_terry_loss3, initial_params, method="L-BFGS-B")
learned_params = result.x

print("\nLearned parameters (a, b, c):", learned_params)

# Check rewards after training

print("\nAfter training:")
for trial in comparisons:
    print("Preference between", trial[0], "and", trial[1], ":", reward_func3(learned_params, trial[0]), reward_func3(learned_params, trial[1]))
    if trial[2] == 1:
        success = "Success" if reward_func3(learned_params, trial[0]) > reward_func3(learned_params, trial[1]) else "Fail"
    else:
        success = "Success" if reward_func3(learned_params, trial[0]) < reward_func3(learned_params, trial[1]) else "Fail"
    print(success)


# Trajectory example:
rollouts = {
    2: np.array([0, 1, 2, 3, 2, 1, 3]),
    5: np.array([5, 4, 4, 6, 6, 7, 3]),
    7: np.array([7, 8, 6, 6, 7, 7, 8]),
    4: np.array([4, 5, 3, 5, 3, 5, 3]),
    6: np.array([6, 7, 5, 7, 5, 6, 6]),
    8: np.array([6, 10, 8, 8, 8, 8, 8])
}

comparisons = np.array([
    (2, 5, 0),  # Prefer 5 over 2
    (4, 7, 0),  # Prefer 7 over 4
    (6, 8, 1)   # Prefer 6 over 8 (inverted preference)
])

num_items = np.max(comparisons[:, :2]) + 1  # Number of items

# Quadratic reward function with 3 trainable parameters
def reward_func4(params, x_array):
    reward_sum = 0 
    a, b, c = params
    for x in x_array:
        reward_sum += a * x**2 + b * x + c
    reward_sum /= len(x_array)
    return reward_sum

### Bradley-Terry Loss for reward_func3
def bradley_terry_loss4 (trainable_params):
    # scores = np.exp(np.clip([reward_func3(trainable_params, x) for x in range(num_items)], -50, 50))  # Prevent overflow
    scores = {}
    for x in rollouts:
        scores[x] = np.exp(np.clip(reward_func4(trainable_params, rollouts[x]), -50, 50))
    loss = 0
    for item1, item2, outcome in comparisons:
        prob = scores[item1] / (scores[item1] + scores[item2])
        prob = np.clip(prob, 1e-10, 1 - 1e-10)  # Prevent log(0) issues
        loss -= outcome * np.log(prob) + (1 - outcome) * np.log(1 - prob)
    return loss

# Initial parameters for training (random initialization)
initial_params = [-1, 1, 1]  # a, b, c

print("EXPERIMENT 4: Quadratic Harder Preferences")

print("Before training:")
for trial in comparisons:
    print("Preference between", trial[0], "and", trial[1], ":", reward_func4(initial_params, rollouts[trial[0]]), reward_func4(initial_params, rollouts[trial[1]]))
    if trial[2] == 1:
        success = "Success" if reward_func4(initial_params, rollouts[trial[0]]) > reward_func4(initial_params, rollouts[trial[1]]) else "Fail"
    else:
        success = "Success" if reward_func4(initial_params, rollouts[trial[0]]) < reward_func4(initial_params, rollouts[trial[1]]) else "Fail"
    print(success)

# Optimize using Bradley-Terry loss
result = minimize(bradley_terry_loss4, initial_params, method="L-BFGS-B")
learned_params = result.x

print("\nLearned parameters (a, b, c):", learned_params)

# Check rewards after training

print("\nAfter training:")
for trial in comparisons:
    print("Preference between", trial[0], "and", trial[1], ":", reward_func4(learned_params, rollouts[trial[0]]), reward_func4(learned_params, rollouts[trial[1]]))
    if trial[2] == 1:
        success = "Success" if reward_func4(learned_params, rollouts[trial[0]]) > reward_func4(learned_params, rollouts[trial[1]]) else "Fail"
    else:
        success = "Success" if reward_func4(learned_params, rollouts[trial[0]]) < reward_func4(learned_params, rollouts[trial[1]]) else "Fail"
    print(success)