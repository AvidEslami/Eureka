import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tabulate import tabulate  # for pretty printing tables

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

# Move bradley_terry_loss_general outside the test function
def bradley_terry_loss_general(trainable_params, reward_func, comparisons, num_items, reg_lambda=0.01):
    # Compute rewards and scale them to prevent overflow
    rewards = np.array([reward_func(trainable_params, x) for x in range(num_items)])
    # Center and scale rewards
    rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
    scores = np.exp(np.clip(rewards, -10, 10))
    
    # Compute loss using Bradley-Terry model
    loss = 0
    for item1, item2, outcome in comparisons:
        prob = scores[item1] / (scores[item1] + scores[item2])
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        loss -= outcome * np.log(prob) + (1 - outcome) * np.log(1 - prob)
    
    # Add L2 regularization
    loss += reg_lambda * np.sum(trainable_params ** 2)
    return loss

def test_bradley_terry_loss_general():
    """Test case for the generalized Bradley-Terry loss with diverse preferences"""
    
    def reward_func_polynomial(params, x):
        return sum((p * (x ** i))/np.math.factorial(i) for i, p in enumerate(params))
    
    def safe_prob(r1, r2):
        rewards = np.array([r1, r2])
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        scores = np.exp(np.clip(rewards, -10, 10))
        return scores[0] / (scores[0] + scores[1])
    
    # More complex test cases with different patterns
    num_items = 10
    
    # Generate diverse comparison patterns:
    comparisons = [
        # Pattern 1: Prefer lower values
        (0, 3, 1), (1, 4, 1), (2, 5, 1),
        
        # Pattern 2: Prefer middle values (around 5)
        (5, 2, 1), (5, 8, 1), (4, 1, 1), (4, 7, 1),
        
        # Pattern 3: Cyclic preferences
        (1, 3, 1), (3, 6, 1), (6, 1, 0),
        
        # Pattern 4: Strong preferences (far apart items)
        (0, 9, 1), (1, 8, 1), (2, 7, 1),
        
        # Pattern 5: Close preferences (adjacent items)
        (4, 5, 1), (5, 6, 1), (6, 7, 1),
        
        # Pattern 6: Indirect relationships
        (1, 4, 1), (4, 7, 1), (7, 9, 1)
    ]
    comparisons = np.array(comparisons)
    
    print("Running Bradley-Terry Loss Test with Diverse Preferences...")
    print(f"Number of items: {num_items}")
    print(f"Number of comparisons: {len(comparisons)}")
    
    # Test with higher degree polynomial
    n_params = 6  # 5th degree polynomial
    initial_params = np.ones(n_params) * 0.1
    
    result = minimize(
        lambda params: bradley_terry_loss_general(params, reward_func_polynomial, comparisons, num_items),
        initial_params,
        method="L-BFGS-B",
        options={'ftol': 1e-8}
    )
    
    print(f"\nOptimization Results:")
    print(f"{'Initial parameters:':<20} {initial_params}")
    print(f"{'Learned parameters:':<20} {result.x}")
    print(f"{'Optimization success:':<20} {result.success}")
    
    # Create comparison results table grouped by patterns
    patterns = ["Lower values", "Middle values", "Cyclic", "Strong", "Adjacent", "Indirect"]
    pattern_sizes = [3, 4, 3, 3, 3, 3]
    
    table_data = []
    start_idx = 0
    for pattern, size in zip(patterns, pattern_sizes):
        pattern_comparisons = comparisons[start_idx:start_idx + size]
        table_data.append([f"--- {pattern} Preferences ---", "", "", "", "", ""])
        
        for item1, item2, outcome in pattern_comparisons:
            init_reward1 = reward_func_polynomial(initial_params, item1)
            init_reward2 = reward_func_polynomial(initial_params, item2)
            learned_reward1 = reward_func_polynomial(result.x, item1)
            learned_reward2 = reward_func_polynomial(result.x, item2)
            
            init_prob = safe_prob(init_reward1, init_reward2)
            learned_prob = safe_prob(learned_reward1, learned_reward2)
            
            preference_satisfied = (outcome == 1 and learned_reward1 > learned_reward2) or \
                                 (outcome == 0 and learned_reward1 < learned_reward2)
            
            table_data.append([
                f"{item1} vs {item2}",
                f"{init_reward1:.3f} vs {init_reward2:.3f}",
                f"{learned_reward1:.3f} vs {learned_reward2:.3f}",
                f"{init_prob:.2f}",
                f"{learned_prob:.2f}",
                "✓" if preference_satisfied else "✗"
            ])
        
        start_idx += size
    
    headers = ["Comparison", "Initial Rewards", "Learned Rewards", 
              "Initial P(i>j)", "Learned P(i>j)", "Satisfied"]
    print("\nComparison Results by Pattern:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Print reward values and preference chain
    print("\nLearned Reward Values and Preference Chain:")
    reward_data = []
    rewards = [reward_func_polynomial(result.x, x) for x in range(num_items)]
    
    for i in range(num_items):
        if i < num_items - 1:
            prob = safe_prob(rewards[i], rewards[i + 1])
            prob_str = f"{prob:.2f}"
        else:
            prob_str = "-"
            
        reward_data.append([
            i, 
            f"{rewards[i]:.3f}",
            prob_str
        ])
    
    print(tabulate(reward_data, headers=["Item", "Reward", "P(prefer over next)"], tablefmt="grid"))
    
    # Count violations by pattern
    print("\nViolations by Pattern:")
    start_idx = 0
    for pattern, size in zip(patterns, pattern_sizes):
        pattern_violations = sum(1 for row in table_data[start_idx+1:start_idx+size+1] 
                               if row[-1] == "✗")
        print(f"{pattern+':':<15} {pattern_violations}/{size}")
        start_idx += size + 1

if __name__ == "__main__":
    test_bradley_terry_loss_general()