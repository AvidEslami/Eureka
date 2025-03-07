import torch
import torch.nn as nn
import torch.optim as optim

good_example_x = 3
bad_example_x = 6

# Updated paired comparison data to ensure meaningful ranking constraints
# (smaller x values should be preferred over larger ones, for now x is our fitness)
comparisons = torch.tensor([
    (good_example_x, bad_example_x, 1),
    (2, 5, 1),  # Example: 2 should be preferred over 5
    (1, 4, 1),  # Example: 1 should be preferred over 4
    (0, 3, 1),  # Example: 0 should be preferred over 3
    (1, 5, 1),  # Example: 1 should be preferred over 5
    (2, 4, 1)   # Example: 2 should be preferred over 4
], dtype=torch.float32)

num_items = int(comparisons[:, :2].max().item()) + 1 # Number of items

# Linear reward function
class LinearReward(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial parameter for training
        self.w = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        return self.w * x

# Quadratic reward function with 3 trainable parameters
class QuadraticReward(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial parameters for training (random initialization)
        self.a = nn.Parameter(torch.tensor([1.0]))
        self.b = nn.Parameter(torch.tensor([-2.0]))
        self.c = nn.Parameter(torch.tensor([-3.0]))

    def forward(self, x):
        return self.a * x**2 + self.b * x + self.c

# Bradley-Terry loss function
def bradley_terry_loss(model, comparisons):
    scores = torch.exp(torch.clamp(torch.tensor([model(x) for x in range(num_items)]), -50, 50))
    loss = 0
    for item1, item2, outcome in comparisons:
        prob = scores[int(item1)] / (scores[int(item1)] + scores[int(item2)])
        prob = torch.clamp(prob, 1e-10, 1 - 1e-10)
        loss -= outcome * torch.log(prob) + (1 - outcome) * torch.log(1 - prob)
    return loss

# Optimize using Bradley-Terry loss
def train_model(model):
    optimizer = optim.LBFGS(model.parameters(), lr=0.1)
    
    def closure():
        optimizer.zero_grad()
        loss = bradley_terry_loss(model, comparisons)
        loss.backward()
        return loss
    
    # Check rewards before training
    
    print(f"\nTraining {model.__class__.__name__}")
    print("Good example reward (before training):", model(good_example_x).item())
    print("Bad example reward (before training):", model(bad_example_x).item())
    
    optimizer.step(closure)

    # Check rewards after training
    
    print("\nLearned parameters:")
    for name, param in model.named_parameters():
        print(name, param.item())
    print("Good example reward (after training):", model(good_example_x).item())
    print("Bad example reward (after training):", model(bad_example_x).item())

train_model(LinearReward())
train_model(QuadraticReward())

# Harder example:
hard_comparisons = torch.tensor([
    (2, 5, 0),  # Prefer 5 over 2
    (4, 7, 0),  # Prefer 7 over 4
    (6, 8, 1)   # Prefer 6 over 8 (inverted preference)
], dtype=torch.float32)
num_items = int(hard_comparisons[:, :2].max().item()) + 1

# Quadratic reward function with 3 trainable parameters
class HarderQuadraticReward(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([-1.0]))
        self.b = nn.Parameter(torch.tensor([1.0]))
        self.c = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        return self.a * x**2 + self.b * x + self.c

train_model(HarderQuadraticReward())

# Trajectory example:
rollouts = {
    2: torch.tensor([0, 1, 2, 3, 2, 1, 3], dtype=torch.float32),
    5: torch.tensor([5, 4, 4, 6, 6, 7, 3], dtype=torch.float32),
    7: torch.tensor([7, 8, 6, 6, 7, 7, 8], dtype=torch.float32),
    4: torch.tensor([4, 5, 3, 5, 3, 5, 3], dtype=torch.float32),
    6: torch.tensor([6, 7, 5, 7, 5, 6, 6], dtype=torch.float32),
    8: torch.tensor([6, 10, 8, 8, 8, 8, 8], dtype=torch.float32)
}

# Quadratic reward function with 3 trainable parameters
class TrajectoryReward(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([-1.0]))
        self.b = nn.Parameter(torch.tensor([1.0]))
        self.c = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x_array):
        return torch.mean(self.a * x_array**2 + self.b * x_array + self.c)

### Bradley-Terry Loss for reward_func3
def bradley_terry_loss_trajectory(model):
    scores = {x: torch.exp(torch.clamp(model(rollouts[x]), -50, 50)) for x in rollouts}
    loss = 0
    for item1, item2, outcome in hard_comparisons:
        prob = scores[int(item1)] / (scores[int(item1)] + scores[int(item2)])
        prob = torch.clamp(prob, 1e-10, 1 - 1e-10)
        loss -= outcome * torch.log(prob) + (1 - outcome) * torch.log(1 - prob)
    return loss

def train_trajectory_model(model):
    optimizer = optim.LBFGS(model.parameters(), lr=0.1)
    
    def closure():
        optimizer.zero_grad()
        loss = bradley_terry_loss_trajectory(model)
        loss.backward()
        return loss
    
    print(f"\nTraining {model.__class__.__name__}")
    optimizer.step(closure)
    print("\nLearned parameters:")
    for name, param in model.named_parameters():
        print(name, param.item())

train_trajectory_model(TrajectoryReward())
