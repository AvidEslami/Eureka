import torch
import torch.nn as nn
import torch.optim as optim

good_example_x = 3
bad_example_x = 6

# Updated paired comparison data to ensure meaningful ranking constraints
# (smaller x values should be preferred over larger ones, for now x is our fitness)
# comparisons = torch.tensor([
#     (good_example_x, bad_example_x, 1),
#     (2, 5, 1),  # Example: 2 should be preferred over 5
#     (1, 4, 1),  # Example: 1 should be preferred over 4
#     (0, 3, 1),  # Example: 0 should be preferred over 3
#     (1, 5, 1),  # Example: 1 should be preferred over 5
#     (2, 4, 1)   # Example: 2 should be preferred over 4
# ], dtype=torch.float32)

# num_items = int(comparisons[:, :2].max().item()) + 1 # Number of items

# Harder example:
comparisons = torch.tensor([
    (2, 5, 0),  # Prefer 5 over 2
    (4, 7, 0),  # Prefer 7 over 4
    (6, 8, 1)   # Prefer 6 over 8 (inverted preference)
], dtype=torch.float32)
num_items = int(comparisons[:, :2].max().item()) + 1

# Linear reward function
class LinearReward(nn.Module):
    def __init__(self):
        super().__init__()

        # Initial parameter for training
        self.w = nn.Parameter(torch.tensor([1.0], requires_grad=True))

    def forward(self, x):
        return self.w * x

# Quadratic reward function with 3 trainable parameters
class QuadraticReward(nn.Module):
    def __init__(self):
        super().__init__()

        # Initial parameters for training (random initialization)
        self.a = nn.Parameter(torch.tensor([1.0],requires_grad=True))
        self.b = nn.Parameter(torch.tensor([-2.0],requires_grad=True))
        self.c = nn.Parameter(torch.tensor([-3.0],requires_grad=True))

    def forward(self, x):
        return self.a * x**2 + self.b * x + self.c

# Bradley-Terry loss function
def bradley_terry_loss(model, comparisons):
    scores = torch.exp(torch.clamp(torch.stack([model(torch.tensor(x, dtype=torch.float32)) for x in range(num_items)]), -50, 50))

    loss = 0

    # clamp so prob doesn't become 0 or 1 (will lead to NaNs), 1e-8 leads to NaN with harder quadratic reward rn
    epsilon = 1e-7
    for item1, item2, outcome in comparisons:
        prob = scores[int(item1)] / (scores[int(item1)] + scores[int(item2)])
        prob = torch.clamp(prob, epsilon, 1 - epsilon)
        loss -= outcome * torch.log(prob) + (1 - outcome) * torch.log(1 - prob)
    return loss
    # return loss.sum()


# Optimize using Bradley-Terry loss
def train_model(model):
    optimizer = optim.Adam(model.parameters(), lr=1)
    print(f"Loss Before Update: {bradley_terry_loss(model, comparisons)}")

    def closure():
        optimizer.zero_grad()
        loss = bradley_terry_loss(model, comparisons)
        # handling for nan, want warning not error
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN loss ")
            return loss

        loss.backward()
        return loss

    # Check rewards before training
    # TEMP CODE
    good_example_x = 6
    bad_example_x = 2
    print(f"\n---Training {model.__class__.__name__}---")
    print("\nGood example reward (before training):", model(good_example_x).item())
    print("Bad example reward (before training):", model(bad_example_x).item())

    for i in range(5): #TODO: Check if this is needed
        optimizer.step(closure)

    # Check rewards after training

    print("\nLearned parameters:")
    for name, param in model.named_parameters():
        print(name, param.item())
    print("\nGood example reward (after training):", model(good_example_x).item())
    print("Bad example reward (after training):", model(bad_example_x).item())
    # Calculate the loss after training
    print(f"Loss After Update: {bradley_terry_loss(model, comparisons)}")

# train_model(LinearReward())
# train_model(QuadraticReward())


# Quadratic reward function with 3 trainable parameters
class HarderQuadraticReward(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([-1.0],requires_grad=True))
        self.b = nn.Parameter(torch.tensor([-1.0],requires_grad=True))
        self.c = nn.Parameter(torch.tensor([1.0],requires_grad=True))

    def forward(self, x):
        return self.a * x**2 + self.b * x + self.c

# train_model(HarderQuadraticReward())

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
        self.a = nn.Parameter(torch.tensor([-1.0],requires_grad=True))
        self.b = nn.Parameter(torch.tensor([1.0],requires_grad=True))
        self.c = nn.Parameter(torch.tensor([1.0],requires_grad=True))

    def forward(self, x_array):
        return torch.mean(self.a * x_array**2 + self.b * x_array + self.c)

### Bradley-Terry Loss for reward_func3
def bradley_terry_loss_trajectory(model):
    scores = {x: torch.exp(torch.clamp(model(rollouts[x]), -50, 50)) for x in rollouts}
    loss = 0
    epsilon = 1e-7
    for item1, item2, outcome in comparisons:
        prob = scores[int(item1)] / (scores[int(item1)] + scores[int(item2)])
        prob = torch.clamp(prob, epsilon, 1 - epsilon)
        loss = loss - outcome * torch.log(prob) + (1 - outcome) * torch.log(1 - prob)
    return loss

def train_trajectory_model(model):
    optimizer = optim.Adam(model.parameters(), lr=100)

    def closure():
        optimizer.zero_grad()
        loss = bradley_terry_loss_trajectory(model)

        # handling for nan, want warning not error
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN loss ")
            return loss

        loss.backward()
        return loss

    print(f"\nTraining {model.__class__.__name__}")
    optimizer.step(closure)
    print("\nLearned parameters:")
    for name, param in model.named_parameters():
        print(name, param.item())

train_trajectory_model(TrajectoryReward())