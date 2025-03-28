import torch
import torch.nn as nn
import torch.optim as optim

# Preference data
data_points = [
    "ShadowHand_2025-03-07_01-39-39.txt",
    "ShadowHand_2025-03-07_01-40-05.txt",
    "ShadowHand_2025-03-07_01-40-51.txt",
    "ShadowHand_2025-03-07_01-41-18.txt",
    "ShadowHand_2025-03-07_01-41-44.txt",
]

# Construct preference pairs based on prepended scores
preference_pairs = []
for i in range(len(data_points)):
    for j in range(len(data_points)):
        if i != j:
            with open(f"./preference_data/{data_points[i]}", 'r') as f1, \
                 open(f"./preference_data/{data_points[j]}", 'r') as f2:
                score_i = float(f1.readline())
                score_j = float(f2.readline())
            preference_pairs.append((i, j, 0 if score_i > score_j else 1))

comparisons = torch.tensor(preference_pairs, dtype=torch.float32)

# Load rollout trajectories into running memory
def load_rollouts(paths):
    rollouts = {}
    for i, path in enumerate(paths):
        with open(f"./preference_data/{path}", 'r') as f:
            f.readline()  # Skip score
            obs = [eval(line) for line in f]
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            rollouts[i] = obs_tensor
    return rollouts

rollout_data = load_rollouts(data_points)

# Simple MLP reward function (made simpler to prevent instant preference convergence)
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
        return rewards.sum() 

# Bradley-Terry loss
def bradley_terry_loss(model, comparisons, rollout_data):
    rewards = {}
    for i in rollout_data:
        rollout = rollout_data[i]
        rollout.requires_grad_(True)
        total_reward = model(rollout)
        rewards[i] = total_reward

    left = torch.stack([rewards[int(i)].squeeze() for i in comparisons[:, 0]])
    right = torch.stack([rewards[int(i)].squeeze() for i in comparisons[:, 1]])
    scores = torch.stack([left, right], dim=1)

    targets = comparisons[:, -1].to(torch.long).squeeze()

    with torch.no_grad():
        pred = torch.argmax(scores, dim=1)
        acc = (pred == targets).float().mean()
        print(f"Pairwise accuracy: {acc.item():.2f}")

    return nn.CrossEntropyLoss()(scores, targets)

# Training loop
def train(model, rollout_data, comparisons, epochs=20, lr=5e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = bradley_terry_loss(model, comparisons, rollout_data)
        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")
        loss.backward()
        optimizer.step()

    print("Final learned parameters:")
    for name, param in model.named_parameters():
        print(name, param.data.norm().item())


obs_dim = rollout_data[0].shape[1]
model = MLPReward(obs_dim)
train(model, rollout_data, comparisons)

# Save model state dict
torch.save(model.state_dict(), "mlp_reward.pt")
print("Model Saved as mlp_reward.pt")
# Wrap and script the model for deployment? Might works in the torch.jit setup
scripted_model = torch.jit.script(model)
scripted_model.save("mlp_reward_scripted.pt")
print("Model Jit Saved as mlp_reward_scripted.pt")