import os
import torch
import torch.nn as nn


NN_REWARD_SCALE = 1  # Scale factor for NN reward output (must match training)


class NNReward(nn.Module):
    def __init__(self, obs_dim: int, scale: float = NN_REWARD_SCALE):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 768),
            nn.LeakyReLU(0.1),
            nn.Linear(768, 384),
            nn.LeakyReLU(0.1),
            nn.Linear(384, 1),
            nn.Tanh(),  # Output in range [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self.scale  # Scale output to [-20, 20]


def load_nn_reward(ckpt_path: str, obs_dim: int, device: torch.device) -> nn.Module:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"NN reward checkpoint not found at: {ckpt_path}")
    model = NNReward(obs_dim=obs_dim).to(device)
    state = torch.load(ckpt_path, map_location=device)
    
    # Handle both checkpoint formats:
    # 1. Raw state_dict: {"net.0.weight": ..., "net.0.bias": ..., ...}
    # 2. Training checkpoint: {"epoch": ..., "model_state_dict": {...}, ...}
    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
        print(f"Loaded NN reward from training checkpoint (epoch {state.get('epoch', '?')})")
    else:
        state_dict = state
    
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


