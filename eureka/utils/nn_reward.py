import os
import torch
import torch.nn as nn
from typing import List


def get_activation(name: str) -> nn.Module:
    """Return activation module by name."""
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
    }
    return activations.get(name.lower(), nn.ReLU())


class RewardNetwork(nn.Module):
    """
    MLP that maps observation -> scalar reward.
    Architecture is configurable via hidden_layers list.
    """
    
    def __init__(self, obs_dim: int, hidden_layers: List[int], activation: str = "relu"):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(get_activation(activation))
            prev_dim = hidden_dim
        
        # Output layer: scalar reward
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim) or (batch, timesteps, obs_dim)
        Returns:
            reward: (batch,) or (batch, timesteps)
        """
        return self.net(obs).squeeze(-1)


def load_nn_reward(ckpt_path: str, obs_dim: int, device: torch.device, **kwargs) -> nn.Module:
    """
    Load a preference-trained reward model from checkpoint.
    
    Args:
        ckpt_path: Path to checkpoint file (.pt)
        obs_dim: Observation dimension (used as fallback if not in checkpoint)
        device: Device to load model onto
    
    Returns:
        Loaded RewardNetwork in eval mode with frozen parameters
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"NN reward checkpoint not found at: {ckpt_path}")
    
    state = torch.load(ckpt_path, map_location=device)
    
    # Extract architecture config from checkpoint
    if isinstance(state, dict) and "hidden_layers" in state:
        # Preference-trained checkpoint format
        hidden_layers = state["hidden_layers"]
        activation = state.get("activation", "relu")
        checkpoint_obs_dim = state.get("obs_dim", obs_dim)
        state_dict = state["model_state_dict"]
        epoch = state.get("epoch", "?")
        print(f"Loaded preference-trained reward (epoch {epoch}, arch: {checkpoint_obs_dim} -> {hidden_layers} -> 1)")
    else:
        raise ValueError(
            f"Checkpoint format not recognized. Expected preference-trained checkpoint with "
            f"'hidden_layers' and 'model_state_dict' keys. Got keys: {list(state.keys()) if isinstance(state, dict) else type(state)}"
        )
    
    # Create model with correct architecture
    model = RewardNetwork(
        obs_dim=checkpoint_obs_dim,
        hidden_layers=hidden_layers,
        activation=activation
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    
    return model
