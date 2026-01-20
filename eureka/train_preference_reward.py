"""
Train a neural network reward model using Bradley-Terry loss from VLM preference rankings.

Uses existing rankings from auto_preference_data/ranking_results.json - does NOT re-rank.
"""

import os
import json
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Tuple, Dict
import logging
import random

# ============================================================================
# CONFIGURATION - Modify these to change architecture and training
# ============================================================================

# Neural network architecture
HIDDEN_LAYERS = [256, 128, 64]  # List of hidden layer sizes
ACTIVATION = "relu"  # "relu", "tanh", "leaky_relu", "elu"

# Training parameters
LEARNING_RATE = 1e-4
EPOCHS = 100
BATCH_SIZE = 32  # Number of preference pairs per batch
VALIDATION_SPLIT = 0.2  # Fraction of pairs for validation
WEIGHT_DECAY = 1e-5  # L2 regularization

# Data parameters
DATA_FOLDER = "/home/gx22/Desktop/isaacgym/python/Eureka/eureka/auto_preference_data"
MAX_PAIRS = None  # Limit number of pairs (None = use all)
SEED = 42

# Output
CHECKPOINT_DIR = "/home/gx22/Desktop/isaacgym/python/Eureka/eureka/reward_checkpoints"
CHECKPOINT_INTERVAL = 5  # Save checkpoint every N epochs
SAVE_PATH = "/home/gx22/Desktop/isaacgym/python/Eureka/eureka/reward_model.pt"

# ============================================================================
# END CONFIGURATION
# ============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
    Architecture is configurable via HIDDEN_LAYERS.
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


def load_rankings(data_folder: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Load rankings from ranking_results.json.
    
    Returns:
        global_order: List of filenames sorted worst-to-best
        tied_pairs: List of (file1, file2) pairs that are tied
    """
    rankings_path = os.path.join(data_folder, "ranking_results.json")
    
    if not os.path.exists(rankings_path):
        raise FileNotFoundError(f"Rankings file not found: {rankings_path}")
    
    with open(rankings_path, 'r') as f:
        data = json.load(f)
    
    global_order = data["global_order"]
    tied_pairs = [tuple(pair) for pair in data.get("tied_pairs", [])]
    
    logging.info(f"Loaded {len(global_order)} ranked rollouts, {len(tied_pairs)} tied pairs")
    return global_order, tied_pairs


def load_rollout(filepath: str) -> torch.Tensor:
    """
    Parse obs_buf section from rollout file.
    
    Returns:
        obs_tensor: (T, obs_dim) tensor of observations
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find the Obs buf section
    obs_marker = "Obs buf:\n"
    obs_start = content.find(obs_marker)
    
    if obs_start == -1:
        raise ValueError(f"No 'Obs buf:' section found in {filepath}")
    
    obs_start += len(obs_marker)
    obs_section = content[obs_start:]
    
    # Parse each line as a list
    observations = []
    for line in obs_section.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            # Each line is [[...]] - a nested list with one observation
            parsed = ast.literal_eval(line)
            if isinstance(parsed, list) and len(parsed) > 0:
                if isinstance(parsed[0], list):
                    observations.append(parsed[0])
                else:
                    observations.append(parsed)
        except (ValueError, SyntaxError):
            break
    
    if not observations:
        raise ValueError(f"No valid observations found in {filepath}")
    
    return torch.tensor(observations, dtype=torch.float32)


def create_preference_pairs(
    global_order: List[str],
    tied_pairs: List[Tuple[str, str]],
    max_pairs: int = None
) -> List[Tuple[str, str]]:
    """
    Create preference pairs from global ordering.
    
    A rollout at a higher index is preferred over one at a lower index.
    Skip pairs that are in tied_pairs.
    
    Returns:
        List of (winner, loser) filename tuples
    """
    # Create set of tied pairs for fast lookup (both orderings)
    tied_set = set()
    for a, b in tied_pairs:
        tied_set.add((a, b))
        tied_set.add((b, a))
    
    # Create index lookup
    index_map = {name: idx for idx, name in enumerate(global_order)}
    
    pairs = []
    n = len(global_order)
    
    # Sample pairs: higher index wins over lower index
    # Use stratified sampling to get pairs across the ranking spectrum
    for i in range(n):
        for j in range(i + 1, n):
            loser = global_order[i]
            winner = global_order[j]
            
            # Skip tied pairs
            if (winner, loser) in tied_set:
                continue
            
            pairs.append((winner, loser))
    
    logging.info(f"Created {len(pairs)} preference pairs from {n} rollouts")
    
    # Limit pairs if specified
    if max_pairs and len(pairs) > max_pairs:
        random.shuffle(pairs)
        pairs = pairs[:max_pairs]
        logging.info(f"Limited to {max_pairs} pairs")
    
    return pairs


def bradley_terry_loss(r_winner: torch.Tensor, r_loser: torch.Tensor) -> torch.Tensor:
    """
    Bradley-Terry loss for preference learning.
    
    L = -log(sigmoid(r_winner - r_loser))
    
    Args:
        r_winner: Trajectory reward for preferred rollout (batch,)
        r_loser: Trajectory reward for non-preferred rollout (batch,)
    
    Returns:
        loss: Scalar loss
    """
    return -torch.log(torch.sigmoid(r_winner - r_loser) + 1e-8).mean()


def compute_trajectory_reward(model: RewardNetwork, obs: torch.Tensor) -> torch.Tensor:
    """
    Compute trajectory reward as sum of per-timestep rewards.
    
    Args:
        model: RewardNetwork
        obs: (T, obs_dim) observations for one trajectory
    
    Returns:
        reward: Scalar trajectory reward
    """
    per_step_rewards = model(obs)  # (T,)
    return per_step_rewards.sum()


def train():
    """Main training function."""
    random.seed(SEED)
    torch.manual_seed(SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    logging.info(f"Checkpoints will be saved to: {CHECKPOINT_DIR}")
    
    # Load rankings
    global_order, tied_pairs = load_rankings(DATA_FOLDER)
    
    # Create preference pairs
    pairs = create_preference_pairs(global_order, tied_pairs, MAX_PAIRS)
    
    if not pairs:
        logging.error("No preference pairs created!")
        return
    
    # Shuffle and split into train/val
    random.shuffle(pairs)
    val_size = int(len(pairs) * VALIDATION_SPLIT)
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]
    
    logging.info(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
    
    # Load first rollout to determine obs_dim
    first_file = os.path.join(DATA_FOLDER, global_order[0])
    sample_obs = load_rollout(first_file)
    obs_dim = sample_obs.shape[1]
    logging.info(f"Observation dimension: {obs_dim}")
    
    # Preload all rollouts into memory
    logging.info("Loading rollouts into memory...")
    rollout_cache: Dict[str, torch.Tensor] = {}
    loaded_count = 0
    for filename in global_order:
        filepath = os.path.join(DATA_FOLDER, filename)
        try:
            rollout_cache[filename] = load_rollout(filepath).to(device)
            loaded_count += 1
        except Exception as e:
            logging.warning(f"Failed to load {filename}: {e}")
    
    logging.info(f"Loaded {loaded_count}/{len(global_order)} rollouts")
    
    # Filter pairs to only include loaded rollouts
    train_pairs = [(w, l) for w, l in train_pairs if w in rollout_cache and l in rollout_cache]
    val_pairs = [(w, l) for w, l in val_pairs if w in rollout_cache and l in rollout_cache]
    
    logging.info(f"After filtering - Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    # Create model
    model = RewardNetwork(obs_dim, HIDDEN_LAYERS, ACTIVATION).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    logging.info(f"Model architecture: {obs_dim} -> {HIDDEN_LAYERS} -> 1")
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 10
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        random.shuffle(train_pairs)
        
        train_losses = []
        train_correct = 0
        train_total = 0
        
        for i in range(0, len(train_pairs), BATCH_SIZE):
            batch_pairs = train_pairs[i:i + BATCH_SIZE]
            
            batch_winner_rewards = []
            batch_loser_rewards = []
            
            for winner, loser in batch_pairs:
                r_winner = compute_trajectory_reward(model, rollout_cache[winner])
                r_loser = compute_trajectory_reward(model, rollout_cache[loser])
                batch_winner_rewards.append(r_winner)
                batch_loser_rewards.append(r_loser)
            
            r_winners = torch.stack(batch_winner_rewards)
            r_losers = torch.stack(batch_loser_rewards)
            
            loss = bradley_terry_loss(r_winners, r_losers)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_correct += (r_winners > r_losers).sum().item()
            train_total += len(batch_pairs)
        
        train_loss = sum(train_losses) / len(train_losses) if train_losses else 0
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # Validation
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for i in range(0, len(val_pairs), BATCH_SIZE):
                batch_pairs = val_pairs[i:i + BATCH_SIZE]
                
                batch_winner_rewards = []
                batch_loser_rewards = []
                
                for winner, loser in batch_pairs:
                    r_winner = compute_trajectory_reward(model, rollout_cache[winner])
                    r_loser = compute_trajectory_reward(model, rollout_cache[loser])
                    batch_winner_rewards.append(r_winner)
                    batch_loser_rewards.append(r_loser)
                
                if batch_winner_rewards:
                    r_winners = torch.stack(batch_winner_rewards)
                    r_losers = torch.stack(batch_loser_rewards)
                    
                    loss = bradley_terry_loss(r_winners, r_losers)
                    val_losses.append(loss.item())
                    val_correct += (r_winners > r_losers).sum().item()
                    val_total += len(batch_pairs)
        
        val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        logging.info(
            f"Epoch {epoch+1}/{EPOCHS} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        
        # Save checkpoint every N epochs
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pt")
            checkpoint_dict = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "obs_dim": obs_dim,
                "hidden_layers": HIDDEN_LAYERS,
                "activation": ACTIVATION,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
            torch.save(checkpoint_dict, checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model and save
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save model with metadata
    save_dict = {
        "model_state_dict": model.state_dict(),
        "obs_dim": obs_dim,
        "hidden_layers": HIDDEN_LAYERS,
        "activation": ACTIVATION,
        "best_val_loss": best_val_loss,
    }
    torch.save(save_dict, SAVE_PATH)
    logging.info(f"Model saved to {SAVE_PATH}")


def load_trained_model(path: str, device: str = "cpu") -> RewardNetwork:
    """Load a trained reward model from file."""
    checkpoint = torch.load(path, map_location=device)
    
    model = RewardNetwork(
        obs_dim=checkpoint["obs_dim"],
        hidden_layers=checkpoint["hidden_layers"],
        activation=checkpoint["activation"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model


if __name__ == "__main__":
    train()
