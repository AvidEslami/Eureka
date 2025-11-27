import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import os


def compute_oracle_reward(
    object_pos: Tensor,
    object_rot: Tensor,
    goal_pos: Tensor,
    goal_rot: Tensor,
    right_hand_pos: Tensor,
    left_hand_pos: Tensor,
    door_right_handle_pos: Tensor,
    door_left_handle_pos: Tensor,
    right_hand_ff_pos: Tensor,
    right_hand_mf_pos: Tensor,
    right_hand_rf_pos: Tensor,
    right_hand_lf_pos: Tensor,
    right_hand_th_pos: Tensor,
    left_hand_ff_pos: Tensor,
    left_hand_mf_pos: Tensor,
    left_hand_rf_pos: Tensor,
    left_hand_lf_pos: Tensor,
    left_hand_th_pos: Tensor,
    object_linvel: Tensor,
    object_angvel: Tensor,
    dof_force_tensor: Tensor
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Oracle reward function for ShadowHandDoorOpenInward.
    Based on the oracle reward: hands approach handles, then open the door.
    """
    # Finger distances to handles (sum of all 5 fingers per hand)
    right_hand_finger_dist = (
        torch.norm(door_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) +
        torch.norm(door_right_handle_pos - right_hand_mf_pos, p=2, dim=-1) +
        torch.norm(door_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) +
        torch.norm(door_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) +
        torch.norm(door_right_handle_pos - right_hand_th_pos, p=2, dim=-1)
    )
    left_hand_finger_dist = (
        torch.norm(door_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) +
        torch.norm(door_left_handle_pos - left_hand_mf_pos, p=2, dim=-1) +
        torch.norm(door_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) +
        torch.norm(door_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) +
        torch.norm(door_left_handle_pos - left_hand_th_pos, p=2, dim=-1)
    )

    # Opening reward: when both hands are close, reward handle separation (door opening)
    up_rew = torch.zeros_like(right_hand_finger_dist)
    up_rew = torch.where(
        right_hand_finger_dist < 0.5,
        torch.where(
            left_hand_finger_dist < 0.5,
            torch.abs(door_right_handle_pos[..., 1] - door_left_handle_pos[..., 1]) * 2.0,
            up_rew
        ),
        up_rew
    )

    # Total reward: base reward (2) minus finger distances plus opening bonus
    reward = 2.0 - right_hand_finger_dist - left_hand_finger_dist + up_rew

    reward_dict: Dict[str, Tensor] = {
        "right_hand_finger_dist": right_hand_finger_dist,
        "left_hand_finger_dist": left_hand_finger_dist,
        "up_rew": up_rew,
    }

    return reward, reward_dict


def load_rollout_observations(rollout_path: str):
    """
    Load rollout observations for ShadowHandDoorOpenInward task.
    Returns both structured observations (for oracle) and obs_buf (for NN).
    """
    with open(rollout_path, 'r') as f:
        f.readline()  # Skip first line (video path or score)
        f.readline()  # Skip "Object Pos:" header
        data = [line for line in f]

    # Find section indices
    object_rot_index = next(i for i, line in enumerate(data) if "Object Rot:" in line)
    goal_pos_index = next(i for i, line in enumerate(data) if "Goal Pos:" in line)
    goal_rot_index = next(i for i, line in enumerate(data) if "Goal Rot:" in line)
    door_left_handle_pos_index = next(i for i, line in enumerate(data) if "Door Left Handle Pos:" in line)
    door_right_handle_pos_index = next(i for i, line in enumerate(data) if "Door Right Handle Pos:" in line)
    left_hand_pos_index = next(i for i, line in enumerate(data) if "Left Hand Pos:" in line)
    right_hand_pos_index = next(i for i, line in enumerate(data) if "Right Hand Pos:" in line)
    right_hand_ff_pos_index = next(i for i, line in enumerate(data) if "Right Hand Ff Pos:" in line)
    right_hand_mf_pos_index = next(i for i, line in enumerate(data) if "Right Hand Mf Pos:" in line)
    right_hand_rf_pos_index = next(i for i, line in enumerate(data) if "Right Hand Rf Pos:" in line)
    right_hand_lf_pos_index = next(i for i, line in enumerate(data) if "Right Hand Lf Pos:" in line)
    right_hand_th_pos_index = next(i for i, line in enumerate(data) if "Right Hand Th Pos:" in line)
    left_hand_ff_pos_index = next(i for i, line in enumerate(data) if "Left Hand Ff Pos:" in line)
    left_hand_mf_pos_index = next(i for i, line in enumerate(data) if "Left Hand Mf Pos:" in line)
    left_hand_rf_pos_index = next(i for i, line in enumerate(data) if "Left Hand Rf Pos:" in line)
    left_hand_lf_pos_index = next(i for i, line in enumerate(data) if "Left Hand Lf Pos:" in line)
    left_hand_th_pos_index = next(i for i, line in enumerate(data) if "Left Hand Th Pos:" in line)
    actions_index = next(i for i, line in enumerate(data) if "Actions:" in line)
    obs_buf_index = next(i for i, line in enumerate(data) if "Obs buf:" in line)

    # Parse structured observations
    object_pos = [eval(data[i].strip())[0] for i in range(0, object_rot_index)]
    object_rot = [eval(data[i].strip())[0] for i in range(object_rot_index + 1, goal_pos_index)]
    goal_pos = [eval(data[i].strip())[0] for i in range(goal_pos_index + 1, goal_rot_index)]
    goal_rot = [eval(data[i].strip())[0] for i in range(goal_rot_index + 1, door_left_handle_pos_index)]
    door_left_handle_pos = [eval(data[i].strip())[0] for i in range(door_left_handle_pos_index + 1, door_right_handle_pos_index)]
    door_right_handle_pos = [eval(data[i].strip())[0] for i in range(door_right_handle_pos_index + 1, left_hand_pos_index)]
    left_hand_pos = [eval(data[i].strip())[0] for i in range(left_hand_pos_index + 1, right_hand_pos_index)]
    right_hand_pos = [eval(data[i].strip())[0] for i in range(right_hand_pos_index + 1, right_hand_ff_pos_index)]
    right_hand_ff_pos = [eval(data[i].strip())[0] for i in range(right_hand_ff_pos_index + 1, right_hand_mf_pos_index)]
    right_hand_mf_pos = [eval(data[i].strip())[0] for i in range(right_hand_mf_pos_index + 1, right_hand_rf_pos_index)]
    right_hand_rf_pos = [eval(data[i].strip())[0] for i in range(right_hand_rf_pos_index + 1, right_hand_lf_pos_index)]
    right_hand_lf_pos = [eval(data[i].strip())[0] for i in range(right_hand_lf_pos_index + 1, right_hand_th_pos_index)]
    right_hand_th_pos = [eval(data[i].strip())[0] for i in range(right_hand_th_pos_index + 1, left_hand_ff_pos_index)]
    left_hand_ff_pos = [eval(data[i].strip())[0] for i in range(left_hand_ff_pos_index + 1, left_hand_mf_pos_index)]
    left_hand_mf_pos = [eval(data[i].strip())[0] for i in range(left_hand_mf_pos_index + 1, left_hand_rf_pos_index)]
    left_hand_rf_pos = [eval(data[i].strip())[0] for i in range(left_hand_rf_pos_index + 1, left_hand_lf_pos_index)]
    left_hand_lf_pos = [eval(data[i].strip())[0] for i in range(left_hand_lf_pos_index + 1, left_hand_th_pos_index)]
    left_hand_th_pos = [eval(data[i].strip())[0] for i in range(left_hand_th_pos_index + 1, actions_index)]

    # Parse obs_buf for NN
    obs_buf = [eval(data[i].strip())[0] for i in range(obs_buf_index + 1, len(data))]

    # Determine usable length
    num_timesteps = min(
        len(object_pos), len(object_rot), len(goal_pos), len(goal_rot),
        len(door_left_handle_pos), len(door_right_handle_pos), len(left_hand_pos),
        len(right_hand_pos), len(right_hand_ff_pos), len(right_hand_mf_pos),
        len(right_hand_rf_pos), len(right_hand_lf_pos), len(right_hand_th_pos),
        len(left_hand_ff_pos), len(left_hand_mf_pos), len(left_hand_rf_pos),
        len(left_hand_lf_pos), len(left_hand_th_pos), len(obs_buf)
    )

    # Build structured observations list
    structured_obs = []
    for i in range(num_timesteps):
        obs = {
            "object_pos": torch.tensor(object_pos[i], dtype=torch.float32).unsqueeze(0),
            "object_rot": torch.tensor(object_rot[i], dtype=torch.float32).unsqueeze(0),
            "goal_pos": torch.tensor(goal_pos[i], dtype=torch.float32).unsqueeze(0),
            "goal_rot": torch.tensor(goal_rot[i], dtype=torch.float32).unsqueeze(0),
            "door_left_handle_pos": torch.tensor(door_left_handle_pos[i], dtype=torch.float32).unsqueeze(0),
            "door_right_handle_pos": torch.tensor(door_right_handle_pos[i], dtype=torch.float32).unsqueeze(0),
            "left_hand_pos": torch.tensor(left_hand_pos[i], dtype=torch.float32).unsqueeze(0),
            "right_hand_pos": torch.tensor(right_hand_pos[i], dtype=torch.float32).unsqueeze(0),
            "right_hand_ff_pos": torch.tensor(right_hand_ff_pos[i], dtype=torch.float32).unsqueeze(0),
            "right_hand_mf_pos": torch.tensor(right_hand_mf_pos[i], dtype=torch.float32).unsqueeze(0),
            "right_hand_rf_pos": torch.tensor(right_hand_rf_pos[i], dtype=torch.float32).unsqueeze(0),
            "right_hand_lf_pos": torch.tensor(right_hand_lf_pos[i], dtype=torch.float32).unsqueeze(0),
            "right_hand_th_pos": torch.tensor(right_hand_th_pos[i], dtype=torch.float32).unsqueeze(0),
            "left_hand_ff_pos": torch.tensor(left_hand_ff_pos[i], dtype=torch.float32).unsqueeze(0),
            "left_hand_mf_pos": torch.tensor(left_hand_mf_pos[i], dtype=torch.float32).unsqueeze(0),
            "left_hand_rf_pos": torch.tensor(left_hand_rf_pos[i], dtype=torch.float32).unsqueeze(0),
            "left_hand_lf_pos": torch.tensor(left_hand_lf_pos[i], dtype=torch.float32).unsqueeze(0),
            "left_hand_th_pos": torch.tensor(left_hand_th_pos[i], dtype=torch.float32).unsqueeze(0),
            "object_linvel": torch.zeros((1, 3), dtype=torch.float32),
            "object_angvel": torch.zeros((1, 3), dtype=torch.float32),
            "dof_force_tensor": torch.zeros((1, 1), dtype=torch.float32),
        }
        structured_obs.append(obs)

    # Build obs_buf tensors for NN
    obs_buf_tensors = [torch.tensor(obs_buf[i], dtype=torch.float32) for i in range(num_timesteps)]

    return structured_obs, obs_buf_tensors


def gen_net(in_size: int = 1, out_size: int = 1, H: int = 128, n_layers: int = 3, activation: str = 'tanh'):
    """Generate a feedforward network with specified architecture."""
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())
    return net


class NNRewardModel(nn.Module):
    """NN reward model that approximates the oracle reward.
    
    Architecture matches reward_tuner.py: obs_dim -> 768 -> 384 -> 1
    """
    def __init__(self, obs_dim: int = 417, scale: float = 10.0):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 768),
            nn.LeakyReLU(0.1),
            nn.Linear(768, 384),
            nn.LeakyReLU(0.1),
            nn.Linear(384, 1),
            nn.Tanh()
        )

    def forward(self, x: Tensor) -> Tensor:
        return (self.net(x) * self.scale).squeeze(-1)


def load_nn_reward_model(checkpoint_path: str, obs_dim: int = 417, scale: float = 1.0):
    """Load the trained NN reward model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if scale is None:
        scale = checkpoint.get('scale', 1.0)
    model = NNRewardModel(obs_dim, scale=scale)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def load_dataset_from_folder(data_folder: str, task: str = "ShadowHandDoorOpenInward") -> Tuple[Tensor, Tensor]:
    """
    Load all rollouts from a folder and compute oracle rewards.
    Returns (obs_buf_all, oracle_rewards_all) as tensors.
    """
    filenames = [f for f in os.listdir(data_folder) if f.endswith(".txt") and task in f]
    
    all_obs_buf = []
    all_oracle_rewards = []
    
    for filename in filenames:
        filepath = os.path.join(data_folder, filename)
        try:
            structured_obs, obs_buf_tensors = load_rollout_observations(filepath)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue
        
        # Compute oracle rewards for each timestep
        for i in range(len(structured_obs)):
            oracle_rew, _ = compute_oracle_reward(**structured_obs[i])
            all_obs_buf.append(obs_buf_tensors[i])
            all_oracle_rewards.append(oracle_rew.squeeze())
    
    if not all_obs_buf:
        raise ValueError(f"No valid rollouts found in {data_folder}")
    
    obs_buf_tensor = torch.stack(all_obs_buf)
    oracle_rewards_tensor = torch.stack(all_oracle_rewards)
    
    print(f"Loaded {len(filenames)} rollouts, {len(all_obs_buf)} total timesteps")
    print(f"Oracle reward range: [{oracle_rewards_tensor.min().item():.3f}, {oracle_rewards_tensor.max().item():.3f}]")
    print(f"Oracle reward mean: {oracle_rewards_tensor.mean().item():.3f}, std: {oracle_rewards_tensor.std().item():.3f}")
    
    return obs_buf_tensor, oracle_rewards_tensor


def train_nn_to_approximate_oracle(
    data_folder: str,
    task: str = "ShadowHandDoorOpenInward",
    obs_dim: int = 417,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.2,
    save_path: str = None,
    live_plot: bool = True,
    scale: float = 1.0,
    H: int = 256,
    n_layers: int = 3,
) -> NNRewardModel:
    """
    Train NN reward model to approximate oracle reward using MSE loss.
    
    Args:
        data_folder: Folder containing rollout .txt files
        task: Task name to filter files
        obs_dim: Observation dimension
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        val_split: Fraction of data for validation
        save_path: Path to save the trained model (if None, saves to data_folder)
        live_plot: Show live training plot
        scale: Output scale for the NN (if None, auto-computed from oracle reward range)
    
    Returns:
        Trained NNRewardModel
    """
    # Load dataset
    obs_buf, oracle_rewards = load_dataset_from_folder(data_folder, task)
    
    # Auto-compute scale based on oracle reward range (Tanh outputs [-1, 1])
    if scale is None:
        reward_abs_max = max(abs(oracle_rewards.min().item()), abs(oracle_rewards.max().item()))
        scale = reward_abs_max * 1.1  # Add 10% margin
        print(f"Auto-computed scale: {scale:.3f} (oracle reward abs max: {reward_abs_max:.3f})")
    
    # Train/val split
    n_samples = len(obs_buf)
    n_val = int(n_samples * val_split)
    indices = torch.randperm(n_samples)
    
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_obs = obs_buf[train_indices]
    train_rewards = oracle_rewards[train_indices]
    val_obs = obs_buf[val_indices]
    val_rewards = oracle_rewards[val_indices]
    
    print(f"Train/Val split: {int((1-val_split)*100)}/{int(val_split*100)}")
    print(f"Train samples: {len(train_obs)}, Val samples: {len(val_obs)}")
    
    # Model, optimizer, loss
    model = NNRewardModel(obs_dim, scale=scale, H=H, n_layers=n_layers)
    print(f"Model: {n_layers} layers, hidden size {H}, scale {scale}")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # Tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_state = None
    
    # Live plot setup
    if live_plot:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Training NN to Approximate Oracle Reward')
        line_train, = ax1.plot([], [], 'b-', label='Train Loss')
        line_val, = ax1.plot([], [], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.legend()
        ax1.set_title('Loss')
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        
        # Shuffle training data
        perm = torch.randperm(len(train_obs))
        train_obs_shuffled = train_obs[perm]
        train_rewards_shuffled = train_rewards[perm]
        
        # Mini-batch training
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_obs), batch_size):
            batch_obs = train_obs_shuffled[i:i+batch_size]
            batch_rewards = train_rewards_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            pred_rewards = model(batch_obs)
            loss = loss_fn(pred_rewards, batch_rewards)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_obs)
            val_loss = loss_fn(val_pred, val_rewards).item()
        val_losses.append(val_loss)
        
        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
        
        # Print every epoch
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Update live plot
        if live_plot and (epoch + 1) % 5 == 0:
            epochs_so_far = list(range(1, len(train_losses) + 1))
            line_train.set_data(epochs_so_far, train_losses)
            line_val.set_data(epochs_so_far, val_losses)
            ax1.relim()
            ax1.autoscale_view()
            
            # Scatter plot of predictions vs oracle on validation set
            ax2.clear()
            with torch.no_grad():
                val_pred_plot = model(val_obs).numpy()
            ax2.scatter(val_rewards.numpy(), val_pred_plot, alpha=0.3, s=5)
            ax2.plot([val_rewards.min(), val_rewards.max()], 
                     [val_rewards.min(), val_rewards.max()], 'r--', label='y=x')
            ax2.set_xlabel('Oracle Reward')
            ax2.set_ylabel('NN Predicted Reward')
            ax2.set_title(f'Validation: Pred vs Oracle (Epoch {epoch+1})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best model with val loss: {best_val_loss:.6f}")
    
    # Save model
    if save_path is None:
        save_path = os.path.join(data_folder, f"{task}_oracle_approx_nn.pth")
    
    # Create output directory if needed
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'obs_dim': obs_dim,
        'scale': scale,
        'H': H,
        'n_layers': n_layers,
    }, save_path)
    print(f"Saved model to {save_path}")
    
    # Save final plot
    if live_plot:
        plot_path = save_path.replace('.pth', '_training.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved training plot to {plot_path}")
        plt.ioff()
        plt.close(fig)
    
    return model


def compare_rewards(rollout_path: str, nn_checkpoint_path: str, save_dir: str = None):
    """
    Compare oracle reward vs NN reward on a rollout.
    
    Args:
        rollout_path: Path to the rollout file
        nn_checkpoint_path: Path to the NN reward model checkpoint
        save_dir: Directory to save figures (if None, just displays)
    
    Returns:
        Tuple of (fig1, fig2) - the two matplotlib figures
    """
    # Load observations
    structured_obs, obs_buf_tensors = load_rollout_observations(rollout_path)
    num_timesteps = len(structured_obs)

    # Load NN model
    nn_model = load_nn_reward_model(nn_checkpoint_path, obs_dim=417)

    # Compute rewards
    oracle_rewards = []
    nn_rewards = []

    with torch.no_grad():
        for i in range(num_timesteps):
            # Oracle reward
            oracle_rew, _ = compute_oracle_reward(**structured_obs[i])
            oracle_rewards.append(oracle_rew.squeeze().item())

            # NN reward
            nn_rew = nn_model(obs_buf_tensors[i].unsqueeze(0))*3.0 + 2
            nn_rewards.append(nn_rew.item())

    oracle_rewards = torch.tensor(oracle_rewards)
    nn_rewards = torch.tensor(nn_rewards)

    # Cumulative rewards
    oracle_cumulative = torch.cumsum(oracle_rewards, dim=0)
    nn_cumulative = torch.cumsum(nn_rewards, dim=0)

    timesteps = list(range(num_timesteps))

    # Figure 1: Per-timestep rewards
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(timesteps, oracle_rewards.numpy(), label='Oracle Reward', color='blue', linewidth=1.5)
    ax1.plot(timesteps, nn_rewards.numpy(), label='NN Reward', color='red', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Reward')
    ax1.set_title('Per-Timestep Reward: Oracle vs NN')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Figure 2: Cumulative rewards
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(timesteps, oracle_cumulative.numpy(), label='Oracle Cumulative', color='blue', linewidth=1.5)
    ax2.plot(timesteps, nn_cumulative.numpy(), label='NN Cumulative', color='red', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title('Cumulative Reward: Oracle vs NN')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Save or show
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        rollout_name = os.path.basename(rollout_path).replace('.txt', '')
        fig1.savefig(os.path.join(save_dir, f'{rollout_name}_per_timestep.png'), dpi=150, bbox_inches='tight')
        fig2.savefig(os.path.join(save_dir, f'{rollout_name}_cumulative.png'), dpi=150, bbox_inches='tight')
        print(f"Saved figures to {save_dir}")

    # Print summary
    print(f"\nRollout: {rollout_path}")
    print(f"Timesteps: {num_timesteps}")
    print(f"Oracle - Total: {oracle_cumulative[-1].item():.2f}, Mean: {oracle_rewards.mean().item():.4f}")
    print(f"NN     - Total: {nn_cumulative[-1].item():.2f}, Mean: {nn_rewards.mean().item():.4f}")

    return fig1, fig2


if __name__ == "__main__":
    # ==================== CONFIGURATION ====================
    # Task
    task = "ShadowHandDoorOpenInward"
    
    # Data folder containing rollout .txt files
    data_folder = "./auto_preference_data"
    
    # Output folder for checkpoints and plots
    output_folder = "./oracle_approx_checkpoints"
    
    # Training hyperparameters
    epochs = 100
    lr = 1e-3
    batch_size = 256
    scale = 10.0  # Output scale (set to None for auto-compute from data)
    H = 256       # Hidden layer size
    n_layers = 3  # Number of hidden layers
    
    # ==================== TRAIN ====================
    # model = train_nn_to_approximate_oracle(
    #     task=task,
    #     data_folder=data_folder,
    #     epochs=epochs,
    #     lr=lr,
    #     batch_size=batch_size,
    #     scale=scale,
    #     H=H,
    #     n_layers=n_layers,
    #     save_path=os.path.join(output_folder, f"{task}_oracle_approx_nn.pth"),
    #     live_plot=True,
    # )
    # print("Training complete.")
    
    # exit()
    # ==================== COMPARE (optional) ====================
    # Uncomment below to compare oracle vs NN on specific rollouts
    #
    checkpoint_path = os.path.join(output_folder, f"{task}_oracle_approx_nn.pth")
    checkpoint_path = "/home/gx22/Desktop/isaacgym/python/Eureka/eureka/auto_preference_data/runs/20_scaling/ShadowHandDoorOpenInward_nn_checkpoint_epoch10.pth"
    rollout_success = "/home/gx22/Desktop/isaacgym/python/Eureka/eureka/auto_preference_data/42_ShadowHandDoorOpenInward_2025-11-27_13-29-09.txt"
    rollout_fail = "/home/gx22/Desktop/isaacgym/python/Eureka/eureka/auto_preference_data/42_ShadowHandDoorOpenInward_2025-11-27_12-43-12.txt"
    # 
    fig1, fig2 = compare_rewards(rollout_success, checkpoint_path, save_dir=output_folder)
    fig3, fig4 = compare_rewards(rollout_fail, checkpoint_path, save_dir=output_folder)
    plt.show()
