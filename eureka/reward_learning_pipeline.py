"""
Iterative Reward Learning Pipeline

This script orchestrates:
1. Train a reward model from rollout data (ground truth or preference-based)
2. Export the model to TorchScript for inference
3. Run RL training using the learned reward model
4. Collect rollouts from policy checkpoints
5. Move collected data to training folder
6. Repeat for N iterations

Reward Training Modes:
- ground_truth: Train using MSE loss against ground truth reward function
- preference: Train using Bradley-Terry loss from VLM preference rankings

Output Structure:
    experiments/
        {task}_{timestamp}/
            config.json              # Run configuration
            iteration_1/
                reward_model.pth         # Model state dict
                reward_model_full.pth    # Full model
                reward_model.pt          # TorchScript model
                rl_training_log.txt      # Training log
                policy/                  # Trained policy checkpoints
                rollouts/                # Collected rollout files
            iteration_2/
                ...
            iteration_3/
                ...

Supported tasks:
- ShadowHandDoorOpenOutward
- ShadowHandDoorOpenInward
- ShadowHandBottleCap (future)
"""

import os
import sys
import re
import signal
import shutil
import subprocess
import datetime
import argparse
import random
import json
import ast
import time
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Paths
EUREKA_DIR = Path(__file__).parent.resolve()
ISAAC_ROOT_DIR = EUREKA_DIR.parent / "isaacgymenvs" / "isaacgymenvs"
EXPERIMENTS_DIR = EUREKA_DIR / "experiments"

sys.path.insert(0, str(EUREKA_DIR))

DATA_SOURCE_DIR = EUREKA_DIR / "auto_preference_data"

# Global path for reward model (where Isaac Gym tasks look for it)
GLOBAL_REWARD_MODEL_PATH = EUREKA_DIR / "reward_model.pt"

MAX_ROLLOUT_LENGTH = 1000000


# =============================================================================
# TASK CONFIGURATIONS
# =============================================================================

# Input keys for each task's ground truth model
TASK_INPUT_KEYS = {
    "ShadowHandDoorOpenOutward": [
        "door_right_handle_pos", "right_hand_ff_pos", "right_hand_mf_pos",
        "right_hand_rf_pos", "right_hand_lf_pos", "right_hand_th_pos",
        "door_left_handle_pos", "left_hand_ff_pos", "left_hand_mf_pos",
        "left_hand_rf_pos", "left_hand_lf_pos", "left_hand_th_pos"
    ],
    "ShadowHandDoorOpenInward": [
        "door_right_handle_pos", "right_hand_ff_pos", "right_hand_mf_pos",
        "right_hand_rf_pos", "right_hand_lf_pos", "right_hand_th_pos",
        "door_left_handle_pos", "left_hand_ff_pos", "left_hand_mf_pos",
        "left_hand_rf_pos", "left_hand_lf_pos", "left_hand_th_pos"
    ],
}

# Data folder for each task
TASK_DATA_FOLDERS = {
    "ShadowHandDoorOpenOutward": EUREKA_DIR / "auto_preference_data_open_outward",
    "ShadowHandDoorOpenInward": EUREKA_DIR / "auto_preference_data_open_inward",
}

# Files to ignore when loading rollout data (not rollout files)
IGNORED_FILES = {"preference_rankings.txt", "ranking_results.json"}

# Conda environments
EUREKA_CONDA_ENV = "eureka"     # Isaac Gym / RL training / rollout collection

# VLM configuration for preference labeling
VLM_CONDA_ENV = "vlm"
VLM_SCRIPT_PATH = EUREKA_DIR / "utils" / "vlm.py"
VLM_OUTPUT_PATH = EUREKA_DIR / "utils" / "vlm_response.txt"
VLM_TIMEOUT = 60  # seconds

# Task descriptions for VLM
TASK_DESCRIPTIONS = {
    "ShadowHandDoorOpenOutward": "Open the door using the two robotic hands, the door handles must first be grabbed, then pushed outwards in order to be opened.",
    "ShadowHandDoorOpenInward": "Open the door using the two robotic hands, the door handles must first be grabbed, then pulled inwards in order to be opened.",
}

# Preference model architecture
PREF_HIDDEN_LAYERS = [256, 128, 64]
PREF_ACTIVATION = "relu"
PREF_WEIGHT_DECAY = 1e-5
PREF_VALIDATION_SPLIT = 0.2
PREF_PATIENCE = 10


# =============================================================================
# GROUND TRUTH REWARD FUNCTIONS
# =============================================================================

def ground_truth_door_open(door_right_handle_pos, right_hand_ff_pos, right_hand_mf_pos,
                           right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos,
                           door_left_handle_pos, left_hand_ff_pos, left_hand_mf_pos,
                           left_hand_rf_pos, left_hand_lf_pos, left_hand_th_pos):
    """
    Ground truth reward function for door opening tasks (both inward and outward).
    The reward encourages:
    1. Moving fingers close to door handles
    2. Opening the door (increasing distance between handles) when fingers are close
    """
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

    right_hand_dist_rew = right_hand_finger_dist
    left_hand_dist_rew = left_hand_finger_dist

    up_rew = torch.zeros_like(right_hand_dist_rew)
    up_rew = torch.where(
        right_hand_finger_dist < 0.5,
        torch.where(
            left_hand_finger_dist < 0.5,
            torch.abs(door_right_handle_pos[:, 1] - door_left_handle_pos[:, 1]) * 2,
            up_rew
        ),
        up_rew
    )

    reward = 2 - right_hand_dist_rew - left_hand_dist_rew + up_rew
    return reward


TASK_GROUND_TRUTH_FUNCTIONS = {
    "ShadowHandDoorOpenOutward": ground_truth_door_open,
    "ShadowHandDoorOpenInward": ground_truth_door_open,
}


def get_ground_truth_reward(task, gt_input):
    """Compute ground truth reward for a given task and input."""
    if task not in TASK_GROUND_TRUTH_FUNCTIONS:
        raise ValueError(f"Unknown task: {task}. Supported tasks: {list(TASK_GROUND_TRUTH_FUNCTIONS.keys())}")
    
    gt_func = TASK_GROUND_TRUTH_FUNCTIONS[task]
    input_keys = TASK_INPUT_KEYS[task]
    args = [gt_input[key] for key in input_keys]
    return gt_func(*args)


class NNRewardModel(nn.Module):
    """Neural network reward model"""
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, input_tensor):
        return self.net(input_tensor)


def get_rollout_observations(rollout_path, task, required_keys, nn=False):
    """Parse rollout data file and extract observations."""
    with open(rollout_path, 'r') as f:
        f.readline()
        f.readline()
        data = [line for line in f]

    if task in ("ShadowHandDoorOpenOutward", "ShadowHandDoorOpenInward"):
        return _parse_door_open_observations(data, required_keys, nn)
    else:
        raise ValueError(f"Unknown task data format: {task}")


def _parse_door_open_observations(data, required_keys, nn):
    """Parse observations for door opening tasks (both inward and outward)"""
    door_left_handle_pos_index = next(i for i, line in enumerate(data) if "Door Left Handle Pos:" in line)
    door_right_handle_pos_index = next(i for i, line in enumerate(data) if "Door Right Handle Pos:" in line)
    left_hand_pos_index = next(i for i, line in enumerate(data) if "Left Hand Pos:" in line)
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

    if not nn:
        door_left_handle_pos = [eval(data[i].strip())[0] for i in range(door_left_handle_pos_index + 1, door_right_handle_pos_index)]
        door_right_handle_pos = [eval(data[i].strip())[0] for i in range(door_right_handle_pos_index + 1, left_hand_pos_index)]
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

        input_dicts = []
        usable_length = min(
            MAX_ROLLOUT_LENGTH,
            len(door_left_handle_pos), len(door_right_handle_pos),
            len(right_hand_ff_pos), len(right_hand_mf_pos),
            len(right_hand_rf_pos), len(right_hand_lf_pos), len(right_hand_th_pos),
            len(left_hand_ff_pos), len(left_hand_mf_pos), len(left_hand_rf_pos),
            len(left_hand_lf_pos), len(left_hand_th_pos)
        )
        for i in range(usable_length):
            full_vars = {
                "door_left_handle_pos": torch.tensor(door_left_handle_pos[i], dtype=torch.float32).unsqueeze(0),
                "door_right_handle_pos": torch.tensor(door_right_handle_pos[i], dtype=torch.float32).unsqueeze(0),
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
            }
            filtered_vars = {k: full_vars[k] for k in required_keys}
            input_dicts.append(filtered_vars)
    else:
        obs_buf = [eval(data[i].strip())[0] for i in range(obs_buf_index + 1, len(data))]
        input_dicts = []
        for i in range(len(obs_buf)):
            obs_buf_tensor = torch.tensor(obs_buf[i], dtype=torch.float32).unsqueeze(0)
            input_dicts.append({"obs_buf": obs_buf_tensor})

    return input_dicts


def loss_func(data_batch, nn_model, task):
    """Compute MSE loss between ground truth and NN reward"""
    total_loss = 0.0
    for gt_input, nn_input in data_batch:
        gt_rew = get_ground_truth_reward(task, gt_input)
        nn_rew = nn_model(nn_input["obs_buf"])
        total_loss = total_loss + F.mse_loss(nn_rew.squeeze(), gt_rew.squeeze())
    return total_loss / len(data_batch)


def train_reward_model(task, data_folder, iteration_dir, iteration_num, tensorboard_dir,
                       batch_size=64, num_epochs=40, lr=1e-4, validation_size=1024):
    """Train the reward model from rollout data"""
    print("\n" + "="*60)
    print(f"STEP 1: Training Reward Model for {task}")
    print("="*60)

    if task not in TASK_INPUT_KEYS:
        print(f"ERROR: Unknown task {task}. Supported: {list(TASK_INPUT_KEYS.keys())}")
        return None
    
    input_keys = TASK_INPUT_KEYS[task]

    filenames = [f for f in os.listdir(data_folder) 
                 if f.endswith(".txt") and f not in IGNORED_FILES]
    if not filenames:
        print(f"ERROR: No data files found in {data_folder}")
        return None

    print(f"Loading {len(filenames)} rollout files from {data_folder}")

    pairwise_data = []
    for filename in filenames:
        filepath = os.path.join(data_folder, filename)
        try:
            gt_obs = get_rollout_observations(filepath, task, input_keys, nn=False)
            nn_obs = get_rollout_observations(filepath, task, input_keys, nn=True)
            for row in range(len(gt_obs)):
                pairwise_data.append((gt_obs[row], nn_obs[row]))
        except Exception as e:
            print(f"  Warning: Failed to load {filename}: {e}")
            continue

    print(f"Loaded {len(pairwise_data)} data points")

    if len(pairwise_data) < validation_size + batch_size:
        print("ERROR: Not enough data for training and validation")
        return None

    obs_dim = pairwise_data[0][1]["obs_buf"].shape[-1]
    print(f"Observation dimension: {obs_dim}")

    nn_model = NNRewardModel(obs_dim)

    random.shuffle(pairwise_data)
    validation_batch = pairwise_data[:validation_size]
    train_data = pairwise_data[validation_size:]

    optimizer = optim.Adam(nn_model.parameters(), lr=lr)
    best_validation_loss = float('inf')
    best_model_state = None

    print(f"Training with {len(train_data)} samples, validating with {len(validation_batch)} samples")
    initial_val_loss = loss_func(validation_batch, nn_model, task).item()
    print(f"Initial Validation Loss: {initial_val_loss:.4f}")

    # TensorBoard writer for this iteration
    writer = SummaryWriter(log_dir=str(tensorboard_dir / f"iteration_{iteration_num}"))

    # Training log
    training_log = {
        "epochs": [],
        "train_losses": [],
        "val_losses": [],
        "best_val_loss": None,
        "best_epoch": None
    }

    for epoch in range(num_epochs):
        total_loss = 0.0
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            optimizer.zero_grad()
            loss = loss_func(batch, nn_model, task)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_data) / batch_size)
        validation_loss = loss_func(validation_batch, nn_model, task).item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.5f}, Validation Loss: {validation_loss:.5f}")

        # Log to TensorBoard
        writer.add_scalar("Loss/train", avg_loss, epoch + 1)
        writer.add_scalar("Loss/validation", validation_loss, epoch + 1)

        training_log["epochs"].append(epoch + 1)
        training_log["train_losses"].append(avg_loss)
        training_log["val_losses"].append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model_state = nn_model.state_dict().copy()
            training_log["best_val_loss"] = best_validation_loss
            training_log["best_epoch"] = epoch + 1
            print(f"  New Best: {best_validation_loss:.5f}")

    # Log best validation loss as a summary scalar
    writer.add_scalar("Summary/best_val_loss", best_validation_loss, iteration_num)
    writer.close()

    # Load best model state
    nn_model.load_state_dict(best_model_state)

    # Save models to iteration directory
    model_state_path = iteration_dir / "reward_model.pth"
    model_full_path = iteration_dir / "reward_model_full.pth"
    model_jit_path = iteration_dir / "reward_model.pt"

    torch.save(best_model_state, str(model_state_path))
    torch.save(nn_model, str(model_full_path))

    # Export to TorchScript
    nn_model.eval()
    ts_model = torch.jit.script(nn_model)
    ts_model.save(str(model_jit_path))

    # Copy to global location for Isaac Gym tasks
    shutil.copy(str(model_jit_path), str(GLOBAL_REWARD_MODEL_PATH))

    # Save training log
    with open(iteration_dir / "reward_model_training_log.json", 'w') as f:
        json.dump(training_log, f, indent=2)

    print(f"Training complete. Best validation loss: {best_validation_loss:.5f}")
    print(f"Models saved to: {iteration_dir}")
    print(f"Global reward model updated: {GLOBAL_REWARD_MODEL_PATH}")

    return nn_model


# =============================================================================
# PREFERENCE-BASED REWARD LEARNING
# =============================================================================

def get_activation(name: str) -> nn.Module:
    """Return activation module by name."""
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
    }
    return activations.get(name.lower(), nn.ReLU())


class PreferenceRewardNetwork(nn.Module):
    """MLP that maps observation -> scalar reward for preference learning."""
    
    def __init__(self, obs_dim: int, hidden_layers: List[int], activation: str = "relu"):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(get_activation(activation))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def load_rankings(data_folder: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Load rankings from ranking_results.json."""
    rankings_path = os.path.join(data_folder, "ranking_results.json")
    
    if not os.path.exists(rankings_path):
        return [], []
    
    with open(rankings_path, 'r') as f:
        data = json.load(f)
    
    global_order = data.get("global_order", [])
    tied_pairs = [tuple(pair) for pair in data.get("tied_pairs", [])]
    
    print(f"Loaded {len(global_order)} ranked rollouts, {len(tied_pairs)} tied pairs")
    return global_order, tied_pairs


def save_rankings(data_folder: str, global_order: List[str], tied_pairs: List[Tuple[str, str]]):
    """Save rankings to ranking_results.json."""
    rankings_path = os.path.join(data_folder, "ranking_results.json")
    
    data = {
        "global_order": global_order,
        "tied_pairs": [list(pair) for pair in tied_pairs],
    }
    
    with open(rankings_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved rankings to {rankings_path}")


def load_rollout_obs(filepath: str) -> torch.Tensor:
    """Parse obs_buf section from rollout file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    obs_marker = "Obs buf:\n"
    obs_start = content.find(obs_marker)
    
    if obs_start == -1:
        raise ValueError(f"No 'Obs buf:' section found in {filepath}")
    
    obs_start += len(obs_marker)
    obs_section = content[obs_start:]
    
    observations = []
    for line in obs_section.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
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


def get_rollout_metadata(filepath: str) -> Tuple[float, int, str]:
    """
    Get rollout score, length, and video path from rollout file.
    
    Returns:
        score: Rollout score (0 or 1 typically)
        length: Number of timesteps
        video_path: Path to video if present, else empty string
    """
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        
        if first_line.startswith("/"):
            video_path = first_line
            score = 0.0
        else:
            video_path = ""
            try:
                score = float(first_line)
            except ValueError:
                score = 0.0
        
        length = sum(1 for _ in f)
    
    return score, length, video_path


def vlm_compare(task: str, video_path_a: str, video_path_b: str) -> int:
    """
    Query VLM to compare two rollout videos.
    
    Returns:
        1: video_a is better
        2: video_b is better
        0: tie
        5: VLM failed
    """
    print(f"Querying VLM for preference between {video_path_a} and {video_path_b}")
    
    if VLM_OUTPUT_PATH.exists():
        os.remove(VLM_OUTPUT_PATH)
    
    try:
        subprocess.run(
            ["conda", "run", "-n", VLM_CONDA_ENV, "python", str(VLM_SCRIPT_PATH), task, video_path_a, video_path_b],
            check=True,
            cwd=str(EUREKA_DIR),
        )
    except Exception as e:
        print(f"Error running VLM query: {e}")
        return 5
    
    start_time = time.time()
    while time.time() - start_time < VLM_TIMEOUT:
        if VLM_OUTPUT_PATH.exists():
            with open(VLM_OUTPUT_PATH, 'r') as f:
                vlm_output = f.read().strip()
            try:
                return int(float(vlm_output))
            except ValueError:
                print(f"Error parsing VLM output: {vlm_output}")
                return 5
        time.sleep(0.5)
    
    print("VLM query timed out")
    return 5


def label_unlabeled_rollouts(task: str, data_folder: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Label unlabeled rollouts using VLM-based binary insertion.
    
    For each unlabeled rollout:
    1. Compare scores first (higher is better)
    2. If scores equal, compare lengths (success prefers shorter, failure prefers longer)
    3. If exact tie, use VLM to compare videos
    
    Returns:
        global_order: Updated global ranking (worst to best)
        tied_pairs: List of tied pair tuples
    """
    print("\n" + "="*60)
    print(f"Labeling Unlabeled Rollouts for {task}")
    print("="*60)
    
    global_order, tied_pairs = load_rankings(data_folder)
    
    filenames = [f for f in os.listdir(data_folder) 
                 if f.endswith(".txt") and f not in IGNORED_FILES]
    
    if not filenames:
        print(f"No rollout files found in {data_folder}")
        return global_order, tied_pairs
    
    rollout_scores = {}
    rollout_lengths = {}
    video_paths = {}
    
    for filename in filenames:
        filepath = os.path.join(data_folder, filename)
        try:
            score, length, video_path = get_rollout_metadata(filepath)
            rollout_scores[filename] = score
            rollout_lengths[filename] = length
            if video_path:
                video_paths[filename] = video_path
        except Exception as e:
            print(f"  Warning: Failed to read metadata for {filename}: {e}")
            continue
    
    tied_set = set()
    for a, b in tied_pairs:
        tied_set.add(a)
        tied_set.add(b)
    
    unlabeled = [f for f in filenames if f not in global_order and f not in tied_set]
    print(f"Found {len(unlabeled)} unlabeled rollouts out of {len(filenames)} total")
    
    def compare_rollouts(name_a: str, name_b: str) -> int:
        """Compare two rollouts. Returns: 1 if a>b, -1 if a<b, 0 if tie."""
        score_a = rollout_scores.get(name_a, 0)
        score_b = rollout_scores.get(name_b, 0)
        
        if score_a != score_b:
            return 1 if score_a > score_b else -1
        
        if name_a in video_paths and name_b in video_paths:
            vlm_result = vlm_compare(task, video_paths[name_a], video_paths[name_b])
            if vlm_result == 1:
                return 1
            elif vlm_result == 2:
                return -1
            elif vlm_result == 0:
                return 0
        
        return 0
    
    for name in unlabeled:
        lo, hi = 0, len(global_order)
        
        while lo < hi:
            mid = (lo + hi) // 2
            b_name = global_order[mid]
            
            cmp_result = compare_rollouts(name, b_name)
            
            if cmp_result == 0:
                tied_pairs.append((name, b_name))
                print(f"  {name} tied with {b_name}")
                break
            elif cmp_result > 0:
                lo = mid + 1
            else:
                hi = mid
        else:
            global_order.insert(lo, name)
            print(f"  Inserted {name} at position {lo}/{len(global_order)}")
        
        save_rankings(data_folder, global_order, tied_pairs)
    
    print(f"Final ranking: {len(global_order)} rollouts, {len(tied_pairs)} tied pairs")
    return global_order, tied_pairs


def create_preference_pairs(
    global_order: List[str],
    tied_pairs: List[Tuple[str, str]],
    max_pairs: int = None
) -> List[Tuple[str, str]]:
    """
    Create preference pairs from global ordering.
    Higher index in global_order = better rollout.
    """
    tied_set = set()
    for a, b in tied_pairs:
        tied_set.add((a, b))
        tied_set.add((b, a))
    
    pairs = []
    n = len(global_order)
    
    for i in range(n):
        for j in range(i + 1, n):
            loser = global_order[i]
            winner = global_order[j]
            
            if (winner, loser) in tied_set:
                continue
            
            pairs.append((winner, loser))
    
    print(f"Created {len(pairs)} preference pairs from {n} rollouts")
    
    if max_pairs and len(pairs) > max_pairs:
        random.shuffle(pairs)
        pairs = pairs[:max_pairs]
        print(f"Limited to {max_pairs} pairs")
    
    return pairs


def bradley_terry_loss(r_winner: torch.Tensor, r_loser: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry loss: -log(sigmoid(r_winner - r_loser))"""
    return -torch.log(torch.sigmoid(r_winner - r_loser) + 1e-8).mean()


def compute_trajectory_reward(model: PreferenceRewardNetwork, obs: torch.Tensor) -> torch.Tensor:
    """Compute trajectory reward as sum of per-timestep rewards."""
    per_step_rewards = model(obs)
    return per_step_rewards.sum()


def train_preference_reward_model(task, data_folder, iteration_dir, iteration_num, tensorboard_dir,
                                   batch_size=32, num_epochs=100, lr=1e-4):
    """Train reward model using Bradley-Terry loss from preference rankings."""
    print("\n" + "="*60)
    print(f"STEP 1: Training Preference Reward Model for {task}")
    print("="*60)
    
    global_order, tied_pairs = label_unlabeled_rollouts(task, str(data_folder))
    
    if len(global_order) < 2:
        print("ERROR: Need at least 2 ranked rollouts for preference learning")
        return None
    
    pairs = create_preference_pairs(global_order, tied_pairs)
    
    if not pairs:
        print("ERROR: No preference pairs created!")
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    first_file = os.path.join(data_folder, global_order[0])
    sample_obs = load_rollout_obs(first_file)
    obs_dim = sample_obs.shape[1]
    print(f"Observation dimension: {obs_dim}")
    
    print("Loading rollouts into memory...")
    rollout_cache: Dict[str, torch.Tensor] = {}
    for filename in global_order:
        filepath = os.path.join(data_folder, filename)
        try:
            rollout_cache[filename] = load_rollout_obs(filepath).to(device)
        except Exception as e:
            print(f"  Warning: Failed to load {filename}: {e}")
    
    print(f"Loaded {len(rollout_cache)}/{len(global_order)} rollouts")
    
    pairs = [(w, l) for w, l in pairs if w in rollout_cache and l in rollout_cache]
    
    random.shuffle(pairs)
    val_size = int(len(pairs) * PREF_VALIDATION_SPLIT)
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]
    
    print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
    
    model = PreferenceRewardNetwork(obs_dim, PREF_HIDDEN_LAYERS, PREF_ACTIVATION).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=PREF_WEIGHT_DECAY)
    
    writer = SummaryWriter(log_dir=str(tensorboard_dir / f"iteration_{iteration_num}"))
    
    training_log = {
        "epochs": [],
        "train_losses": [],
        "val_losses": [],
        "train_accs": [],
        "val_accs": [],
        "best_val_loss": None,
        "best_epoch": None
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        random.shuffle(train_pairs)
        
        train_losses = []
        train_correct = 0
        train_total = 0
        
        for i in range(0, len(train_pairs), batch_size):
            batch_pairs = train_pairs[i:i + batch_size]
            
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
        
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for i in range(0, len(val_pairs), batch_size):
                batch_pairs = val_pairs[i:i + batch_size]
                
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
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/validation", val_loss, epoch + 1)
        writer.add_scalar("Accuracy/train", train_acc, epoch + 1)
        writer.add_scalar("Accuracy/validation", val_acc, epoch + 1)
        
        training_log["epochs"].append(epoch + 1)
        training_log["train_losses"].append(train_loss)
        training_log["val_losses"].append(val_loss)
        training_log["train_accs"].append(train_acc)
        training_log["val_accs"].append(val_acc)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            training_log["best_val_loss"] = best_val_loss
            training_log["best_epoch"] = epoch + 1
            patience_counter = 0
            print(f"  New Best: {best_val_loss:.5f}")
        else:
            patience_counter += 1
            if patience_counter >= PREF_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    writer.add_scalar("Summary/best_val_loss", best_val_loss, iteration_num)
    writer.close()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model_state_path = iteration_dir / "reward_model.pth"
    model_full_path = iteration_dir / "reward_model_full.pth"
    model_jit_path = iteration_dir / "reward_model.pt"
    
    torch.save(best_model_state, str(model_state_path))
    torch.save(model, str(model_full_path))
    
    model.eval()
    ts_model = torch.jit.script(model)
    ts_model.save(str(model_jit_path))
    
    shutil.copy(str(model_jit_path), str(GLOBAL_REWARD_MODEL_PATH))
    
    with open(iteration_dir / "reward_model_training_log.json", 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"Training complete. Best validation loss: {best_val_loss:.5f}")
    print(f"Models saved to: {iteration_dir}")
    print(f"Global reward model updated: {GLOBAL_REWARD_MODEL_PATH}")
    
    return model


def run_rl_training(task, seed, rl_epochs, save_frequency, iteration_dir, iteration_num, tensorboard_dir):
    """Run RL training with the learned reward model"""
    print("\n" + "="*60)
    print(f"STEP 2: Running RL Training for {task}")
    print(f"  Epochs: {rl_epochs}, Save Frequency: {save_frequency}")
    print("="*60)

    rl_log_path = iteration_dir / "rl_training_log.txt"
    suffix = "GPT"

    # Record timestamp before training starts - used to find correct policy dir
    training_start_time = time.time()

    with open(rl_log_path, 'w') as f:
        process = subprocess.Popen(
            [
                'conda', 'run', '-n', EUREKA_CONDA_ENV, '--no-capture-output',
                'python', '-u', f'{ISAAC_ROOT_DIR}/train.py',
                'hydra/output=subprocess',
                f'task={task}{suffix}',
                'headless=True',
                'capture_video=False',
                'force_render=False',
                f'seed={seed}',
                f'max_iterations={rl_epochs}',
                f'train.params.config.save_frequency={save_frequency}',
            ],
            stdout=f,
            stderr=f,
            cwd=str(ISAAC_ROOT_DIR),
        )
        process.wait()

    print(f"RL training completed. Log: {rl_log_path}")

    # Find the policy directory created during THIS training run
    policy_dir = _find_policy_dir(task, training_start_time)
    if policy_dir is None:
        print("ERROR: No policy directory found after training")
        return None
    
    # Move policy directory to iteration folder
    dest_policy_dir = iteration_dir / "policy"
    shutil.move(str(policy_dir), str(dest_policy_dir))
    
    print(f"Policy moved to: {dest_policy_dir}")

    # Copy RL TensorBoard logs to experiment tensorboard directory
    _copy_rl_tensorboard_logs(task, dest_policy_dir, tensorboard_dir, iteration_num)
    
    return dest_policy_dir


def _find_policy_dir(task: str, start_time: float) -> Path:
    """
    Find the policy directory created during this training run.
    
    Args:
        task: Task name (e.g., "ShadowHandDoorOpenInward")
        start_time: Unix timestamp when training started
    
    Returns:
        Path to the policy directory, or None if not found
    """
    suffix = "GPT"
    
    # Get all policy directories (created under ISAAC_ROOT_DIR, where train.py runs)
    all_policy_dirs = list(ISAAC_ROOT_DIR.glob("policy-*"))
    
    # Filter by creation time (must be created after training started)
    recent_dirs = [p for p in all_policy_dirs if p.stat().st_mtime >= start_time]
    
    if not recent_dirs:
        print(f"  Warning: No policy directories created after training start time")
        # Fallback: try all directories sorted by mtime
        recent_dirs = sorted(all_policy_dirs, key=lambda p: p.stat().st_mtime)
    
    # Filter by task name - check if the runs subdirectory matches our task
    matching_dirs = []
    for policy_dir in recent_dirs:
        runs_pattern = f"runs/{task}{suffix}-*"
        if list(policy_dir.glob(runs_pattern)):
            matching_dirs.append(policy_dir)
    
    if matching_dirs:
        # Return the most recently modified matching directory
        best_dir = max(matching_dirs, key=lambda p: p.stat().st_mtime)
        print(f"  Found policy directory: {best_dir}")
        return best_dir
    
    # Last resort: return the most recent directory (original behavior)
    if recent_dirs:
        best_dir = max(recent_dirs, key=lambda p: p.stat().st_mtime)
        print(f"  Warning: No task-matching policy dir found, using most recent: {best_dir}")
        return best_dir
    
    return None


def _copy_rl_tensorboard_logs(task, policy_dir, tensorboard_dir, iteration_num):
    """Copy RL training TensorBoard logs to experiment tensorboard directory"""
    suffix = "GPT"
    
    # Find summaries directory in the policy runs folder
    summaries_dirs = list(policy_dir.glob(f"runs/{task}{suffix}-*/summaries"))
    if not summaries_dirs:
        print(f"  Warning: No RL tensorboard summaries found in {policy_dir}")
        return
    
    summaries_dir = summaries_dirs[0]
    
    # Create destination directory for RL logs
    rl_tb_dir = tensorboard_dir / f"rl_iteration_{iteration_num}"
    rl_tb_dir.mkdir(exist_ok=True)
    
    # Copy all event files
    event_files = list(summaries_dir.glob("events.out.tfevents.*"))
    for event_file in event_files:
        dest_file = rl_tb_dir / event_file.name
        shutil.copy(str(event_file), str(dest_file))
    
    print(f"  Copied {len(event_files)} RL tensorboard files to {rl_tb_dir}")


def find_checkpoints(task, policy_dir, checkpoint_start, checkpoint_step, checkpoint_end):
    """Find checkpoint files matching the specified epochs"""
    suffix = "GPT"

    nn_dirs = list(policy_dir.glob(f"runs/{task}{suffix}-*/nn"))
    if not nn_dirs:
        print(f"ERROR: No nn directory found in {policy_dir}")
        return []

    nn_dir = nn_dirs[0]
    print(f"Looking for checkpoints in: {nn_dir}")

    checkpoints = []
    for epoch in range(checkpoint_start, checkpoint_end + 1, checkpoint_step):
        pattern = f"{task}{suffix}_successes_{epoch}_*.pth"
        matches = list(nn_dir.glob(pattern))
        if matches:
            checkpoints.append((epoch, matches[0]))
        else:
            pattern2 = f"*_{epoch}_*.pth"
            matches2 = list(nn_dir.glob(pattern2))
            if matches2:
                checkpoints.append((epoch, matches2[0]))

    print(f"Found {len(checkpoints)} checkpoints")
    return checkpoints


def capture_single_rollout(seed, checkpoint_path, task, suffix, output_log, capture_video=False):
    """Capture a rollout from a policy checkpoint"""
    from utils.misc import block_until_rollout_captured

    # For video capture, need headless=False
    headless = "False" if capture_video else "True"
    
    with open(output_log, 'w') as f:
        process = subprocess.Popen(
            [
                'conda', 'run', '-n', EUREKA_CONDA_ENV, '--no-capture-output',
                'python', '-u', f'{ISAAC_ROOT_DIR}/train.py',
                'hydra/output=subprocess',
                'test=True',
                f'checkpoint={checkpoint_path}',
                f'task={task}{suffix}',
                f'headless={headless}',
                f'capture_video={capture_video}',
                'force_render=True',
                f'seed={seed}',
                'task.env.printNumSuccesses=False',
            ],
            stdout=f,
            stderr=f,
            cwd=str(ISAAC_ROOT_DIR),
            preexec_fn=os.setsid,  # Create process group for clean termination
        )

        block_until_rollout_captured(
            str(output_log),
            log_status=True,
            task_name=task,
            stop_at_success=False,
            seed=seed,
            success_reached=None,
            capture_video=capture_video,
        )

        # Kill entire process group (conda run + child python)
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    return True


def collect_rollouts(task, policy_dir, checkpoints, seed, iteration_dir, capture_video=False):
    """Collect rollouts from policy checkpoints"""
    print("\n" + "="*60)
    print(f"STEP 3: Collecting Rollouts")
    print(f"  Number of checkpoints: {len(checkpoints)}")
    print(f"  Video capture: {capture_video}")
    print("="*60)

    # Use non-GPT task for rollout capture (has print statements for data collection)
    suffix = ""
    rollouts_dir = iteration_dir / "rollouts"
    rollouts_dir.mkdir(exist_ok=True)

    for i, (epoch, checkpoint_path) in enumerate(checkpoints):
        print(f"  Collecting rollout {i+1}/{len(checkpoints)} from epoch {epoch}")
        output_log = rollouts_dir / f"rollout_epoch{epoch}.txt"

        try:
            capture_single_rollout(seed, str(checkpoint_path), task, suffix, str(output_log), capture_video=capture_video)
        except Exception as e:
            print(f"    Warning: Failed to capture rollout for epoch {epoch}: {e}")
            continue

    print(f"Rollouts saved to: {rollouts_dir}")
    return rollouts_dir


def cleanup_unused_checkpoints(task, policy_dir, keep_epochs: set):
    """
    Delete checkpoint .pth files not needed for rollout collection.
    Saves a scores log (epoch -> success_score) before deleting.
    """
    suffix = "GPT"
    nn_dirs = list(policy_dir.glob(f"runs/{task}{suffix}-*/nn"))
    if not nn_dirs:
        print("  No nn directory found, skipping checkpoint cleanup")
        return

    nn_dir = nn_dirs[0]
    all_pth = list(nn_dir.glob("*.pth"))
    if not all_pth:
        print("  No checkpoint files found")
        return

    scores_log = []
    deleted = 0

    for pth in all_pth:
        name = pth.name

        # Extract epoch: ShadowHand...GPT_successes_{epoch}_{score}.pth
        m = re.search(r'_successes_(\d+)_([\d.]+)\.pth$', name)
        if m:
            epoch, score = int(m.group(1)), float(m.group(2))
            scores_log.append({"epoch": epoch, "score": score, "file": name})
            if epoch not in keep_epochs:
                pth.unlink()
                deleted += 1
            continue

        # Extract epoch: last_...GPT_ep_{epoch}.pth
        m = re.search(r'_ep_(\d+)\.pth$', name)
        if m:
            epoch = int(m.group(1))
            scores_log.append({"epoch": epoch, "score": None, "file": name})
            if epoch not in keep_epochs:
                pth.unlink()
                deleted += 1
            continue

        # Unknown format — keep it
        scores_log.append({"epoch": None, "score": None, "file": name})

    scores_log.sort(key=lambda x: (x["epoch"] is None, x["epoch"] or 0))

    log_path = policy_dir / "checkpoint_scores.json"
    with open(log_path, 'w') as f:
        json.dump(scores_log, f, indent=2)

    kept = len(all_pth) - deleted
    print(f"  Checkpoint cleanup: kept {kept}, deleted {deleted} (log: {log_path})")


def move_rollouts_to_training_data(data_folder):
    """Move collected rollouts from auto_preference_data to the experiment's data folder"""
    print("\n" + "="*60)
    print(f"STEP 4: Moving Rollouts to Training Data")
    print("="*60)

    data_target_dir = Path(data_folder)

    if not DATA_SOURCE_DIR.exists():
        print(f"Warning: Source directory does not exist: {DATA_SOURCE_DIR}")
        return 0

    data_target_dir.mkdir(exist_ok=True)

    source_files = list(DATA_SOURCE_DIR.glob("*.txt"))
    moved_count = 0

    for src_file in source_files:
        # Skip ignored files
        if src_file.name in IGNORED_FILES:
            continue
            
        dst_file = data_target_dir / src_file.name
        if dst_file.exists():
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            dst_file = data_target_dir / f"{src_file.stem}_{timestamp}{src_file.suffix}"

        shutil.move(str(src_file), str(dst_file))
        moved_count += 1

    print(f"Moved {moved_count} files from {DATA_SOURCE_DIR} to {data_target_dir}")
    return moved_count


def create_experiment_dir(task):
    """Create a new experiment directory with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{task}_{timestamp}"
    experiment_dir = EXPERIMENTS_DIR / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def save_config(experiment_dir, config):
    """Save configuration to JSON file"""
    config_path = experiment_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")


def run_pipeline(task, num_iterations, seed, rl_epochs, save_frequency,
                 checkpoint_start, checkpoint_step, checkpoint_end,
                 training_epochs, batch_size, learning_rate, reward_type="ground_truth"):
    """
    Run the full iterative reward learning pipeline.
    
    Args:
        reward_type: "ground_truth" for MSE-based training, "preference" for Bradley-Terry
    """
    
    # Create experiment directory
    experiment_dir = create_experiment_dir(task)
    
    print("\n" + "#"*60)
    print("ITERATIVE REWARD LEARNING PIPELINE")
    print("#"*60)
    print(f"Experiment Directory: {experiment_dir}")
    print(f"\nConfiguration:")
    print(f"  Task: {task}")
    print(f"  Reward Type: {reward_type}")
    print(f"  Loop Iterations: {num_iterations}")
    print(f"  Seed: {seed}")
    print(f"  RL Epochs: {rl_epochs}")
    print(f"  Checkpoint Save Frequency: {save_frequency}")
    print(f"  Checkpoints to collect: {checkpoint_start} to {checkpoint_end} step {checkpoint_step}")
    print(f"  Reward Model Training Epochs: {training_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")

    # Validate task
    if task not in TASK_INPUT_KEYS:
        print(f"ERROR: Unknown task '{task}'. Supported tasks: {list(TASK_INPUT_KEYS.keys())}")
        return False

    # Validate reward_type
    if reward_type not in ("ground_truth", "preference"):
        print(f"ERROR: Unknown reward_type '{reward_type}'. Must be 'ground_truth' or 'preference'")
        return False

    base_data_folder = TASK_DATA_FOLDERS[task]
    data_folder = experiment_dir / "data"
    data_folder.mkdir(exist_ok=True)

    # Copy starter data into experiment-local folder
    if base_data_folder.exists():
        base_files = [f for f in base_data_folder.iterdir() if f.is_file()]
        for src in base_files:
            shutil.copy2(str(src), str(data_folder / src.name))
        print(f"  Copied {len(base_files)} base data files from {base_data_folder}")
    else:
        print(f"  Warning: Base data folder does not exist: {base_data_folder}")

    print(f"  Data Folder: {data_folder}")

    # For preference learning, enable video capture
    capture_video = (reward_type == "preference")
    if capture_video:
        print(f"  Video Capture: Enabled (required for VLM preference labeling)")

    # Save configuration
    config = {
        "task": task,
        "reward_type": reward_type,
        "num_iterations": num_iterations,
        "seed": seed,
        "rl_epochs": rl_epochs,
        "save_frequency": save_frequency,
        "checkpoint_start": checkpoint_start,
        "checkpoint_step": checkpoint_step,
        "checkpoint_end": checkpoint_end,
        "training_epochs": training_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "base_data_folder": str(base_data_folder),
        "data_folder": str(data_folder),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    save_config(experiment_dir, config)

    # Create tensorboard directory for all iterations
    tensorboard_dir = experiment_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    print(f"TensorBoard logs: {tensorboard_dir}")
    print(f"  To view: tensorboard --logdir={tensorboard_dir}")

    for iteration in range(1, num_iterations + 1):
        print("\n" + "#"*60)
        print(f"ITERATION {iteration}/{num_iterations} - {task} ({reward_type})")
        print("#"*60)

        # Create iteration directory
        iteration_dir = experiment_dir / f"iteration_{iteration}"
        iteration_dir.mkdir(exist_ok=True)
        print(f"Iteration directory: {iteration_dir}")

        # Step 1: Train reward model based on reward_type
        if reward_type == "ground_truth":
            model = train_reward_model(
                task,
                str(data_folder),
                iteration_dir,
                iteration_num=iteration,
                tensorboard_dir=tensorboard_dir,
                batch_size=batch_size,
                num_epochs=training_epochs,
                lr=learning_rate
            )
        else:  # preference
            model = train_preference_reward_model(
                task,
                str(data_folder),
                iteration_dir,
                iteration_num=iteration,
                tensorboard_dir=tensorboard_dir,
                batch_size=batch_size,
                num_epochs=training_epochs,
                lr=learning_rate
            )
        
        if model is None:
            print(f"ERROR: Iteration {iteration} failed at reward model training")
            return False

        # Step 2: Run RL training
        policy_dir = run_rl_training(task, seed, rl_epochs, save_frequency, iteration_dir,
                                     iteration_num=iteration, tensorboard_dir=tensorboard_dir)
        if policy_dir is None:
            print(f"ERROR: Iteration {iteration} failed at RL training")
            return False

        # Step 3: Find and collect rollouts from checkpoints
        checkpoints = find_checkpoints(task, policy_dir, checkpoint_start, checkpoint_step, checkpoint_end)
        if not checkpoints:
            print(f"Warning: No checkpoints found for iteration {iteration}")
        else:
            collect_rollouts(task, policy_dir, checkpoints, seed, iteration_dir, capture_video=capture_video)

        # Step 3.5: Delete unused checkpoints to save disk space
        keep_epochs = set(range(checkpoint_start, checkpoint_end + 1, checkpoint_step))
        cleanup_unused_checkpoints(task, policy_dir, keep_epochs)

        # Step 4: Move rollouts to experiment data directory
        move_rollouts_to_training_data(data_folder)

        # Save iteration summary
        iteration_summary = {
            "iteration": iteration,
            "reward_type": reward_type,
            "completed_at": datetime.datetime.now().isoformat(),
            "num_checkpoints_collected": len(checkpoints),
            "policy_dir": str(policy_dir),
        }
        with open(iteration_dir / "summary.json", 'w') as f:
            json.dump(iteration_summary, f, indent=2)

        print(f"\nIteration {iteration} completed successfully!")

    # Save final summary
    final_summary = {
        "status": "completed",
        "reward_type": reward_type,
        "total_iterations": num_iterations,
        "completed_at": datetime.datetime.now().isoformat(),
    }
    with open(experiment_dir / "final_summary.json", 'w') as f:
        json.dump(final_summary, f, indent=2)

    print("\n" + "#"*60)
    print("PIPELINE COMPLETED")
    print(f"Results saved to: {experiment_dir}")
    print("#"*60)
    return True


SUPPORTED_TASKS = list(TASK_INPUT_KEYS.keys())


def main():
    parser = argparse.ArgumentParser(
        description="Iterative Reward Learning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported tasks:
  {chr(10).join(f'  - {t}' for t in SUPPORTED_TASKS)}

Reward Types:
  - ground_truth: Train reward model using MSE loss against ground truth reward
  - preference: Train using Bradley-Terry loss from VLM preference rankings

Output Structure:
  experiments/
    {{task}}_{{timestamp}}/
      config.json
      iteration_1/
        reward_model.pth
        reward_model_full.pth
        reward_model.pt
        rl_training_log.txt
        reward_model_training_log.json
        summary.json
        policy/
        rollouts/
      iteration_2/
        ...

Example usage:
  # Ground truth mode (default)
  python reward_learning_pipeline.py --task ShadowHandDoorOpenOutward --num_iterations 3
  
  # Preference learning mode (uses VLM for labeling)
  python reward_learning_pipeline.py --task ShadowHandDoorOpenInward --reward_type preference --training_epochs 100
"""
    )
    parser.add_argument("--task", type=str, default="ShadowHandDoorOpenInward",
                        choices=SUPPORTED_TASKS,
                        help=f"Task to run the pipeline for (default: ShadowHandDoorOpenInward)")
    parser.add_argument("--reward_type", type=str, default="preference",
                        choices=["ground_truth", "preference"],
                        help="Reward model training approach: ground_truth (MSE) or preference (Bradley-Terry)")
    parser.add_argument("--num_iterations", type=int, default=5,
                        help="Number of iterations to run the full loop")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for RL training and rollout collection")
    parser.add_argument("--rl_epochs", type=int, default=1700,
                        help="Number of epochs to train RL")
    parser.add_argument("--save_frequency", type=int, default=3,
                        help="How often to save checkpoints during RL training")
    parser.add_argument("--checkpoint_start", type=int, default=3,
                        help="First checkpoint epoch to collect rollouts from")
    parser.add_argument("--checkpoint_step", type=int, default=3,
                        help="Step between checkpoint epochs to collect")
    parser.add_argument("--checkpoint_end", type=int, default=99,
                        help="Last checkpoint epoch to collect rollouts from")
    parser.add_argument("--training_epochs", type=int, default=40,
                        help="Number of epochs to train the reward model (use ~100 for preference)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for reward model training (use ~32 for preference)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for reward model training")

    args = parser.parse_args()

    run_pipeline(
        task=args.task,
        num_iterations=args.num_iterations,
        seed=args.seed,
        rl_epochs=args.rl_epochs,
        save_frequency=args.save_frequency,
        checkpoint_start=args.checkpoint_start,
        checkpoint_step=args.checkpoint_step,
        checkpoint_end=args.checkpoint_end,
        training_epochs=args.training_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        reward_type=args.reward_type,
    )


if __name__ == "__main__":
    main()
