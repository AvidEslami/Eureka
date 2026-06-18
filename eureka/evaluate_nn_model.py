import torch
import torch.nn as nn
import torch.nn.functional as F
import os

MAX_ROLLOUT_LENGTH = 1000000

# Must match the class used during training
class nn_reward_model(nn.Module):
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

def ground_truth_model(door_right_handle_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos, door_left_handle_pos, left_hand_ff_pos, left_hand_mf_pos, left_hand_rf_pos, left_hand_lf_pos, left_hand_th_pos):
    right_hand_finger_dist = (torch.norm(door_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(door_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                            + torch.norm(door_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(door_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                            + torch.norm(door_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
    left_hand_finger_dist = (torch.norm(door_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(door_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                            + torch.norm(door_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(door_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                            + torch.norm(door_left_handle_pos - left_hand_th_pos, p=2, dim=-1))

    right_hand_dist_rew = right_hand_finger_dist
    left_hand_dist_rew = left_hand_finger_dist

    up_rew = torch.zeros_like(right_hand_dist_rew)
    up_rew = torch.where(right_hand_finger_dist < 0.5,
                    torch.where(left_hand_finger_dist < 0.5,
                                    torch.abs(door_right_handle_pos[:, 1] - door_left_handle_pos[:, 1]) * 2, up_rew), up_rew)

    reward = 2 - right_hand_dist_rew - left_hand_dist_rew + up_rew
    return reward

input_keys = ["door_right_handle_pos", "right_hand_ff_pos", "right_hand_mf_pos", "right_hand_rf_pos", "right_hand_lf_pos", "right_hand_th_pos", "door_left_handle_pos", "left_hand_ff_pos", "left_hand_mf_pos", "left_hand_rf_pos", "left_hand_lf_pos", "left_hand_th_pos"]

def get_rollout_observations(rollout_path, task, required_keys=None, max_length=None, nn=False):
    if task in ("ShadowHandDoorOpenInward", "ShadowHandDoorOpenOutward"):
        with open(rollout_path, 'r') as f:
            f.readline()
            f.readline()
            data = [line for line in f]

            try:
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
            except StopIteration:
                return []

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
                usable_length = min(MAX_ROLLOUT_LENGTH, 
                                    len(door_left_handle_pos), len(door_right_handle_pos), 
                                    len(right_hand_ff_pos), len(right_hand_mf_pos),
                                    len(right_hand_rf_pos), len(right_hand_lf_pos), len(right_hand_th_pos),
                                    len(left_hand_ff_pos), len(left_hand_mf_pos), len(left_hand_rf_pos),
                                    len(left_hand_lf_pos), len(left_hand_th_pos))
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
    return []


if __name__ == "__main__":
    # ========== CONFIGURATION ==========
    eval_data_folder = "./auto_preference_data"  # Evaluation dataset folder
    model_path = "./best_nn_reward_model_full.pth"  # Path to trained model
    task = "ShadowHandDoorOpenOutward"
    # ===================================

    print(f"Loading model from: {model_path}")
    nn_model = torch.load(model_path)
    nn_model.eval()

    print(f"Loading evaluation data from: {eval_data_folder}")
    filenames = [f for f in os.listdir(eval_data_folder) if f.endswith(".txt")]
    print(f"Found {len(filenames)} files")

    # Load all evaluation data
    eval_data = []
    for file in filenames:
        filepath = os.path.join(eval_data_folder, file)
        try:
            gt_obs = get_rollout_observations(filepath, task, input_keys, nn=False)
            nn_obs = get_rollout_observations(filepath, task, input_keys, nn=True)
            if len(gt_obs) > 0 and len(nn_obs) > 0:
                for row in range(min(len(gt_obs), len(nn_obs))):
                    eval_data.append((gt_obs[row], nn_obs[row]))
                print(f"  Loaded {min(len(gt_obs), len(nn_obs))} samples from {file}")
            else:
                print(f"  Skipped {file} - no valid data")
        except Exception as e:
            print(f"  Error loading {file}: {e}")

    print(f"\nTotal evaluation samples: {len(eval_data)}")

    if len(eval_data) == 0:
        print("No evaluation data loaded!")
        exit(1)

    # Compute loss on evaluation data
    total_loss = 0.0
    total_samples = 0
    gt_rewards = []
    nn_rewards = []

    with torch.no_grad():
        for gt_input, nn_input in eval_data:
            gt_rew = ground_truth_model(
                gt_input["door_right_handle_pos"],
                gt_input["right_hand_ff_pos"],
                gt_input["right_hand_mf_pos"],
                gt_input["right_hand_rf_pos"],
                gt_input["right_hand_lf_pos"],
                gt_input["right_hand_th_pos"],
                gt_input["door_left_handle_pos"],
                gt_input["left_hand_ff_pos"],
                gt_input["left_hand_mf_pos"],
                gt_input["left_hand_rf_pos"],
                gt_input["left_hand_lf_pos"],
                gt_input["left_hand_th_pos"]
            )
            nn_rew = nn_model(nn_input["obs_buf"])
            
            loss = F.mse_loss(nn_rew.squeeze(), gt_rew)
            total_loss += loss.item()
            total_samples += 1
            
            gt_rewards.append(gt_rew.item())
            nn_rewards.append(nn_rew.item())

    avg_loss = total_loss / total_samples
    print(f"\n========== EVALUATION RESULTS ==========")
    print(f"Total samples: {total_samples}")
    print(f"Average MSE Loss: {avg_loss:.6f}")
    print(f"RMSE: {avg_loss**0.5:.6f}")
    print(f"GT Reward - Mean: {sum(gt_rewards)/len(gt_rewards):.4f}, Min: {min(gt_rewards):.4f}, Max: {max(gt_rewards):.4f}")
    print(f"NN Reward - Mean: {sum(nn_rewards)/len(nn_rewards):.4f}, Min: {min(nn_rewards):.4f}, Max: {max(nn_rewards):.4f}")

    # Optional: Plot comparison
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(gt_rewards[:1000], label='Ground Truth', alpha=0.7)
        plt.plot(nn_rewards[:1000], label='NN Model', alpha=0.7)
        plt.xlabel('Sample')
        plt.ylabel('Reward')
        plt.title('Reward Comparison (first 1000 samples)')
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.scatter(gt_rewards, nn_rewards, alpha=0.3, s=1)
        plt.xlabel('Ground Truth Reward')
        plt.ylabel('NN Model Reward')
        plt.title('GT vs NN Reward Scatter')
        plt.grid()
        
        plt.tight_layout()
        plt.savefig('evaluation_results.png')
        print(f"\nPlot saved to: evaluation_results.png")
        plt.show()
    except Exception as e:
        print(f"Could not create plot: {e}")
