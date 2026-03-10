import torch
from torch import Tensor
from typing import Tuple, Dict


# ============================================================
# ShadowHandDoorOpenOutward — Eureka reported reward function
# Source: https://eureka-research.github.io/
# ============================================================

@torch.jit.script
def compute_reward_door_open_outward(
    left_hand_pos: Tensor,
    door_left_handle_pos: Tensor,
    door_left_handle_rot: Tensor,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    # Define the temperature parameters for reward components
    handle_temperature: float = 0.5
    door_position_temperature: float = 5.0
    door_orientation_temperature: float = 0.5

    # Calculate the distance between the left hand and the door left handle position
    handle_dist = torch.norm(left_hand_pos - door_left_handle_pos, dim=-1)

    # Calculate the door position reward based on door's rotation (assuming the door is perfectly closed when the quaternion is (1, 0, 0, 0))
    door_closed_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], device=door_left_handle_rot.device)
    door_position_reward = torch.exp(-door_position_temperature * (1 - torch.sum(door_left_handle_rot * door_closed_quaternion, dim=-1)))

    # Calculate the door orientation reward based on the angle difference between the door's current orientation and the goal orientation
    dot_product = torch.sum(door_left_handle_rot * door_closed_quaternion, dim=-1)
    angle_diff = 2 * torch.acos(torch.clamp(dot_product, -1, 1))
    door_orientation_reward = torch.exp(-door_orientation_temperature * angle_diff)

    # Calculate the handle reward
    handle_reward = torch.exp(-handle_temperature * handle_dist)

    # Calculate the total reward
    total_reward = door_position_reward + handle_reward + door_orientation_reward

    # Store individual reward components in a dictionary
    reward_dict = {
        "door_position_reward": door_position_reward,
        "handle_reward": handle_reward,
        "door_orientation_reward": door_orientation_reward,
    }

    return total_reward, reward_dict


# ============================================================
# ShadowHandDoorOpenInward — Eureka reported reward function
# Source: https://eureka-research.github.io/
# ============================================================

@torch.jit.script
def compute_reward_door_open_inward(
    left_hand_pos: Tensor,
    right_hand_pos: Tensor,
    left_hand_rot: Tensor,
    right_hand_rot: Tensor,
    door_left_handle_pos: Tensor,
    door_right_handle_pos: Tensor,
    door_left_handle_rot: Tensor,
    door_right_handle_rot: Tensor,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    dist_left_handle_left_hand = torch.norm(left_hand_pos - door_left_handle_pos, dim=-1)
    dist_right_handle_left_hand = torch.norm(left_hand_pos - door_right_handle_pos, dim=-1)
    dist_left_handle_right_hand = torch.norm(right_hand_pos - door_left_handle_pos, dim=-1)
    dist_right_handle_right_hand = torch.norm(right_hand_pos - door_right_handle_pos, dim=-1)

    min_dist_left_handle_left_hand = torch.min(dist_left_handle_left_hand, dist_right_handle_left_hand)
    min_dist_right_handle_right_hand = torch.min(dist_left_handle_right_hand, dist_right_handle_right_hand)

    hand_handle_dist_weight: float = 0.9
    hand_handle_dist_temp: float = 1.2

    exp_min_dist_left_handle_left_hand = torch.exp(-hand_handle_dist_temp * min_dist_left_handle_left_hand)
    exp_min_dist_right_handle_right_hand = torch.exp(-hand_handle_dist_temp * min_dist_right_handle_right_hand)

    reward_hand_handle_dist_left_hand = hand_handle_dist_weight * exp_min_dist_left_handle_left_hand
    reward_hand_handle_dist_right_hand = hand_handle_dist_weight * exp_min_dist_right_handle_right_hand

    # Calculate door handle orientation reward
    door_handle_orientation_weight: float = 1.0
    door_handle_orientation_temp: float = 1.5

    door_left_handle_rot_vec = torch.atan2(
        2 * (door_left_handle_rot[:, 0] * door_left_handle_rot[:, 1] + door_left_handle_rot[:, 2] * door_left_handle_rot[:, 3]),
        1 - 2 * (door_left_handle_rot[:, 1] * door_left_handle_rot[:, 1] + door_left_handle_rot[:, 2] * door_left_handle_rot[:, 2]),
    )
    door_right_handle_rot_vec = torch.atan2(
        2 * (door_right_handle_rot[:, 0] * door_right_handle_rot[:, 1] + door_right_handle_rot[:, 2] * door_right_handle_rot[:, 3]),
        1 - 2 * (door_right_handle_rot[:, 1] * door_right_handle_rot[:, 1] + door_right_handle_rot[:, 2] * door_right_handle_rot[:, 2]),
    )

    left_hand_rot_vec = torch.atan2(
        2 * (left_hand_rot[:, 0] * left_hand_rot[:, 1] + left_hand_rot[:, 2] * left_hand_rot[:, 3]),
        1 - 2 * (left_hand_rot[:, 1] * left_hand_rot[:, 1] + left_hand_rot[:, 2] * left_hand_rot[:, 2]),
    )
    right_hand_rot_vec = torch.atan2(
        2 * (right_hand_rot[:, 0] * right_hand_rot[:, 1] + right_hand_rot[:, 2] * right_hand_rot[:, 3]),
        1 - 2 * (right_hand_rot[:, 1] * right_hand_rot[:, 1] + right_hand_rot[:, 2] * right_hand_rot[:, 2]),
    )

    door_left_handle_orientation_diff = torch.abs(left_hand_rot_vec - door_left_handle_rot_vec)
    door_right_handle_orientation_diff = torch.abs(right_hand_rot_vec - door_right_handle_rot_vec)

    exp_door_left_handle_orientation_diff = torch.exp(-door_handle_orientation_temp * door_left_handle_orientation_diff)
    exp_door_right_handle_orientation_diff = torch.exp(-door_handle_orientation_temp * door_right_handle_orientation_diff)

    reward_door_handle_orientation_left = door_handle_orientation_weight * exp_door_left_handle_orientation_diff
    reward_door_handle_orientation_right = door_handle_orientation_weight * exp_door_right_handle_orientation_diff

    overall_reward = (
        reward_hand_handle_dist_left_hand.mean()
        + reward_hand_handle_dist_right_hand.mean()
        + reward_door_handle_orientation_left.mean()
        + reward_door_handle_orientation_right.mean()
    ) / 4

    rewards = {
        "reward_hand_handle_dist_left_hand": reward_hand_handle_dist_left_hand.mean(),
        "reward_hand_handle_dist_right_hand": reward_hand_handle_dist_right_hand.mean(),
        "reward_door_handle_orientation_left": reward_door_handle_orientation_left.mean(),
        "reward_door_handle_orientation_right": reward_door_handle_orientation_right.mean(),
        "overall_reward": overall_reward,
    }

    return overall_reward, rewards
