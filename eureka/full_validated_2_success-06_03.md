class DynamicReward(nn.Module):
    def __init__(self):
        super().__init__()
        self.rotation_reward_temp = nn.Parameter(torch.tensor(20.0, dtype=torch.float32))
        self.angvel_threshold = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        self.angvel_penalty_temp = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        self.min_distance_temp = nn.Parameter(torch.tensor(10.0, dtype=torch.float32))

    def compute_reward(self,object_rot: torch. Tensor, goal_rot: torch. Tensor, object_angvel: torch. Tensor, object_pos: torch. Tensor, fingertip_pos: torch.Tensor):

        rot_diff = torch.abs(torch.sum(object_rot * goal_rot, dim=1) - 1) / 2
        rotation_reward_temp = self.rotation_reward_temp
        rotation_reward = torch.exp(-rotation_reward_temp * rot_diff)

        # Angular velocity penalty
        angvel_norm = torch.norm(object_angvel, dim=1)
        angvel_threshold = self.angvel_threshold
        angvel_penalty_temp = self.angvel_penalty_temp
        angular_velocity_penalty = torch.where(angvel_norm > angvel_threshold, torch.exp(-angvel_penalty_temp * (angvel_norm - angvel_threshold)), torch.zeros_like(angvel_norm))

        # Distance reward
        min_distance_temp = self.min_distance_temp
        min_distance = torch.min(torch.norm(fingertip_pos - object_pos[:, None], dim=2), dim=1).values
        uncapped_distance_reward = torch.exp(-min_distance_temp * min_distance)
        distance_reward = torch.clamp(uncapped_distance_reward, 0.0, 1.0)

        total_reward = rotation_reward - angular_velocity_penalty + distance_reward

        reward_components = {
            "rotation_reward": rotation_reward,
            "angular_velocity_penalty": angular_velocity_penalty,
            "distance_reward": distance_reward
        }
        return total_reward, reward_components

    def forward(self, **inputs):
        return self.compute_reward(object_rot= inputs['object_rot'], goal_rot= inputs['goal_rot'], object_angvel= inputs['object_angvel'], object_pos= inputs['object_pos'], fingertip_pos= inputs['fingertip_pos'])

Pairwise accuracy: 0.76
Failure tracking:
1_ShadowHand_2025-06-03_03-42-06.txt: [0]
3_ShadowHand_2025-06-03_03-42-53.txt: [0]
2_ShadowHand_2025-06-03_03-41-21.txt: [0]
2_ShadowHand_2025-06-03_03-47-03.txt: [6]
2_ShadowHand_2025-06-03_03-44-46.txt: [0]
2_ShadowHand_2025-06-03_03-50-24.txt: [0]
2_ShadowHand_2025-06-03_03-40-12.txt: [0]
3_ShadowHand_2025-06-03_03-47-26.txt: [7]
1_ShadowHand_2025-06-03_03-45-32.txt: [0]
2_ShadowHand_2025-06-03_03-51-32.txt: [0]
3_ShadowHand_2025-06-03_03-46-24.txt: [1]
2_ShadowHand_2025-06-03_03-42-28.txt: [1]
3_ShadowHand_2025-06-03_03-41-45.txt: [0]
3_ShadowHand_2025-06-03_03-53-16.txt: [0]
3_ShadowHand_2025-06-03_03-45-12.txt: [1]
1_ShadowHand_2025-06-03_03-31-46.txt: [0]
3_ShadowHand_2025-06-03_03-40-38.txt: [0]
3_ShadowHand_2025-06-03_03-51-58.txt: [0]
1_ShadowHand_2025-06-03_03-50-03.txt: [0]
1_ShadowHand_2025-06-03_03-46-43.txt: [7]
2_ShadowHand_2025-06-03_03-45-58.txt: [0]
1_ShadowHand_2025-06-03_03-52-24.txt: [0]
2_ShadowHand_2025-06-03_03-52-50.txt: [0]
1_ShadowHand_2025-06-03_03-39-46.txt: [1]
3_ShadowHand_2025-06-03_03-32-31.txt: [0]
2_ShadowHand_2025-06-03_03-43-38.txt: [1]
2_ShadowHand_2025-06-03_03-32-05.txt: [0]
1_ShadowHand_2025-06-03_03-51-10.txt: [0]
3_ShadowHand_2025-06-03_03-44-04.txt: [0]
1_ShadowHand_2025-06-03_03-41-01.txt: [0]
1_ShadowHand_2025-06-03_03-43-16.txt: [0]
3_ShadowHand_2025-06-03_03-50-50.txt: [0]
1_ShadowHand_2025-06-03_03-44-25.txt: [2]
Pairwise accuracy: 0.71
Epoch 1/45, Train Loss: 2.1518, Validation Loss: 1.9873
Learned parameters:
rotation_reward_temp: 20.5
angvel_threshold: 2.5
angvel_penalty_temp: 1.500000238418579
min_distance_temp: 9.5
Pairwise accuracy: 0.79
Pairwise accuracy: 0.75
Epoch 2/45, Train Loss: 1.9933, Validation Loss: 2.1220
Pairwise accuracy: 0.78
Pairwise accuracy: 0.71
Epoch 3/45, Train Loss: 1.9877, Validation Loss: 2.0472
Pairwise accuracy: 0.77
Pairwise accuracy: 0.75
Epoch 4/45, Train Loss: 1.9592, Validation Loss: 1.7186
Pairwise accuracy: 0.78
Pairwise accuracy: 0.79
Epoch 5/45, Train Loss: 2.0006, Validation Loss: 1.7865
Pairwise accuracy: 0.78
Pairwise accuracy: 0.75
Epoch 6/45, Train Loss: 2.0281, Validation Loss: 1.6749
Pairwise accuracy: 0.78
Pairwise accuracy: 0.79
Epoch 7/45, Train Loss: 1.8812, Validation Loss: 1.7037
Pairwise accuracy: 0.79
Pairwise accuracy: 0.75
Epoch 8/45, Train Loss: 1.8913, Validation Loss: 1.7012
Pairwise accuracy: 0.79
Pairwise accuracy: 0.79
Epoch 9/45, Train Loss: 1.8695, Validation Loss: 1.6386
Pairwise accuracy: 0.79
Pairwise accuracy: 0.79
Epoch 10/45, Train Loss: 1.7590, Validation Loss: 1.6093
Pairwise accuracy: 0.79
Failure tracking:
1_ShadowHand_2025-06-03_03-42-06.txt: [0, 0]
3_ShadowHand_2025-06-03_03-42-53.txt: [0, 0]
2_ShadowHand_2025-06-03_03-41-21.txt: [0, 0]
2_ShadowHand_2025-06-03_03-47-03.txt: [6, 6]
2_ShadowHand_2025-06-03_03-44-46.txt: [0, 0]
2_ShadowHand_2025-06-03_03-50-24.txt: [0, 0]
2_ShadowHand_2025-06-03_03-40-12.txt: [0, 0]
3_ShadowHand_2025-06-03_03-47-26.txt: [7, 7]
1_ShadowHand_2025-06-03_03-45-32.txt: [0, 0]
2_ShadowHand_2025-06-03_03-51-32.txt: [0, 0]
3_ShadowHand_2025-06-03_03-46-24.txt: [1, 1]
2_ShadowHand_2025-06-03_03-42-28.txt: [1, 1]
3_ShadowHand_2025-06-03_03-41-45.txt: [0, 0]
3_ShadowHand_2025-06-03_03-53-16.txt: [0, 0]
3_ShadowHand_2025-06-03_03-45-12.txt: [1, 1]
1_ShadowHand_2025-06-03_03-31-46.txt: [0, 0]
3_ShadowHand_2025-06-03_03-40-38.txt: [0, 0]
3_ShadowHand_2025-06-03_03-51-58.txt: [0, 0]
1_ShadowHand_2025-06-03_03-50-03.txt: [0, 0]
1_ShadowHand_2025-06-03_03-46-43.txt: [7, 6]
2_ShadowHand_2025-06-03_03-45-58.txt: [0, 0]
1_ShadowHand_2025-06-03_03-52-24.txt: [0, 0]
2_ShadowHand_2025-06-03_03-52-50.txt: [0, 0]
1_ShadowHand_2025-06-03_03-39-46.txt: [1, 1]
3_ShadowHand_2025-06-03_03-32-31.txt: [0, 0]
2_ShadowHand_2025-06-03_03-43-38.txt: [1, 0]
2_ShadowHand_2025-06-03_03-32-05.txt: [0, 0]
1_ShadowHand_2025-06-03_03-51-10.txt: [0, 0]
3_ShadowHand_2025-06-03_03-44-04.txt: [0, 0]
1_ShadowHand_2025-06-03_03-41-01.txt: [0, 0]
1_ShadowHand_2025-06-03_03-43-16.txt: [0, 0]
3_ShadowHand_2025-06-03_03-50-50.txt: [0, 0]
1_ShadowHand_2025-06-03_03-44-25.txt: [2, 1]
Pairwise accuracy: 0.79
Epoch 11/45, Train Loss: 1.6576, Validation Loss: 1.6611
Learned parameters:
rotation_reward_temp: 25.39907455444336
angvel_threshold: 4.200765132904053
angvel_penalty_temp: 1.9005541801452637
min_distance_temp: 4.41518497467041
Pairwise accuracy: 0.79
Pairwise accuracy: 0.79
Epoch 12/45, Train Loss: 1.6485, Validation Loss: 1.6561
Pairwise accuracy: 0.79
Pairwise accuracy: 0.79
Epoch 13/45, Train Loss: 1.6031, Validation Loss: 1.6393
Pairwise accuracy: 0.79
Pairwise accuracy: 0.79
Epoch 14/45, Train Loss: 1.4948, Validation Loss: 1.4833
Pairwise accuracy: 0.79
Pairwise accuracy: 0.75
Epoch 15/45, Train Loss: 1.3403, Validation Loss: 1.4314
Pairwise accuracy: 0.79
Pairwise accuracy: 0.75
Epoch 16/45, Train Loss: 1.2734, Validation Loss: 1.3864
Pairwise accuracy: 0.80
Pairwise accuracy: 0.82
Epoch 17/45, Train Loss: 1.2031, Validation Loss: 1.3090
Pairwise accuracy: 0.79
Pairwise accuracy: 0.71
Epoch 18/45, Train Loss: 1.1534, Validation Loss: 1.3842
Pairwise accuracy: 0.79
Pairwise accuracy: 0.75
Epoch 19/45, Train Loss: 1.1429, Validation Loss: 1.3435
Pairwise accuracy: 0.80
Pairwise accuracy: 0.75
Epoch 20/45, Train Loss: 1.0905, Validation Loss: 1.3328
Pairwise accuracy: 0.80
Failure tracking:
1_ShadowHand_2025-06-03_03-42-06.txt: [0, 0, 0]
3_ShadowHand_2025-06-03_03-42-53.txt: [0, 0, 0]
2_ShadowHand_2025-06-03_03-41-21.txt: [0, 0, 0]
2_ShadowHand_2025-06-03_03-47-03.txt: [6, 6, 5]
2_ShadowHand_2025-06-03_03-44-46.txt: [0, 0, 0]
2_ShadowHand_2025-06-03_03-50-24.txt: [0, 0, 0]
2_ShadowHand_2025-06-03_03-40-12.txt: [0, 0, 0]
3_ShadowHand_2025-06-03_03-47-26.txt: [7, 7, 5]
1_ShadowHand_2025-06-03_03-45-32.txt: [0, 0, 0]
2_ShadowHand_2025-06-03_03-51-32.txt: [0, 0, 0]
3_ShadowHand_2025-06-03_03-46-24.txt: [1, 1, 1]
2_ShadowHand_2025-06-03_03-42-28.txt: [1, 1, 1]
3_ShadowHand_2025-06-03_03-41-45.txt: [0, 0, 0]
3_ShadowHand_2025-06-03_03-53-16.txt: [0, 0, 0]
3_ShadowHand_2025-06-03_03-45-12.txt: [1, 1, 1]
1_ShadowHand_2025-06-03_03-31-46.txt: [0, 0, 0]
3_ShadowHand_2025-06-03_03-40-38.txt: [0, 0, 0]
3_ShadowHand_2025-06-03_03-51-58.txt: [0, 0, 0]
1_ShadowHand_2025-06-03_03-50-03.txt: [0, 0, 0]
1_ShadowHand_2025-06-03_03-46-43.txt: [7, 6, 6]
2_ShadowHand_2025-06-03_03-45-58.txt: [0, 0, 0]
1_ShadowHand_2025-06-03_03-52-24.txt: [0, 0, 1]
2_ShadowHand_2025-06-03_03-52-50.txt: [0, 0, 0]
1_ShadowHand_2025-06-03_03-39-46.txt: [1, 1, 1]
3_ShadowHand_2025-06-03_03-32-31.txt: [0, 0, 0]
2_ShadowHand_2025-06-03_03-43-38.txt: [1, 0, 0]
2_ShadowHand_2025-06-03_03-32-05.txt: [0, 0, 0]
1_ShadowHand_2025-06-03_03-51-10.txt: [0, 0, 0]
3_ShadowHand_2025-06-03_03-44-04.txt: [0, 0, 0]
1_ShadowHand_2025-06-03_03-41-01.txt: [0, 0, 0]
1_ShadowHand_2025-06-03_03-43-16.txt: [0, 0, 0]
3_ShadowHand_2025-06-03_03-50-50.txt: [0, 0, 0]
1_ShadowHand_2025-06-03_03-44-25.txt: [2, 1, 1]
Pairwise accuracy: 0.75
Epoch 21/45, Train Loss: 1.0834, Validation Loss: 1.3239
Learned parameters:
rotation_reward_temp: 29.90869140625
angvel_threshold: 4.032163143157959
angvel_penalty_temp: 2.993711233139038
min_distance_temp: -0.8857179284095764
Pairwise accuracy: 0.80
Pairwise accuracy: 0.71
Epoch 22/45, Train Loss: 1.0648, Validation Loss: 1.3588
Pairwise accuracy: 0.80
Pairwise accuracy: 0.64
Epoch 23/45, Train Loss: 1.0552, Validation Loss: 1.2688
Pairwise accuracy: 0.81
Pairwise accuracy: 0.64
Epoch 24/45, Train Loss: 0.9989, Validation Loss: 1.2493
Pairwise accuracy: 0.82
Pairwise accuracy: 0.64
Epoch 25/45, Train Loss: 1.0020, Validation Loss: 1.2427
Pairwise accuracy: 0.82
Pairwise accuracy: 0.64
Epoch 26/45, Train Loss: 0.9967, Validation Loss: 1.3505
Pairwise accuracy: 0.80
Pairwise accuracy: 0.68
Epoch 27/45, Train Loss: 1.0183, Validation Loss: 1.3146
Pairwise accuracy: 0.80
Pairwise accuracy: 0.71
Epoch 28/45, Train Loss: 1.0187, Validation Loss: 1.3233
Pairwise accuracy: 0.79
Pairwise accuracy: 0.71
Epoch 29/45, Train Loss: 1.0237, Validation Loss: 1.3157
Pairwise accuracy: 0.79
Pairwise accuracy: 0.68
Epoch 30/45, Train Loss: 1.0191, Validation Loss: 1.2943
Pairwise accuracy: 0.80
Failure tracking:
1_ShadowHand_2025-06-03_03-42-06.txt: [0, 0, 0, 0]
3_ShadowHand_2025-06-03_03-42-53.txt: [0, 0, 0, 0]
2_ShadowHand_2025-06-03_03-41-21.txt: [0, 0, 0, 0]
2_ShadowHand_2025-06-03_03-47-03.txt: [6, 6, 5, 6]
2_ShadowHand_2025-06-03_03-44-46.txt: [0, 0, 0, 0]
2_ShadowHand_2025-06-03_03-50-24.txt: [0, 0, 0, 0]
2_ShadowHand_2025-06-03_03-40-12.txt: [0, 0, 0, 0]
3_ShadowHand_2025-06-03_03-47-26.txt: [7, 7, 5, 4]
1_ShadowHand_2025-06-03_03-45-32.txt: [0, 0, 0, 0]
2_ShadowHand_2025-06-03_03-51-32.txt: [0, 0, 0, 0]
3_ShadowHand_2025-06-03_03-46-24.txt: [1, 1, 1, 1]
2_ShadowHand_2025-06-03_03-42-28.txt: [1, 1, 1, 1]
3_ShadowHand_2025-06-03_03-41-45.txt: [0, 0, 0, 0]
3_ShadowHand_2025-06-03_03-53-16.txt: [0, 0, 0, 0]
3_ShadowHand_2025-06-03_03-45-12.txt: [1, 1, 1, 1]
1_ShadowHand_2025-06-03_03-31-46.txt: [0, 0, 0, 0]
3_ShadowHand_2025-06-03_03-40-38.txt: [0, 0, 0, 0]
3_ShadowHand_2025-06-03_03-51-58.txt: [0, 0, 0, 0]
1_ShadowHand_2025-06-03_03-50-03.txt: [0, 0, 0, 0]
1_ShadowHand_2025-06-03_03-46-43.txt: [7, 6, 6, 6]
2_ShadowHand_2025-06-03_03-45-58.txt: [0, 0, 0, 0]
1_ShadowHand_2025-06-03_03-52-24.txt: [0, 0, 1, 1]
2_ShadowHand_2025-06-03_03-52-50.txt: [0, 0, 0, 0]
1_ShadowHand_2025-06-03_03-39-46.txt: [1, 1, 1, 1]
3_ShadowHand_2025-06-03_03-32-31.txt: [0, 0, 0, 0]
2_ShadowHand_2025-06-03_03-43-38.txt: [1, 0, 0, 0]
2_ShadowHand_2025-06-03_03-32-05.txt: [0, 0, 0, 0]
1_ShadowHand_2025-06-03_03-51-10.txt: [0, 0, 0, 0]
3_ShadowHand_2025-06-03_03-44-04.txt: [0, 0, 0, 0]
1_ShadowHand_2025-06-03_03-41-01.txt: [0, 0, 0, 0]
1_ShadowHand_2025-06-03_03-43-16.txt: [0, 0, 0, 0]
3_ShadowHand_2025-06-03_03-50-50.txt: [0, 0, 0, 0]
1_ShadowHand_2025-06-03_03-44-25.txt: [2, 1, 1, 1]
Pairwise accuracy: 0.64
Epoch 31/45, Train Loss: 1.0056, Validation Loss: 1.3220
Learned parameters:
rotation_reward_temp: 33.89820861816406
angvel_threshold: 3.8580095767974854
angvel_penalty_temp: 3.8610970973968506
min_distance_temp: -3.908681631088257
Pairwise accuracy: 0.80
Pairwise accuracy: 0.64
Epoch 32/45, Train Loss: 0.9970, Validation Loss: 1.2016
Pairwise accuracy: 0.80
Pairwise accuracy: 0.64
Epoch 33/45, Train Loss: 0.9667, Validation Loss: 1.3138
Pairwise accuracy: 0.80
Pairwise accuracy: 0.68
Epoch 34/45, Train Loss: 0.9894, Validation Loss: 1.2736
Pairwise accuracy: 0.80
Pairwise accuracy: 0.68
Epoch 35/45, Train Loss: 0.9902, Validation Loss: 1.2819
Pairwise accuracy: 0.79
Pairwise accuracy: 0.68
Epoch 36/45, Train Loss: 0.9957, Validation Loss: 1.2759
Pairwise accuracy: 0.79
Pairwise accuracy: 0.68
Epoch 37/45, Train Loss: 0.9921, Validation Loss: 1.2578
Pairwise accuracy: 0.80
Pairwise accuracy: 0.64
Epoch 38/45, Train Loss: 0.9797, Validation Loss: 1.2647
Pairwise accuracy: 0.80
Pairwise accuracy: 0.64
Epoch 39/45, Train Loss: 0.9747, Validation Loss: 1.2776
Pairwise accuracy: 0.80
Pairwise accuracy: 0.64
Epoch 40/45, Train Loss: 0.9695, Validation Loss: 1.2026
Pairwise accuracy: 0.80
Failure tracking:
1_ShadowHand_2025-06-03_03-42-06.txt: [0, 0, 0, 0, 0]
3_ShadowHand_2025-06-03_03-42-53.txt: [0, 0, 0, 0, 0]
2_ShadowHand_2025-06-03_03-41-21.txt: [0, 0, 0, 0, 0]
2_ShadowHand_2025-06-03_03-47-03.txt: [6, 6, 5, 6, 6]
2_ShadowHand_2025-06-03_03-44-46.txt: [0, 0, 0, 0, 0]
2_ShadowHand_2025-06-03_03-50-24.txt: [0, 0, 0, 0, 0]
2_ShadowHand_2025-06-03_03-40-12.txt: [0, 0, 0, 0, 0]
3_ShadowHand_2025-06-03_03-47-26.txt: [7, 7, 5, 4, 4]
1_ShadowHand_2025-06-03_03-45-32.txt: [0, 0, 0, 0, 0]
2_ShadowHand_2025-06-03_03-51-32.txt: [0, 0, 0, 0, 0]
3_ShadowHand_2025-06-03_03-46-24.txt: [1, 1, 1, 1, 1]
2_ShadowHand_2025-06-03_03-42-28.txt: [1, 1, 1, 1, 1]
3_ShadowHand_2025-06-03_03-41-45.txt: [0, 0, 0, 0, 0]
3_ShadowHand_2025-06-03_03-53-16.txt: [0, 0, 0, 0, 0]
3_ShadowHand_2025-06-03_03-45-12.txt: [1, 1, 1, 1, 1]
1_ShadowHand_2025-06-03_03-31-46.txt: [0, 0, 0, 0, 0]
3_ShadowHand_2025-06-03_03-40-38.txt: [0, 0, 0, 0, 0]
3_ShadowHand_2025-06-03_03-51-58.txt: [0, 0, 0, 0, 0]
1_ShadowHand_2025-06-03_03-50-03.txt: [0, 0, 0, 0, 0]
1_ShadowHand_2025-06-03_03-46-43.txt: [7, 6, 6, 6, 6]
2_ShadowHand_2025-06-03_03-45-58.txt: [0, 0, 0, 0, 0]
1_ShadowHand_2025-06-03_03-52-24.txt: [0, 0, 1, 1, 1]
2_ShadowHand_2025-06-03_03-52-50.txt: [0, 0, 0, 0, 0]
1_ShadowHand_2025-06-03_03-39-46.txt: [1, 1, 1, 1, 1]
3_ShadowHand_2025-06-03_03-32-31.txt: [0, 0, 0, 0, 0]
2_ShadowHand_2025-06-03_03-43-38.txt: [1, 0, 0, 0, 0]
2_ShadowHand_2025-06-03_03-32-05.txt: [0, 0, 0, 0, 0]
1_ShadowHand_2025-06-03_03-51-10.txt: [0, 0, 0, 0, 0]
3_ShadowHand_2025-06-03_03-44-04.txt: [0, 0, 0, 0, 0]
1_ShadowHand_2025-06-03_03-41-01.txt: [0, 0, 0, 0, 0]
1_ShadowHand_2025-06-03_03-43-16.txt: [0, 0, 0, 0, 0]
3_ShadowHand_2025-06-03_03-50-50.txt: [0, 0, 0, 0, 0]
1_ShadowHand_2025-06-03_03-44-25.txt: [2, 1, 1, 1, 1]
Pairwise accuracy: 0.64
Epoch 41/45, Train Loss: 0.9594, Validation Loss: 1.2512
Learned parameters:
rotation_reward_temp: 37.47400665283203
angvel_threshold: 3.8757753372192383
angvel_penalty_temp: 4.438076972961426
min_distance_temp: -5.097723960876465
Pairwise accuracy: 0.80
Pairwise accuracy: 0.68
Epoch 42/45, Train Loss: 0.9654, Validation Loss: 1.2358
Pairwise accuracy: 0.80
Pairwise accuracy: 0.71
Epoch 43/45, Train Loss: 0.9641, Validation Loss: 1.2296
Pairwise accuracy: 0.79
Pairwise accuracy: 0.68
Epoch 44/45, Train Loss: 0.9705, Validation Loss: 1.2387
Pairwise accuracy: 0.79
Pairwise accuracy: 0.68
Epoch 45/45, Train Loss: 0.9674, Validation Loss: 1.2221
Learned parameters:
rotation_reward_temp: 38.814964294433594
angvel_threshold: 3.906949758529663
angvel_penalty_temp: 4.602588653564453
min_distance_temp: -5.336449146270752
Done