# train_dynamic_reward.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import inspect
from typing import Dict, Tuple
from collections import defaultdict
torch.autograd.set_detect_anomaly(True)

LOG_FAILURES = False
LOG_SUCCESS = False
TRACK_FAILURES = True
FAILURE_TRACK_PROGRESS = defaultdict(list)
MAXIMIZE_LOSS = False # If True, the loss will be maximized instead of minimized

def return_env_vars(obs_buf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    object_pos = obs_buf[72:75].unsqueeze(0)
    object_rot = obs_buf[75:79].unsqueeze(0)
    object_angvel = (obs_buf[82:85] / 0.2).unsqueeze(0) # Velocities are scaled by 0.2 -> this is a hardcoded environment constant
    goal_rot = obs_buf[88:92].unsqueeze(0)
    
    fingertip_start_index = 96
    fingertip_state_size = 13
    fingertip_data = []
    for i in range(5): # 5 fingertips
        idx = fingertip_start_index + i * fingertip_state_size
        fingertip_data.append(obs_buf[idx:idx + 3])
    
    # fingertip_pos = torch.tensor(fingertip_data, dtype=torch.float32).reshape(1, 5, 3)
    fingertip_pos = torch.stack(fingertip_data).unsqueeze(0)

    return {
        "object_pos": object_pos,
        "object_rot":object_rot, 
        "goal_rot":goal_rot, 
        "object_angvel":object_angvel, 
        "fingertip_pos":fingertip_pos
    }

def get_reward_input_keys(model):
    method = model.compute_reward
    sig = inspect.signature(method)
    return list(sig.parameters.keys())[0:]  # exclude 'self'
    

def get_preference_pairs(data_folder: str):
    filenames = [f for f in os.listdir(data_folder) if f.endswith(".txt")]
    
    # First count lines in each file to determine rollout length
    rollout_lengths = {}
    rollout_scores = {}
    for i, filename in enumerate(filenames):
        with open(os.path.join(data_folder, filename), 'r') as f:
            # f.readline()  # Skip the score line
            # Count the remaining lines which represent the rollout length
            rollout_scores[i] = float(f.readline())
            rollout_lengths[i] = sum(1 for _ in f)
    
    preference_pairs = []
    for i in range(len(filenames)):
        for j in range(i,len(filenames)):
            # Seed is the first number before the first underscore in the filename, if seeds are not equal, skip the pair
            if filenames[i].split("_")[0] != filenames[j].split("_")[0]:
                continue
            if i != j:
                if True:
                    # Prefer the shorter rollout (0 means i is preferred, 1 means j is preferred)
                    if rollout_scores[i] == rollout_scores[j]:
                        # If scores are equal and 1, prefer the shorter rollout
                        # If the scores are equal and 0, prefer the longer rollout
                        # If the lengths are equal, prefer neither
                        if rollout_lengths[i] == rollout_lengths[j]:
                            continue
                        elif rollout_scores[i] == 2:
                            preference_pairs.append((i, j, 0 if rollout_lengths[i] < rollout_lengths[j] else 1))
                        else:
                            preference_pairs.append((i, j, 0 if rollout_lengths[i] > rollout_lengths[j] else 1))
                    elif rollout_scores[i] > rollout_scores[j]:
                        preference_pairs.append((i, j, 0))
                    elif rollout_scores[i] < rollout_scores[j]:
                        preference_pairs.append((i, j, 1))
                else:
                    with open(os.path.join(data_folder, filenames[i]), 'r') as f1:
                        score_i = float(f1.readline())
                        file1_length = len(f1.readlines())
                    with open(os.path.join(data_folder, filenames[j]), 'r') as f2:
                        score_j = float(f2.readline())
                        file2_length = len(f2.readlines())
                    if score_i == score_j:
                        continue
                    # Check the length of both files and discard if they are not the same
                    if file1_length != file2_length:
                        continue
                    preference_pairs.append((i, j, 0 if score_i > score_j else 1))
    return filenames, torch.tensor(preference_pairs, dtype=torch.float32)


def get_rollout_observations(rollout_path, required_keys, max_length=None):
    with open(rollout_path, 'r') as f:
        f.readline()  # Skip score line
        data = [eval(line) for line in f]
    
    # If max_length is specified, truncate the data
    if max_length is not None:
        data = data[:max_length]
        
    data = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    # object_rot_list, goal_rot_list = [], []
    input_dicts = []
    for i in range(data.shape[0]):
        full_vars = return_env_vars(data[i])
        filtered_vars = {k: full_vars[k] for k in required_keys}
        input_dicts.append(filtered_vars)
        # object_rot_list.append(object_rot)
        # goal_rot_list.append(goal_rot)
    # return object_rot_list, goal_rot_list
    return input_dicts

def pairwise_focal_bpr_loss(
    model,
    comparisons,      # LongTensor [N,3] : (i, j, label)   label 0 ⇒ i ≻ j
    filenames,        # list[str]        : rollout files
    data_folder,      # str              : folder path
    lambda_l2=1e-4,   # ℓ² weight on total scores
    tau=5.0,          # temperature for σ
    gamma=2.0,        # focal exponent γ
    verbose_accuracy=False
):
    """
    Loss =  mean( (1-p)^γ · (-log p) )  +  λ · mean(score²),
    where p = σ( y·Δ / τ ),  y∈{±1},  Δ = r_i − r_j.
    Signature is identical to previous loss functions.
    """
    device = next(model.parameters()).device

    # --- 1. rollout lengths ---------------------------------------------------
    rlen = {}
    for idx, fname in enumerate(filenames):
        with open(os.path.join(data_folder, fname), 'r') as f:
            f.readline();  rlen[idx] = sum(1 for _ in f)

    # --- 2. cache observations -----------------------------------------------
    obs_cache = {}
    keys = get_reward_input_keys(model)
    for k, fname in enumerate(filenames):
        obs_cache[k] = get_rollout_observations(os.path.join(data_folder, fname), keys)

    # --- 3. total reward per (rollout, L) ------------------------------------
    total = {}
    for i, j, _ in comparisons.tolist():
        L = min(rlen[i], rlen[j])
        for k in (i, j):
            key = (k, L)
            if key not in total:
                s = torch.tensor(0.0, device=device)
                for inp in obs_cache[k][:L]:
                    r_t, _ = model(**inp)
                    s += r_t.squeeze()
                total[key] = s

    # --- 4. build Δ and labels -----------------------------------------------
    Δ, y, key_batch = [], [], set()
    for i, j, lbl in comparisons.tolist():
        L = min(rlen[i], rlen[j])
        Δ.append((total[(i, L)] - total[(j, L)]).unsqueeze(0))
        y.append(+1.0 if lbl == 0 else -1.0)
        key_batch.update({(i, L), (j, L)})

    Δ = torch.cat(Δ)                    # [N]
    y = torch.tensor(y, device=device)  # [N]

    # --- 5. focal-BPR term ----------------------------------------------------
    logits = y * Δ / tau
    p = torch.sigmoid(logits)
    focal_bpr = ((1.0 - p) ** gamma) * F.softplus(-logits)   # −log σ with focal weight

    # --- 6. score ℓ² regulariser ---------------------------------------------
    score_sq = torch.stack([total[k].pow(2) for k in key_batch])
    loss = focal_bpr.mean() + lambda_l2 * score_sq.mean()

    # --- 7. accuracy (optional) ----------------------------------------------
    with torch.no_grad():
        acc = ((Δ > 0).float() == (y == +1).float()).float().mean()
        print(f"Pairwise accuracy: {acc.item():.4f}")
        if verbose_accuracy:
            for (i, j, lbl) in comparisons.tolist():
                L = min(rlen[i], rlen[j])
                ri = total[(i, L)].item()
                rj = total[(j, L)].item()
                pred = 'i>j' if ri > rj else 'i<j'
                truth = 'i>j' if lbl == 0 else 'i<j'
                tag = '[OK]' if pred == truth else '[WRONG]'
                print(f"{tag} C{i} ({ri:.3f}) vs C{j} ({rj:.3f}); true={truth}")

    return loss

def pairwise_bpr_loss(
    model,
    comparisons,      # LongTensor [N, 3] : (i, j, label) — label 0 ⇒ i ≻ j, 1 ⇒ j ≻ i
    filenames,        # list[str]          : rollout-file names, index = rollout id
    data_folder,      # str               : path to folder containing those files
    lambda_l2=1e-4,   # ℓ² weight on total-score magnitudes
    tau=5.0,          # temperature that prevents logit blow-up
    verbose_accuracy=False
):
    """
    Temperature-controlled BPR loss  +  ℓ² penalty on each rollout’s total score.

    Returns:
        loss =   mean(-log σ(y·Δ / τ))    +   λ · mean(score²)
    """

    device = next(model.parameters()).device

    # ---------- 1. gather rollout lengths ----------
    rollout_len = {}
    for idx, fname in enumerate(filenames):
        with open(os.path.join(data_folder, fname), "r") as f:
            f.readline()                      # skip header
            rollout_len[idx] = sum(1 for _ in f)

    # ---------- 2. cache all observations ----------
    cached_obs = {}
    input_keys = get_reward_input_keys(model)
    for k, fname in enumerate(filenames):
        cached_obs[k] = get_rollout_observations(
            os.path.join(data_folder, fname), input_keys
        )

    # ---------- 3. pre-compute truncated total scores ----------
    total_reward = {}
    for i, j, _ in comparisons.tolist():
        L = min(rollout_len[i], rollout_len[j])
        for k in (i, j):
            key = (k, L)
            if key not in total_reward:
                r_sum = torch.tensor(0.0, device=device)
                for inp in cached_obs[k][:L]:
                    r_t, _ = model(**inp)
                    r_sum += r_t.squeeze()
                total_reward[key] = r_sum

    # ---------- 4. build Δ and targets ----------
    delta, y_label, keys_in_batch = [], [], set()
    for i, j, lbl in comparisons.tolist():
        L = min(rollout_len[i], rollout_len[j])
        ri = total_reward[(i, L)]
        rj = total_reward[(j, L)]
        delta.append((ri - rj).unsqueeze(0))
        y_label.append(+1.0 if lbl == 0 else -1.0)
        keys_in_batch.update({(i, L), (j, L)})

    delta = torch.cat(delta, dim=0)                   # [N]
    y      = torch.tensor(y_label, device=device)     # [N]

    # ---------- 5. BPR (logistic) term ----------
    # Softplus(x) = log(1 + e^x).  Here we want  -log σ(y·Δ/τ) = Softplus(-y·Δ/τ).
    bpr_term = F.softplus(-y * delta / tau)

    # ---------- 6. ℓ² score penalty ----------
    score_sq = torch.stack([total_reward[k].pow(2) for k in keys_in_batch])
    l2_term  = score_sq.mean()

    # ---------- 7. final loss ----------
    loss = bpr_term.mean() + lambda_l2 * l2_term

    # ---------- 8. optional accuracy ----------
    with torch.no_grad():
        preds = (delta > 0).float()                 # 1 if model ranks i > j
        true  = (y == +1).float()                   # 1 if label says i > j
        acc   = (preds == true).float().mean()
        print(f"Pairwise accuracy: {acc.item():.4f}")
        if verbose_accuracy:
            for idx, (i, j, lbl) in enumerate(comparisons.tolist()):
                L = min(rollout_len[i], rollout_len[j])
                ri = total_reward[(i, L)].item()
                rj = total_reward[(j, L)].item()
                pred = "i>j" if ri > rj else "i<j"
                truth = "i>j" if lbl == 0 else "i<j"
                tag = "[OK]" if pred == truth else "[WRONG]"
                print(f"{tag} C{i}({ri:.3f}) vs C{j}({rj:.3f}); true={truth}")

    return loss

def pairwise_smooth_hinge_loss(
    model,
    comparisons,    # LongTensor of shape [N, 3]: (i, j, label), where label=0 means i ≻ j, label=1 means j ≻ i
    filenames,      # List[str] mapping each rollout index to a filename (length = num_rollouts)
    data_folder,    # Path to the folder containing all rollout files
    lambda_l2=1e-4, # Regularization weight on squared rollout‐scores
    verbose_accuracy=False
):
    """
    Compute a smooth‐hinge + L2‐score penalty ranking loss over a batch of pairwise comparisons.

    Args:
      model: A PyTorch module such that `model(**inp) -> (reward, other_out)`,
             where `reward` is a scalar Tensor for that single timestep.
      comparisons: LongTensor [N,3], each row = (i, j, label).
                   If label == 0 => i ≻ j  (target y = +1).
                   If label == 1 => j ≻ i  (target y = -1).
      filenames: List[str] of all rollout filenames (index corresponds to rollout ID).
      data_folder: Directory where each filename lives.
      lambda_l2: Scalar ≥ 0. Penalty coefficient on the mean of squared total‐rollout scores.
      verbose_accuracy: If True, prints per‐comparison correctness at each call.

    Returns:
      loss: Scalar Tensor = (mean smooth‐hinge over all pairs) + (λ × mean(score²) over involved rollouts).
    """

    device = next(model.parameters()).device

    # 1) Compute length (number of steps) of each rollout file:
    rollout_data_full = {}
    for idx, fname in enumerate(filenames):
        path = os.path.join(data_folder, fname)
        with open(path, 'r') as f:
            f.readline()  # skip header (e.g., score line)
            rollout_data_full[idx] = sum(1 for _ in f)

    # 2) Cache all per‐timesteps observations for each rollout once:
    #    cached_obs[k] = list of dicts, each dict is model input for one timestep
    cached_obs = {}
    input_keys = get_reward_input_keys(model)  # helper: returns list of keys model expects

    for k in range(len(filenames)):
        path = os.path.join(data_folder, filenames[k])
        cached_obs[k] = get_rollout_observations(path, input_keys)

    # 3) Precompute each rollout's total score for truncated length = min(len(i), len(j))
    #    rollout_scores[(k, L)] = sum_{t=0..L-1} r(s_t) for rollout k truncated to length L
    rollout_scores = {}
    for (i, j, _) in comparisons.tolist():
        L = min(rollout_data_full[i], rollout_data_full[j])
        for k in (i, j):
            key = (k, L)
            if key not in rollout_scores:
                tot = torch.tensor(0.0, device=device)
                for inp in cached_obs[k][:L]:
                    r_t, _ = model(**inp)
                    tot = tot + r_t.squeeze()
                rollout_scores[key] = tot

    # 4) Build tensors of Δ = r(i) - r(j) and targets y ∈ {+1, -1}
    deltas = []
    targets = []
    minibatch_keys = set()  # track unique (k, L) used, for the L2 penalty

    for (i, j, lbl) in comparisons.tolist():
        L = min(rollout_data_full[i], rollout_data_full[j])
        ri = rollout_scores[(i, L)]
        rj = rollout_scores[(j, L)]
        deltas.append((ri - rj).unsqueeze(0))  # shape [1]
        y = +1.0 if (lbl == 0) else -1.0
        targets.append(y)
        minibatch_keys.add((i, L))
        minibatch_keys.add((j, L))

    delta_tensor = torch.cat(deltas, dim=0)                # shape [N]
    target_tensor = torch.tensor(targets, device=device)   # shape [N], dtype float

    # 5) Smooth‐hinge loss: log(1 + exp(-y * Δ))
    #    (Equivalent to F.softplus(-y * Δ))
    smooth_hinge = F.softplus(- target_tensor * delta_tensor)  # shape [N]

    # 6) L2 penalty on each unique total‐rollout score in this batch
    score_squares = []
    for key in minibatch_keys:
        score_squares.append( rollout_scores[key].pow(2).unsqueeze(0) )
    all_scores_sq = torch.cat(score_squares, dim=0)  # shape [num_unique_keys]

    # 7) Combine: mean(smooth_hinge) + λ * mean(score²)
    loss_hinge = smooth_hinge.mean()
    loss_l2    = all_scores_sq.mean()
    loss = loss_hinge + lambda_l2 * loss_l2

    # 8) Optional: compute & print pairwise accuracy
    with torch.no_grad():
        preds = (delta_tensor > 0).float()                     # 1 if i ≻ j predicted
        true_labels = (target_tensor == +1.0).float()           # 1 if i ≻ j true
        acc = (preds == true_labels).float().mean()
        print(f"Pairwise accuracy: {acc.item():.4f}")

        if verbose_accuracy:
            for idx, (i, j, lbl) in enumerate(comparisons.tolist()):
                L = min(rollout_data_full[i], rollout_data_full[j])
                ri = rollout_scores[(i, L)].item()
                rj = rollout_scores[(j, L)].item()
                pred_str = "i>j" if ri > rj else "i<j"
                true_str = "i>j" if lbl == 0 else "i<j"
                tag = "[OK]" if (pred_str == true_str) else "[WRONG]"
                print(f"{tag} C{i} ({ri:.3f}) vs C{j} ({rj:.3f}); true={true_str}")

    return loss

def bradley_terry_margin_loss(
    model,
    comparisons,    # Tensor of shape [N, 3]: (i, j, label) where label=0 means i ≻ j, label=1 means j ≻ i
    filenames,      # List[str], length = number of rollouts
    data_folder,    # path to folder containing rollout files
    margin=1.0,
    verbose_accuracy=False
):
    """
    Compute a margin-based pairwise ranking loss over rollouts.

    Args:
      model: A PyTorch module such that `model(**inp) -> (reward, other_output)`,
             where `reward` is a scalar tensor for that single observation.
      comparisons: LongTensor of shape [N, 3], each row = (i, j, label).
                   If label == 0, rollout i is preferred over j; if label == 1, j ≻ i.
      filenames: List of strings, mapping each rollout index to a filename in data_folder.
      data_folder: Path where rollout files live.
      margin: The margin in the hinge loss.
      verbose_accuracy: If True, print per‐pair correct/incorrect.

    Returns:
      loss: the scalar margin‐based ranking loss.
    """

    # 1) Determine the length (number of lines) of each rollout file:
    rollout_data_full = {}
    for idx, fname in enumerate(filenames):
        path = os.path.join(data_folder, fname)
        with open(path, 'r') as f:
            # skip first line (score or metadata), then count the remaining lines
            f.readline()
            rollout_data_full[idx] = len([_ for _ in f])

    # 2) Cache per‐rollout observation sequences (so we don't reload repeatedly):
    #    cached_observations[k] = list of "input dicts" up to full length
    cached_observations = {}
    input_keys = get_reward_input_keys(model)  # user‐provided helper that returns obs→model inputs

    for k in range(len(filenames)):
        path = os.path.join(data_folder, filenames[k])
        # load all observations for rollout k into a list once
        cached_observations[k] = get_rollout_observations(path, input_keys)

    # 3) Precompute total reward for each (rollout index, truncated length) pair:
    #    rollout_rewards[(idx, L)] = sum_{t=0..L-1} r(s_t)
    rollout_rewards = {}
    for i, j, _ in comparisons.tolist():
        # find the shorter rollout length between i and j
        L = min(rollout_data_full[i], rollout_data_full[j])
        for k in (i, j):
            key = (k, L)
            if key not in rollout_rewards:
                # sum model(**inp) over the first L timesteps
                total_reward = torch.tensor(0.0, device=next(model.parameters()).device)
                for inp in cached_observations[k][:L]:
                    reward, _ = model(**inp)
                    # reward is assumed to be a 0‐dim tensor or shape [1]; squeeze to scalar
                    total_reward = total_reward + reward.squeeze()
                rollout_rewards[key] = total_reward

    # 4) Build "left" and "right" score vectors for each comparison pair:
    #    left_scores[n]  = r(rollout_i truncated to L)
    #    right_scores[n] = r(rollout_j truncated to L)
    left_scores = []
    right_scores = []
    labels = []  # 0 if i ≻ j, 1 if j ≻ i

    for (i, j, label) in comparisons.tolist():
        L = min(rollout_data_full[i], rollout_data_full[j])
        left_scores.append( rollout_rewards[(i, L)].unsqueeze(0) )   # shape [1]
        right_scores.append( rollout_rewards[(j, L)].unsqueeze(0) )  # shape [1]
        labels.append(label)

    # Stack into tensors of shape [N]
    left_tensor  = torch.cat(left_scores, dim=0)    # [N]
    right_tensor = torch.cat(right_scores, dim=0)   # [N]
    labels_tensor = torch.tensor(labels, device=left_tensor.device, dtype=torch.long)  # [N]

    # 5) Convert original labels (0/1) → margin targets (+1 or -1):
    #    If label == 0, that means i ≻ j  ⇒ target = +1   (so we want left_tensor > right_tensor + margin)
    #    If label == 1, that means j ≻ i  ⇒ target = -1   (we want right_tensor > left_tensor + margin)
    targets = torch.where(labels_tensor == 0,
                          torch.ones_like(labels_tensor, dtype=torch.float),
                          -torch.ones_like(labels_tensor, dtype=torch.float))  # shape [N], type float

    # 6) Compute margin‐based hinge loss:
    loss_fn = nn.MarginRankingLoss(margin=margin)
    loss = loss_fn(left_tensor, right_tensor, targets)

    # 7) (Optional) Compute & print pairwise accuracy:
    with torch.no_grad():
        # Predicted ordering: left > right  ⇒ predicted i ≻ j
        pred_i_better = (left_tensor > right_tensor).float()  # [N], 1.0 if left>right else 0.0
        true_i_better = torch.where(labels_tensor == 0,
                                    torch.ones_like(labels_tensor, dtype=torch.float),
                                    torch.zeros_like(labels_tensor, dtype=torch.float))
        accuracy = (pred_i_better == true_i_better).float().mean()
        print(f"Pairwise accuracy (i ≻ j): {accuracy.item():.4f}")

        if verbose_accuracy:
            # Optionally, log each pair’s correctness
            for idx, (i, j, label) in enumerate(comparisons.tolist()):
                r_i = left_tensor[idx].item()
                r_j = right_tensor[idx].item()
                pred_order = "i>j" if r_i > r_j else "i<j"
                true_order = "i>j" if label == 0 else "i<j"
                if pred_order == true_order:
                    print(f"[OK]   C{i} ({r_i:.3f}) vs C{j} ({r_j:.3f}); true={true_order}")
                else:
                    print(f"[WRONG] C{i} ({r_i:.3f}) vs C{j} ({r_j:.3f}); true={true_order}")

    return loss

def bradley_terry_loss(model, comparisons, filenames, data_folder, verbose_accuracy=False):
    loss_fn = nn.CrossEntropyLoss()
    input_keys = get_reward_input_keys(model)
    
    # First load all rollout data
    rollout_data_full = {}
    for i, path in enumerate(filenames):
        with open(os.path.join(data_folder, path), 'r') as f:
            f.readline()  # Skip score line
            rollout_data_full[i] = len([line for line in f])
    
    rollout_rewards = {}
    cached_observations = {}
    for idx in range(len(comparisons)):
        i, j = int(comparisons[idx, 0]), int(comparisons[idx, 1])
        min_length = min(rollout_data_full[i], rollout_data_full[j])

        for k in [i, j]:
            if k not in cached_observations:
                # Cache the full observation sequence
                cached_observations[k] = get_rollout_observations(os.path.join(data_folder, filenames[k]), input_keys)
            
            key = (k, min_length)
            if key not in rollout_rewards:
                inputs = cached_observations[k][:min_length]
                total_reward = torch.tensor(0.0, requires_grad=True)
                for inp in inputs:
                    reward, _ = model(**inp)
                    total_reward = total_reward + reward
                rollout_rewards[key] = total_reward

    left = torch.stack([
        rollout_rewards[(int(row[0]), min(rollout_data_full[int(row[0])], rollout_data_full[int(row[1])]))].squeeze()
        for row in comparisons
    ])
    right = torch.stack([
        rollout_rewards[(int(row[1]), min(rollout_data_full[int(row[0])], rollout_data_full[int(row[1])]))].squeeze()
        for row in comparisons
    ])

    logits = torch.stack([left, right], dim=1)
    targets = comparisons[:, -1].long()

    # Handle tri-state targets: 0/1 => CE as before; 2 => MSE(left, right)
    device = left.device
    mask_tie = (targets == 2)
    mask_ce = ~mask_tie

    ce_loss = torch.tensor(0.0, device=device)
    mse_loss = torch.tensor(0.0, device=device)
    n_ce = int(mask_ce.sum().item())
    n_mse = int(mask_tie.sum().item())

    if n_ce > 0:
        ce_loss = loss_fn(logits[mask_ce], targets[mask_ce])  # mean over 0/1 samples
    if n_mse > 0:
        mse_loss = F.mse_loss(left[mask_tie], right[mask_tie], reduction='mean')  # mean over tie samples

    if (n_ce + n_mse) > 0:
        base_loss = (ce_loss * n_ce + mse_loss * n_mse) / (n_ce + n_mse)
    else:
        base_loss = torch.tensor(0.0, device=device)

    # L2 regularization on reward magnitudes to prevent scaling tricks
    reward_l2_penalty = (left.pow(2) + right.pow(2)).mean()
    lambda_l2 = 0.01  # Regularization strength - tune this parameter
    
    total_loss = base_loss + lambda_l2 * reward_l2_penalty

    with torch.no_grad():
        if n_ce > 0:
            predictions = torch.argmax(logits[mask_ce], dim=1)
            acc = (predictions == targets[mask_ce]).float().mean()
            print(f"Pairwise accuracy: {acc.item():.2f}, Base loss: {base_loss.item():.4f}, L2 penalty: {reward_l2_penalty.item():.4f}")
        else:
            print(f"Pairwise accuracy: N/A (only ties), Base loss: {base_loss.item():.4f}, L2 penalty: {reward_l2_penalty.item():.4f}")

    if verbose_accuracy:
        if TRACK_FAILURES:
            failure_per_idx = defaultdict(int)

        for i in range(len(comparisons)):
            left_idx = int(comparisons[i, 0])
            right_idx = int(comparisons[i, 1])
            preference = int(comparisons[i, 2])

            model_rewards = torch.stack([left[i], right[i]])
            if preference == 0:
                if model_rewards[0] > model_rewards[1]:
                    if LOG_SUCCESS:
                        print(f"Correct: {filenames[left_idx]} ({model_rewards[0]:.4f}) > {filenames[right_idx]} ({model_rewards[1]:.4f})")
                else:
                    if LOG_FAILURES:
                        print(f"Incorrect: {filenames[left_idx]} ({model_rewards[0]:.4f}) < {filenames[right_idx]} ({model_rewards[1]:.4f})")   
                    if TRACK_FAILURES:
                        failure_per_idx[left_idx] += 1
            elif preference == 1:
                if model_rewards[0] < model_rewards[1]:
                    if LOG_SUCCESS:
                        print(f"Correct: {filenames[left_idx]} ({model_rewards[0]:.4f}) < {filenames[right_idx]} ({model_rewards[1]:.4f})")
                else:
                    if LOG_FAILURES:
                        print(f"Incorrect: {filenames[left_idx]} ({model_rewards[0]:.4f}) > {filenames[right_idx]} ({model_rewards[1]:.4f})")
                    if TRACK_FAILURES:
                        failure_per_idx[right_idx] += 1
            else:
                # preference == 2 (tie): no failure counting; optional logging only
                if LOG_SUCCESS:
                    print(f"Tie: {filenames[left_idx]} ({model_rewards[0]:.4f}) ~ {filenames[right_idx]} ({model_rewards[1]:.4f})")

        if TRACK_FAILURES:
            # Iterate over all the files and add the failures for those that failed
            for i in range(len(filenames)):
                FAILURE_TRACK_PROGRESS[i].append(failure_per_idx[i])
            print("Failure tracking:")
            for i in FAILURE_TRACK_PROGRESS:
                print(f"{filenames[i]}: {FAILURE_TRACK_PROGRESS[i]}")

    return total_loss

def wrap_reward_module(code_string: str, param_names: dict, module_name="DynamicReward"):
    import_lines = []
    method_lines = []

    for line in code_string.strip().splitlines():
        if line.strip().startswith("import") or line.strip().startswith("from"):
            import_lines.append(line.strip())
        else:
            method_lines.append(line.rstrip())

    # Find the method definition (e.g., def compute_reward(self, ...))
    method_header_index = next(
        (i for i, line in enumerate(method_lines) if line.strip().startswith("def compute_reward")), -1
    )
    if method_header_index == -1:
        raise ValueError("Expected a method named `compute_reward`.")

    method_def = method_lines[method_header_index]
    method_def_line = method_lines[method_header_index].strip()

    method_args = method_def_line[
        method_def_line.index("(") + 1:method_def_line.index(")")
    ].split(",")
    method_args = [arg.strip() for arg in method_args if arg.strip() and arg.strip() != "self"]

    method_body = method_lines[method_header_index + 1:]
    indented_method = [f"    {line}" if line.strip() else "" for line in [method_def] + method_body]

    param_init = "\n".join(
        f"        self.{k} = nn.Parameter(torch.tensor({v}, dtype=torch.float32))"
        for k, v in param_names.items()
    )

    for i in range(len(method_args)):
        # If it has a colon remove it
        if ":" in method_args[i]:
            method_args[i] = method_args[i].split(":")[0].strip()

    # Create a string where it says arg=inputs[arg] for each arg
    arg_assignments = [f"{arg}= inputs['{arg}']" for arg in method_args]
    # Convert arg_assignments to a string with no surrounding brackets
    arg_assignments = ", ".join(arg_assignments)
    class_code = f"""
import torch
import torch.nn as nn
{chr(10).join(import_lines)}

class {module_name}(nn.Module):
    def __init__(self):
        super().__init__()
{param_init}

{chr(10).join(indented_method)}

    def forward(self, **inputs):
        return self.compute_reward({arg_assignments})
"""
    return class_code





def create_model_from_code(code_str: str, param_defaults: dict):
    class_code = wrap_reward_module(code_str, param_defaults)

    # DEBUG: Optional - save generated module
    print(class_code)

    # Use a shared dictionary for globals so imports persist
    exec_scope = {}
    exec(class_code, exec_scope)
    return exec_scope["DynamicReward"]()



def train_reward_model(code_str: str, param_defaults: dict, data_folder: str, epochs=20, lr=5e-2):
    code_str = code_str.replace("-> Tuple[torch.Tensor, Dict[str, torch.Tensor]]","")
    code_str = code_str.replace("compute_reward(", "compute_reward(self,")
    model = create_model_from_code(code_str, param_defaults)
    filenames, comparisons = get_preference_pairs(data_folder)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Set torch randperm seed for reproducibility
    torch.manual_seed(0)
    
    # Split rollouts first to prevent data leakage
    num_rollouts = len(filenames)
    rollout_indices = torch.randperm(num_rollouts)
    val_size = int(num_rollouts * 0.2)
    val_rollouts = set(rollout_indices[:val_size].tolist())
    train_rollouts = set(rollout_indices[val_size:].tolist())
    
    # Split comparisons based on rollout membership
    validation_comparisons = []
    train_comparisons = []
    
    for comparison in comparisons:
        i, j = int(comparison[0]), int(comparison[1])
        # Only add to validation if BOTH rollouts are in validation set
        if i in val_rollouts and j in val_rollouts:
            validation_comparisons.append(comparison)
        # Only add to training if BOTH rollouts are in training set
        elif i in train_rollouts and j in train_rollouts:
            train_comparisons.append(comparison)
        # Skip mixed pairs that would cause data leakage
    
    validation_comparisons = torch.stack(validation_comparisons) if validation_comparisons else torch.empty(0, 3)
    comparisons = torch.stack(train_comparisons) if train_comparisons else torch.empty(0, 3)
    
    print(f"Training comparisons: {len(comparisons)}, Validation comparisons: {len(validation_comparisons)}")

    # input_keys = get_reward_input_keys(model)
    # rollout_data = {
    #     i: get_rollout_observations(os.path.join(data_folder, path), input_keys)
    #     for i, path in enumerate(filenames)
    # }
    

    # print(f"Initial Loss: {bradley_terry_loss(model, comparisons, filenames, data_folder, verbose_accururacy=True)}")
    for i in range(epochs):
        optimizer.zero_grad()
        loss = bradley_terry_loss(model, comparisons, filenames, data_folder, verbose_accuracy=0)
        # If MAXIMIZE_LOSS is True, we need to negate the loss
        if MAXIMIZE_LOSS:
            loss = -loss
        loss.backward()
        optimizer.step()
        # Calculate the validation loss
        with torch.no_grad():
            val_loss = bradley_terry_loss(model, validation_comparisons, filenames, data_folder)
            # print(f"Validation Loss: {val_loss.item():.4f}")
        print(f"Epoch {i+1}/{epochs}, Train Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

        if i % 10 == 0:
            print("Learned parameters:")
            for name, param in model.named_parameters():
                print(f"{name}: {param.item()}")

    print("Learned parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.item()}")

    return model


# Example when running script directly
if __name__ == "__main__":
#     reward_code = '''
# import torch
# from typing import Dict, Tuple

# def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, rot_diff_temp: torch.Tensor, success_threshold: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#     rot_diff = torch.sum((object_rot - goal_rot) ** 2, dim=-1)
#     rot_diff_reward = torch.exp(-rot_diff_temp * rot_diff)
#     success_reward = (rot_diff < success_threshold).float()
#     total_reward = rot_diff_reward + success_reward
#     return total_reward, {"rot_diff_reward": rot_diff_reward, "success_reward": success_reward}
# '''

    reward_code_simple = '''
def compute_reward(object_rot, goal_rot):
    # compute the cosine similarity between object's current orientation and the target orientation
    dist_penalty_scaler = self.dist_penalty_scaler
    reward_temp = self.reward_temp
    garbage_term_scaler = self.garbage_term_scaler
    survival_scaler = self.survival_scaler

    similarity = torch.nn.functional.cosine_similarity(object_rot, goal_rot, dim=-1)

    # transform similarity to a distance-like metric
    distance = 1 - similarity

    # dist_penalty_scaler = self.dist_penalty_scaler # This should learn to be negative if the algo works

    # larger reward the smaller the rotation difference
    reward = dist_penalty_scaler * distance #LLM had -1 instead of the dist_penalty_scaler

    # temperature parameter adjusted for reward scaling
    # reward_temp = reward_temp # LLM set this to 1
    reward_temp = torch.clamp(reward_temp, min=0.5, max=15) # prevent explosion

    # scale the raw reward using an exponential function
    # scaled_reward = torch.exp(torch.clamp(reward / reward_temp, min=-50, max=50)) # prevent gradient explosion NECESSARY
    # safe_reward = reward / reward_temp
    # safe_reward = safe_reward.clone().detach().requires_grad_(True)  # Prevent in-place issues
    # scaled_reward = torch.nn.functional.softplus(torch.clamp(safe_reward, min=-10, max=10)) # torch.exp had grad issues
    safe_reward = reward / reward_temp  # this keeps gradient flow
    safe_reward = torch.clamp(safe_reward, min=-10, max=10)
    scaled_reward = torch.nn.functional.softplus(safe_reward)

    # Survival Reward, just adds a constant equal to the parameter, encourages longer rollouts
    survival_reward = self.survival_scaler
    # for now try with a pure noise reward
    # noise_reward = torch.randn(1) * garbage_term_scaler
    scaled_reward += survival_reward
    scaled_reward += garbage_term_scaler * torch.randn_like(scaled_reward) # Ideally this gets silenced as well?

    return scaled_reward, {}
'''

    reward_code = '''
def compute_reward(object_rot: torch. Tensor, goal_rot: torch. Tensor, object_angvel: torch. Tensor, object_pos: torch. Tensor, fingertip_pos: torch.Tensor):
    
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
    return total_reward, reward_components'''

    param_defaults_simple = {
        "dist_penalty_scaler": -0.9,
        "reward_temp": 1.0,
        "garbage_term_scaler": 0.001,
        "survival_scaler": 0.1,
    }

    param_defaults = {
        "rotation_reward_temp": 13.257881164550781,
        "angvel_threshold": 8.498954772949219,
        "angvel_penalty_temp": 5.115146160125732,
        "min_distance_temp": 3.902719497680664
    }
    

    model = train_reward_model(
        code_str=reward_code,
        param_defaults=param_defaults,
        data_folder="./preference_data",
        epochs=45,
        lr=0.5
    )
    print("Done")
