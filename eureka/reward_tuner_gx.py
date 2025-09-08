import os
import ast
import time
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


# =====================
# Configuration
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS = 150
DT_CONST = 0.0166
OBS_DIM = 16  # default; will be overridden dynamically based on data


# =====================
# New Reward Architecture
# =====================
class FourierFeatureLayer(nn.Module):
    """
    Random Fourier features to enrich input representation.
    x -> concat[x, sin(Bx), cos(Bx)]
    """
    def __init__(self, in_dim: int, num_frequencies: int = 32, scale: float = 3.0):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.register_buffer("B", torch.randn(in_dim, num_frequencies) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, in_dim]
        proj = x @ self.B  # [N, F]
        return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)


class GatedResidualBlock(nn.Module):
    """
    Residual MLP block with gating and LayerNorm for stability.
    Inspired by gated MLPs and Pre-LN Transformers.
    """
    def __init__(self, dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.gate = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.fc1(self.ln(x))
        h = self.act(h)
        h = self.drop(self.fc2(h))
        g = torch.sigmoid(self.gate(x))
        return residual + g * h


class StepRewardNet(nn.Module):
    """
    Per-step reward network r_theta(s_t, a_t) with enriched features and
    gated residual mixing. Outputs a scalar per step.
    """
    def __init__(self, in_dim: int = OBS_DIM, width: int = 128, depth: int = 5, fourier_features: int = 48):
        super().__init__()
        self.ff = FourierFeatureLayer(in_dim, num_frequencies=fourier_features)
        stem_dim = in_dim + 2 * fourier_features
        self.stem = nn.Sequential(
            nn.Linear(stem_dim, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        blocks = []
        for _ in range(depth):
            blocks.append(GatedResidualBlock(width, hidden=width * 2, dropout=0.1))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 1),
            nn.Tanh(),  # bound rewards for stability
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, OBS_DIM]
        z = self.ff(x)
        z = self.stem(z)
        z = self.blocks(z)
        r = self.head(z)  # [N, 1]
        return r


# =====================
# Data I/O
# =====================
def _safe_parse_vector(line: str) -> List[float]:
    """
    Parse a line expected to be in the form [[...]] and return the inner list.
    If the line is not in that form (e.g., [-123.4]), raise ValueError.
    """
    arr = ast.literal_eval(line)
    if isinstance(arr, list) and len(arr) == 1 and isinstance(arr[0], list):
        return [float(v) for v in arr[0]]
    raise ValueError("Expected line like [[...]]")


def load_ant_trajectory(path: str, max_steps: int = MAX_STEPS) -> torch.Tensor:
    """
    Load Ant rollout into a [T, D] tensor.
    - For sanitized data: D=16 (7 root features + 8 actions + 1 dt)
    - For body data: D=22 (13 root state + 8 actions + 1 dt)
    The parser is robust to stray scalar lines between sections and ignores them.
    """
    with open(path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f]

    # Locate section headers
    try:
        rs_idx = next(i for i, l in enumerate(lines) if l.strip() == "Root States:")
        act_idx = next(i for i, l in enumerate(lines) if l.strip() == "Actions:")
    except StopIteration:
        raise ValueError(f"Missing sections in {path}")

    # Parse sequences
    root_list: List[List[float]] = []
    for ln in lines[rs_idx + 1 : act_idx]:
        if not ln.strip():
            continue
        # Some datasets (ant_data_body) contain occasional scalar lines like [-60043.12]
        # between sections. Ignore any line that is not of shape [[...]].
        try:
            vec = _safe_parse_vector(ln)
        except Exception:
            continue
        root_list.append(vec)

    act_list: List[List[float]] = []
    for ln in lines[act_idx + 1 : ]:
        if not ln.strip():
            continue
        try:
            vec = _safe_parse_vector(ln)
        except Exception:
            continue
        act_list.append(vec)

    T = min(len(root_list), len(act_list), max_steps)
    if T <= 0:
        raise ValueError(f"Empty trajectory in {path}")

    rows: List[List[float]] = []
    for t in range(T):
        rows.append(root_list[t] + act_list[t] + [DT_CONST])

    return torch.tensor(rows, dtype=torch.float32)


def read_ant_score(path: str) -> float:
    with open(path, "r") as f:
        first = f.readline().strip()
    if not first.startswith("Mean Success:"):
        raise ValueError(f"Malformed first line in {path}: {first}")
    return float(first.split(":")[1].strip())


def build_pairs_from_scores(root: str, files: List[str]) -> torch.Tensor:
    """
    Pair rollouts with the same seed prefix; label prefers higher Mean Success.
    Returns [N,3] (i, j, label) as float tensor.
    """
    scores: Dict[int, float] = {}
    for i, fn in enumerate(files):
        try:
            scores[i] = read_ant_score(os.path.join(root, fn))
        except Exception:
            continue

    triplets: List[Tuple[int, int, int]] = []
    for i in range(len(files)):
        if i not in scores:
            continue
        seed_i = files[i].split("_")[0]
        for j in range(i + 1, len(files)):
            if j not in scores:
                continue
            if files[j].split("_")[0] != seed_i:
                continue
            if scores[i] == scores[j]:
                continue
            lab = 0 if scores[i] > scores[j] else 1
            triplets.append((i, j, lab))

    if len(triplets) == 0:
        raise RuntimeError("No preference pairs could be constructed.")
    return torch.tensor(triplets, dtype=torch.float32)


def load_all_features(root: str, files: List[str]) -> Dict[int, torch.Tensor]:
    """Load all trajectories and keep tensors on CPU to avoid CUDA init issues."""
    feats: Dict[int, torch.Tensor] = {}
    for idx, fn in enumerate(files):
        path = os.path.join(root, fn)
        try:
            x = load_ant_trajectory(path)
            feats[idx] = x  # keep on CPU; move on demand during training
        except Exception:
            continue
    return feats


# =====================
# Loss and Evaluation
# =====================
def bradley_terry_logistic_loss(model: nn.Module, feats: Dict[int, torch.Tensor], pairs: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    Preference loss: P(i preferred) = sigmoid(sum r(x_i) - sum r(x_j)).
    Returns (loss, accuracy)
    """
    # Use the model's current device consistently
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")

    if pairs.numel() == 0:
        return torch.tensor(0.0, device=model_device, requires_grad=True), 0.0

    logits: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    for row in pairs:
        i = int(row[0].item()); j = int(row[1].item()); lab = int(row[2].item())
        if i not in feats or j not in feats:
            continue
        xi = feats[i]; xj = feats[j]
        T = min(xi.shape[0], xj.shape[0])
        if T <= 0:
            continue
        xi_t = xi[:T].to(model_device)
        xj_t = xj[:T].to(model_device)
        si = model(xi_t).sum()
        sj = model(xj_t).sum()
        logits.append(si - sj)
        targets.append(torch.tensor(float(1 if lab == 0 else 0), device=model_device))

    if not logits:
        return torch.tensor(0.0, device=model_device, requires_grad=True), 0.0

    logit_vec = torch.stack(logits)
    target_vec = torch.stack(targets)
    loss = nn.BCEWithLogitsLoss()(logit_vec, target_vec)
    with torch.no_grad():
        acc = ((logit_vec > 0).float() == target_vec).float().mean().item()
    return loss, acc


# =====================
# Training Loop
# =====================
def train_preference_reward(
    data_dir: str,
    epochs: int = 30,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 4096,
    val_size: int = 8192,
    grad_clip: float = 1.0,
    log_every: int = 1,
    seed: int = 0,
):
    # Seeding
    if seed is not None:
        torch.manual_seed(seed)

    device = DEVICE

    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.txt')])
    if not files:
        raise RuntimeError(f"No .txt files under {data_dir}")

    print("Loading trajectories...")
    t0 = time.time()
    feats = load_all_features(data_dir, files)
    print(f"Loaded {len(feats)}/{len(files)} files in {time.time()-t0:.1f}s")

    print("Constructing preference pairs...")
    pairs = build_pairs_from_scores(data_dir, files)
    mask = torch.tensor([(int(i) in feats and int(j) in feats) for i, j in pairs[:, :2]], dtype=torch.bool)
    pairs = pairs[mask]
    print(f"Usable pairs: {pairs.shape[0]}")
    if pairs.shape[0] == 0:
        raise RuntimeError("No usable pairs for training.")

    # Shuffle and split
    perm = torch.randperm(pairs.shape[0])
    pairs = pairs[perm]
    v = min(val_size, max(1024, pairs.shape[0] // 10))
    val_pairs = pairs[:v].to(device)
    trn_pairs = pairs[v:]

    # Infer input dimension dynamically from loaded features
    try:
        obs_dim = next(iter(feats.values())).shape[1]
    except StopIteration:
        raise RuntimeError("No features loaded; cannot infer input dimension.")

    # Model
    model = StepRewardNet(in_dim=obs_dim, width=128, depth=5, fourier_features=48).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    # Paths
    ckpt_last_train = os.path.join(data_dir, "Ant_nn_train_last.ptt")
    ckpt_best_train = os.path.join(data_dir, "Ant_nn_train_best.ptt")
    ckpt_last_ts = os.path.join(data_dir, "Ant_nn_checkpoint_last.ptt")
    ckpt_best_ts = os.path.join(data_dir, "Ant_nn_checkpoint_best.ptt")  # used by env loader
    ckpt_periodic_tpl = os.path.join(data_dir, "Ant_nn_train_ep{:03d}.ptt")
    weights_log = os.path.join(data_dir, "Ant_nn_weights_log.txt")

    best_val = float('inf')
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        # Sample batch of pairs
        if batch_size is None or trn_pairs.shape[0] <= batch_size:
            batch = trn_pairs
        else:
            idx = torch.randperm(trn_pairs.shape[0])[:batch_size]
            batch = trn_pairs[idx]

        optimizer.zero_grad(set_to_none=True)
        loss, acc = bradley_terry_logistic_loss(model, feats, batch)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            vloss, vacc = bradley_terry_logistic_loss(model, feats, val_pairs)

        if ep % log_every == 0:
            print(f"Epoch {ep:03d}/{epochs} | train_loss={loss.item():.4f} acc={acc:.3f} | val_loss={vloss.item():.4f} acc={vacc:.3f}")

        # Save train-state checkpoint (for potential future fine-tuning)
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": ep,
            "val_loss": vloss.item(),
            "val_acc": vacc,
        }, ckpt_last_train)

        # Save a periodic checkpoint every 5 epochs for later reuse
        if ep % 5 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": ep,
                "val_loss": vloss.item(),
                "val_acc": vacc,
            }, ckpt_periodic_tpl.format(ep))

        # Export TorchScript snapshot (for deployment)
        try:
            example = torch.randn(4, obs_dim, device=next(model.parameters()).device)
            scripted = torch.jit.trace(model, example)
            scripted.save(ckpt_last_ts)
        except Exception:
            pass

        if vloss.item() < best_val:
            best_val = vloss.item()
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({
                "model_state_dict": best_state,
                "epoch": ep,
                "val_loss": best_val,
                "val_acc": vacc,
            }, ckpt_best_train)
            # Also save best TorchScript for env consumption
            try:
                example = torch.randn(4, obs_dim, device=next(model.parameters()).device)
                scripted = torch.jit.trace(model, example)
                scripted.save(ckpt_best_ts)
            except Exception:
                pass

        scheduler.step()

    # Log weights summary for best state (human-readable)
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    with open(weights_log, "w") as wf:
        wf.write("Ant StepRewardNet Weights Summary\n")
        wf.write(f"Best val loss: {best_val:.6f}\n\n")
        for name, tensor in best_state.items():
            p = tensor.float()
            mean = p.mean().item() if p.numel() > 0 else 0.0
            std = p.std().item() if p.numel() > 1 else 0.0
            norm = p.norm().item() if p.numel() > 0 else 0.0
            wf.write(f"{name}: shape={tuple(p.shape)}, mean={mean:.6f}, std={std:.6f}, norm={norm:.6f}\n")

    print("Training complete.")
    print(f"Saved train checkpoints: {ckpt_last_train}, {ckpt_best_train}")
    print(f"Saved TorchScript models: {ckpt_last_ts}, {ckpt_best_ts}")
    print(f"Saved weights log: {weights_log}")

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preference-based reward model training (Ant)")
    parser.add_argument("--data-dir", type=str, default=os.path.join(os.path.dirname(__file__), "ant_data_body_sanitized"))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_preference_reward(
        data_dir=args.data_dir,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        val_size=args.val_size,
        grad_clip=args.grad_clip,
        log_every=1,
        seed=args.seed,
    )
