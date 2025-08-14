import torch
import torch.nn as nn

OBS_DIM = 31

class MLPReward(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        return self.net(obs_seq)

def _extract_state_dict(ckpt: dict) -> dict:
    """
    Return the first value that *looks* like a real state-dict.
    Extend the candidate list if your project uses a different key.
    """
    for key in ("state_dict", "model_state_dict", "net", "model"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    # If someone saved the *entire* scripted model, you can just return it
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict()
    raise KeyError("No state_dict-like entry found in checkpoint")

def convert_pth_to_pt(pth_path: str,
                      pt_path: str,
                      example_input: torch.Tensor,
                      use_trace: bool = True) -> None:
    model = MLPReward(OBS_DIM)

    # ---- load checkpoint ----
    raw_ckpt = torch.load(pth_path, map_location="cpu")
    state_dict = _extract_state_dict(raw_ckpt)
    model.load_state_dict(state_dict, strict=True)   # strict=False if you *really* want to ignore extras
    model.eval()

    # ---- script/trace ----
    ts_model = (
        torch.jit.trace(model, example_input) if use_trace
        else torch.jit.script(model)
    )

    # ---- save ----
    ts_model.save(pt_path)
    print(f"TorchScript model saved ➜  {pt_path}")

if __name__ == "__main__":
    dummy = torch.randn(1, OBS_DIM)      # (batch, features)  ⇒ shape (1, 13)
    convert_pth_to_pt(
        "/home/avidavid/Downloads/checkpoint_epoch_40.pth",
        "/home/avidavid/Eureka/eureka/checkpoint_epoch_40.pt",
        dummy,
        use_trace=True,
    )
