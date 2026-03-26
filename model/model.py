"""
model.py — WeatherCNN definition + get_model() loader.

Called by evaluate.py:
    model = get_model(metadata)
    pred  = model(x.unsqueeze(0))   # x: (450, 449, C) float32  →  pred: (B, 6)

The wrapper handles:
  1. NaN replacement
  2. Spatial crop (same window used during training)
  3. Input normalisation
  4. Continuous output de-normalisation
"""

import os
import torch
import torch.nn as nn

# ── Paths (override via env variables if needed) ──────────────────────────────
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/cluster/tufts/c26sp1cs0137/ashen05/best_model_20260324_210752.pt",
    
)
NORM_PATH = os.getenv(
    "NORM_PATH",
    "/cluster/tufts/c26sp1cs0137/ashen05/norm_stats.pt",
)

# ── Spatial crop parameters (must match cnn.py) ───────────────────────────────
CROP_H, CROP_W = 352, 352
CROP_R0        = max(0, 225 - CROP_H // 2)   # 49
CROP_C0        = max(0, 200 - CROP_W // 2)   # 24


# ── Model building blocks ─────────────────────────────────────────────────────
class _ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm → ReLU, with optional stride-2 downsampling."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _ResBlock(nn.Module):
    """Two Conv-BN-ReLU layers with a residual skip connection.
    A 1×1 projection is added whenever channels or spatial size changes."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = _ConvBnRelu(in_ch, out_ch, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if (stride != 1 or in_ch != out_ch)
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)) + self.shortcut(x))


class WeatherCNN(nn.Module):
    """
    Encoder-only ResNet CNN for weather scalar regression.

    Input : (B, C, CROP_H, CROP_W)  normalised spatial crop
    Output: cont (B, 6)  continuous predictions (normalised space)
            bin  (B,)   binary precip logit
    """
    def __init__(self, in_channels=42):
        super().__init__()
        self.stem   = _ConvBnRelu(in_channels, 64)
        self.stage1 = _ResBlock(64,  128, stride=2)
        self.stage2 = _ResBlock(128, 256, stride=2)
        self.stage3 = _ResBlock(256, 256, stride=2)
        self.stage4 = _ResBlock(256, 512, stride=2)
        self.gap    = nn.AdaptiveAvgPool2d(1)

        self.cont_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 6),
        )
        self.bin_head = nn.Sequential(
            nn.Linear(512, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).flatten(1)
        return self.cont_head(x), self.bin_head(x).squeeze(1)


# ── Wrapper: handles all pre/post-processing ──────────────────────────────────
class WrappedModel(nn.Module):
    """
    Wraps WeatherCNN so evaluate.py can call it directly:

        pred = model(x.unsqueeze(0))  # (B, H_full, W_full, C) → (B, 6)

    Steps:
      1. Replace NaN with 0
      2. Crop to (CROP_H, CROP_W) window (same window used at training time)
      3. Normalise inputs channel-wise
      4. Forward through WeatherCNN
      5. De-normalise continuous outputs → physical units
    """
    def __init__(self, base_model, inp_mean, inp_std, cont_mean, cont_std):
        super().__init__()
        self.model = base_model
        # shapes: (C,) — stored as buffers so they move with .to(device)
        self.register_buffer("inp_mean",  inp_mean)
        self.register_buffer("inp_std",   inp_std)
        self.register_buffer("cont_mean", cont_mean)
        self.register_buffer("cont_std",  cont_std)

    def forward(self, x):
        # x: (B, H_full, W_full, C)  float32
        x = torch.nan_to_num(x, nan=0.0)

        # ── crop ──────────────────────────────────────────────────────────────
        x = x[:, CROP_R0:CROP_R0 + CROP_H, CROP_C0:CROP_C0 + CROP_W, :]
        # (B, CROP_H, CROP_W, C) → (B, C, CROP_H, CROP_W)
        x = x.permute(0, 3, 1, 2)

        # ── normalise ─────────────────────────────────────────────────────────
        x = (x - self.inp_mean[None, :, None, None]) / self.inp_std[None, :, None, None]

        # ── forward ───────────────────────────────────────────────────────────
        cont_norm, _bin = self.model(x)

        # ── de-normalise ──────────────────────────────────────────────────────
        cont = cont_norm * self.cont_std[None, :] + self.cont_mean[None, :]
        return cont   # (B, 6)


# ── Public API ────────────────────────────────────────────────────────────────
def get_model(metadata):
    """
    Load and return the trained, wrapped model ready for inference.

    Parameters
    ----------
    metadata : dict
        Metadata dict loaded by evaluate.py (used to infer in_channels).

    Returns
    -------
    WrappedModel (eval mode, on CPU)
    """
    # Infer number of input channels from metadata if available
    if "variable_names" in metadata:
        in_channels = len(metadata["variable_names"])
    else:
        # fall back to shape stored in norm stats
        norm = torch.load(NORM_PATH, map_location="cpu", weights_only=False)
        in_channels = norm["mean"].shape[0]

    # Load base model
    base_model = WeatherCNN(in_channels=in_channels)
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    # Allow loading checkpoints that may contain extra keys
    base_model.load_state_dict(state_dict, strict=False)

    # Load normalisation stats
    norm = torch.load(NORM_PATH, map_location="cpu", weights_only=False)
    inp_mean  = norm["mean"].float()       # (C,)
    inp_std   = norm["std"].float()        # (C,)
    cont_mean = norm["cont_mean"].float()  # (6,)
    cont_std  = norm["cont_std"].float()   # (6,)

    if cont_mean is None or cont_std is None:
        raise RuntimeError(
            "norm_stats.pt must include 'cont_mean' and 'cont_std'. "
            "Re-run cnn.py to regenerate the file."
        )

    wrapped = WrappedModel(base_model, inp_mean, inp_std, cont_mean, cont_std)
    wrapped.eval()
    return wrapped