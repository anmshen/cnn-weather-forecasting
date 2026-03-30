#!/usr/bin/env python3
"""
cnn.py — Train a ResNet-style CNN for 24-hour weather forecasting.

Outputs
-------
  best_model.pt   : state_dict of the best WeatherCNN (lowest val loss)
  norm_stats.pt   : normalisation statistics used at training time
                    keys: mean, std, cont_mean, cont_std

Usage
-----
  python cnn.py
"""

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR        = '/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset'
INPUT_DIR       = os.path.join(DATA_DIR, 'inputs')
TARGET_PT       = os.path.join(DATA_DIR, 'targets.pt')
META_PT         = os.path.join(DATA_DIR, 'metadata.pt')

DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE      = 32           # reduce if GPU memory is tight
LR              = 4e-4
EPOCHS          = 50
SPLIT_SEED      = 42
FORECAST_HORIZON = 24         # predict t+24h

_run_time       = time.strftime('%Y%m%d_%H%M%S')
MODEL_PATH      = f'/cluster/tufts/c26sp1cs0137/nliu05/best_model_{_run_time}.pt'
NORM_CACHE      = '/cluster/tufts/c26sp1cs0137/nliu05/norm_stats.pt'

# Spatial crop: take a 352×352 window centred on the target point
# (Jumbo statue is near row ~225, col ~200 in the 450×449 grid).
# Adjust CROP_R0/C0 if your target point is elsewhere.
CROP_H, CROP_W  = 352, 352
CROP_R0         = max(0, 225 - CROP_H // 2)   # 49
CROP_C0         = max(0, 200 - CROP_W // 2)   # 24

# ── File index ────────────────────────────────────────────────────────────────
def build_file_index(input_dir):
    """Walk INPUT_DIR once and build filename → full-path mapping."""
    file_index = {}
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith('.pt'):
                file_index[f] = os.path.join(root, f)
    return file_index

FILE_INDEX = build_file_index(INPUT_DIR)
print(f"Indexed {len(FILE_INDEX)} input files.")

# ── Load metadata / targets ───────────────────────────────────────────────────
meta   = torch.load(META_PT,   weights_only=False)
target = torch.load(TARGET_PT, weights_only=False)

times       = meta['times']                    # numpy datetime64, shape (T,)
target_vals = target['values'].float()         # (T, 6)
target_bin  = target['binary_label'].float()   # (T,)

n = len(times)

def time_to_filename(t):
    """Convert a datetime64 timestamp to the expected filename, e.g. X_2018010100.pt."""
    return "X_" + str(t)[:13].replace('T', '').replace('-', '').replace(':', '') + ".pt"

def input_path_for_index(i):
    return FILE_INDEX.get(time_to_filename(times[i]))

def is_from_trainval_years(i):
    path = input_path_for_index(i)
    if path is None:
        return False
    return any(os.path.join(INPUT_DIR, str(y)) in path for y in (2018, 2019, 2020))

def is_valid_sample(i):
    """True iff:
      - X_t input file exists
      - times[i+24] is exactly 24 hours after times[i] (no gaps in the series)
      - neither X_t nor y_(t+24h) contain NaN
    """
    j = i + FORECAST_HORIZON
    if j >= n:
        return False

    # Confirm the time array is contiguous at this step — guards against
    # missing hours where times[i+24] would be t+25h, t+48h, etc.
    expected_delta = np.timedelta64(FORECAST_HORIZON, 'h')
    if (times[j] - times[i]) != expected_delta:
        return False

    path = input_path_for_index(i)
    if path is None:
        return False

    # lightweight NaN check on targets (fast — avoids loading input tensor here)
    if torch.isnan(target_vals[j]).any():
        return False
    if torch.isnan(target_bin[j]):
        return False

    # NOTE: NaN inputs are caught lazily in Dataset.__getitem__ (returns None)
    # and filtered by collate_skip_none, keeping index-building fast.
    return True

# ── Build train / val split ───────────────────────────────────────────────────
print("Building train/val split (this may take a minute) …")
pool_idx = [i for i in range(n) if is_from_trainval_years(i) and is_valid_sample(i)]
random.Random(SPLIT_SEED).shuffle(pool_idx)
train_end  = int(len(pool_idx) * 0.80)
train_idx  = pool_idx[:train_end]
val_idx    = pool_idx[train_end:]
print(f"  pool={len(pool_idx)}  train={len(train_idx)}  val={len(val_idx)}")

# ── Compute / load normalisation statistics ───────────────────────────────────
def compute_norm_stats(sample_indices, n_samples=2000):
    """Estimate per-channel mean & std from a random subsample of training inputs.

    We also compute mean/std of the continuous targets so the model can be
    trained on normalised outputs and predictions can be de-normalised at
    inference time.
    """
    print(f"  Computing input norm stats from {n_samples} random training samples …")
    rng   = random.Random(0)
    picks = rng.sample(sample_indices, min(n_samples, len(sample_indices)))

    sum_x  = None
    sum_x2 = None
    count  = 0

    for i in picks:
        path = input_path_for_index(i)
        x = torch.load(path, weights_only=True).float()   # (450, 449, C)
        if torch.isnan(x).any():
            continue
        x_crop = x[CROP_R0:CROP_R0+CROP_H, CROP_C0:CROP_C0+CROP_W, :]  # (H,W,C)
        x_flat = x_crop.reshape(-1, x_crop.shape[-1])     # (H*W, C)
        if sum_x is None:
            C = x_flat.shape[1]
            sum_x  = torch.zeros(C)
            sum_x2 = torch.zeros(C)
        sum_x  += x_flat.mean(0)
        sum_x2 += (x_flat ** 2).mean(0)
        count  += 1

    mean = sum_x / count
    std  = (sum_x2 / count - mean ** 2).clamp(min=1e-8).sqrt()

    # continuous targets: collect y_(t+24h) for all training indices
    j_vals = torch.stack([target_vals[i + FORECAST_HORIZON] for i in sample_indices])
    cont_mean = j_vals.mean(0)   # (6,)
    cont_std  = j_vals.std(0).clamp(min=1e-8)

    return {"mean": mean, "std": std, "cont_mean": cont_mean, "cont_std": cont_std}

if os.path.exists(NORM_CACHE):
    print(f"Loading cached norm stats from {NORM_CACHE}")
    norm_stats = torch.load(NORM_CACHE, map_location='cpu', weights_only=False)
else:
    print("Norm cache not found — computing from scratch …")
    norm_stats = compute_norm_stats(train_idx)
    os.makedirs(os.path.dirname(NORM_CACHE), exist_ok=True)
    torch.save(norm_stats, NORM_CACHE)
    print(f"Saved norm stats to {NORM_CACHE}")

# Keep CPU copies for use inside DataLoader worker processes (workers cannot
# access the CUDA context of the main process — putting tensors on CUDA here
# would cause "CUDA initialization error" in every worker).
inp_mean_cpu  = norm_stats['mean'].float()       # (C,)  — used in __getitem__
inp_std_cpu   = norm_stats['std'].float()        # (C,)
cont_mean_cpu = norm_stats['cont_mean'].float()  # (6,)
cont_std_cpu  = norm_stats['cont_std'].float()   # (6,)

# GPU copies used only in the training loop (main process)
inp_mean  = inp_mean_cpu.to(DEVICE)
inp_std   = inp_std_cpu.to(DEVICE)
cont_mean = cont_mean_cpu.to(DEVICE)
cont_std  = cont_std_cpu.to(DEVICE)

IN_CHANNELS = inp_mean_cpu.shape[0]
print(f"  in_channels={IN_CHANNELS}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class WeatherDataset(Dataset):
    """Yields (X_norm, y_cont_norm, y_bin) tuples.

    X_norm        : (C, CROP_H, CROP_W) float32  — normalised spatial crop
    y_cont_norm   : (6,) float32                  — normalised continuous targets
    y_bin         : scalar float32                — binary precip label
    """
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i    = self.indices[idx]
        j    = i + FORECAST_HORIZON
        path = input_path_for_index(i)

        x = torch.load(path, weights_only=True).float()   # (450, 449, C)
        if torch.isnan(x).any():
            return None   # filtered by collate_fn

        # crop
        x_crop = x[CROP_R0:CROP_R0+CROP_H, CROP_C0:CROP_C0+CROP_W, :]   # (H,W,C)
        # (H,W,C) → (C,H,W)
        x_crop = x_crop.permute(2, 0, 1)

        # normalise input — use CPU tensors (workers have no CUDA context)
        x_norm = (x_crop - inp_mean_cpu[:, None, None]) / inp_std_cpu[:, None, None]

        # normalise continuous targets
        y_cont      = target_vals[j]                                   # (6,)
        y_cont_norm = (y_cont - cont_mean_cpu) / cont_std_cpu

        y_bin = target_bin[j]

        return x_norm, y_cont_norm, y_bin


def collate_skip_none(batch):
    """Drop None samples (NaN inputs) that __getitem__ may return."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    x, y_cont, y_bin = zip(*batch)
    return torch.stack(x), torch.stack(y_cont), torch.stack(y_bin)


train_ds = WeatherDataset(train_idx)
val_ds   = WeatherDataset(val_idx)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, collate_fn=collate_skip_none,
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True, collate_fn=collate_skip_none,
)

# ── Model ─────────────────────────────────────────────────────────────────────
class _ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            # depthwise: each channel filtered independently
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            # pointwise: mix channels cheaply
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            # nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class _SE(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // ratio),
            nn.ReLU(),
            nn.Linear(channels // ratio, channels),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.se(x).view(x.size(0), -1, 1, 1)

class _ResBlock(nn.Module):
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
        self.se = _SE(out_ch)

    def forward(self, x):
        return self.relu(self.se(self.conv2(self.conv1(x))) + self.shortcut(x))


class WeatherCNN(nn.Module):
    """
    Encoder-only ResNet CNN for weather scalar regression.

    Input : (B, C, CROP_H, CROP_W)  — already normalised
    Output: (B, 6) continuous targets (in normalised space),
            (B,)  binary logit for precip > 2 mm
    """
    def __init__(self, in_channels):
        super().__init__()
        self.stem   = _ConvBnRelu(in_channels, 64)
        self.stage1 = _ResBlock(64,  128, stride=2)
        self.stage2 = _ResBlock(128, 256, stride=2)
        self.stage3 = _ResBlock(256, 256, stride=2)   # 256→512 (was 256→256)
        self.stage4 = _ResBlock(256, 512, stride=2)  # 512→1024 (was 256→512)
        self.gap    = nn.AdaptiveAvgPool2d(1)

        self.cont_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True),  # 1024 (was 512)
            nn.Dropout(0.3),
            nn.Linear(256, 6),
        )
        self.bin_head = nn.Sequential(
            nn.Linear(512, 64), nn.ReLU(inplace=True),   # 1024 (was 512)
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


# ── Training ──────────────────────────────────────────────────────────────────
# Let cuDNN benchmark and pick the fastest *supported* plan for our fixed
# input shape instead of retrying unsupported plans each forward pass.
torch.backends.cudnn.benchmark       = True
# TF32 avoids the strict fp32 cuDNN convolution plan that raises
# "CUDNN_STATUS_NOT_SUPPORTED" on some driver/PyTorch combinations.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

model     = WeatherCNN(in_channels=IN_CHANNELS).to(DEVICE)
optimiser = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
mse_loss  = nn.MSELoss()
bce_loss  = nn.BCEWithLogitsLoss()

best_val_loss = float('inf')

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_cont = 0.0
    total_bin  = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            if batch is None:
                continue
            x, y_cont, y_bin = batch
            x      = x.to(DEVICE)
            y_cont = y_cont.to(DEVICE)
            y_bin  = y_bin.to(DEVICE)

            pred_cont, pred_bin = model(x)

            loss_cont = mse_loss(pred_cont, y_cont)
            loss_bin  = bce_loss(pred_bin, y_bin)
            loss      = loss_cont + 0.1 * loss_bin   # weight binary term lightly

            if train:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimiser.step()

            total_cont += loss_cont.item()
            total_bin  += loss_bin.item()
            n_batches  += 1

    return total_cont / max(n_batches, 1), total_bin / max(n_batches, 1)


print(f"\nTraining on {DEVICE}  —  {EPOCHS} epochs\n{'─'*55}")
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_cont, tr_bin = run_epoch(train_loader, train=True)
    va_cont, va_bin = run_epoch(val_loader,   train=False)
    scheduler.step()

    val_loss = va_cont + 0.1 * va_bin
    elapsed  = time.time() - t0
    print(
        f"Epoch {epoch:02d}/{EPOCHS}  "
        f"train cont={tr_cont:.4f} bin={tr_bin:.4f}  "
        f"val cont={va_cont:.4f} bin={va_bin:.4f}  "
        f"({elapsed:.0f}s)"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

print(f"\nDone. Best val loss = {best_val_loss:.4f}")
print(f"Model  → {MODEL_PATH}")
print(f"Norms  → {NORM_CACHE}")