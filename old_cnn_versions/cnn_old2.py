import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR   = '/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset'
INPUT_DIR  = os.path.join(DATA_DIR, 'inputs')
TARGET_PT  = os.path.join(DATA_DIR, 'targets.pt')
META_PT    = os.path.join(DATA_DIR, 'metadata.pt')

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
LR         = 3e-4
EPOCHS     = 20
SPLIT_SEED = 42  # shuffle train/val pool before 80:20 split (reproducible)

MODEL_PATH = '/cluster/tufts/c26sp1cs0137/ashen05/best_model.pt'
NORM_CACHE = '/cluster/tufts/c26sp1cs0137/ashen05/norm_stats.pt'

# Hourly time series: input at index i is X_t; targets are weather at t + 24h
# (same convention as evaluation/evaluate.py: t24_idx = t_idx + 24).
FORECAST_HORIZON = 24

# ── File index ───────────────────────────────────────────────────────────────
def build_file_index(input_dir):
    file_index = {}
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith('.pt'):
                file_index[f] = os.path.join(root, f)
    return file_index

FILE_INDEX = build_file_index(INPUT_DIR)
print(f"Indexed {len(FILE_INDEX)} input files.")

# ── Load data ────────────────────────────────────────────────────────────────
meta   = torch.load(META_PT,   weights_only=False)
target = torch.load(TARGET_PT, weights_only=False)

times        = meta['times']
target_vals  = target['values'].float()
target_bin   = target['binary_label'].float()

def time_to_filename(t):
    return "X_" + str(t)[:13].replace('T', '').replace('-', '').replace(':', '') + ".pt"

def input_path_for_index(i):
    return FILE_INDEX.get(time_to_filename(times[i]))

def is_from_trainval_years(i):
    """Inputs for train/val must live under inputs/2018, inputs/2019, or inputs/2020."""
    path = input_path_for_index(i)
    if path is None:
        return False
    return any(os.path.join(INPUT_DIR, str(y)) in path for y in (2018, 2019, 2020))

# ── Split ────────────────────────────────────────────────────────────────────
# Train and validation only from 2018–2020 folders; shuffle pool, then 80% / 20%.
# Require i + FORECAST_HORIZON so targets at t+24h exist (evaluate.py uses the same offset).
n = len(times)
pool_idx = [
    i for i in range(n)
    if is_from_trainval_years(i) and i + FORECAST_HORIZON < n
]
random.Random(SPLIT_SEED).shuffle(pool_idx)
train_end = int(len(pool_idx) * 0.80)
train_idx = pool_idx[:train_end]
val_idx   = pool_idx[train_end:]
print(f"Split (2018–2020, t+{FORECAST_HORIZON}h targets, seed={SPLIT_SEED}): "
      f"train {len(train_idx)} | val {len(val_idx)} (pool {len(pool_idx)})")

# ── Normalize targets (IMPORTANT) ────────────────────────────────────────────
# Stats from training *forecast* rows: target at time index i + FORECAST_HORIZON.
train_tgt_idx = torch.tensor(train_idx, dtype=torch.long) + FORECAST_HORIZON
clean = torch.nan_to_num(target_vals[train_tgt_idx], nan=0.0)

cont_mean = clean.mean(dim=0)
cont_std  = clean.std(dim=0).clamp(min=1e-6)

target_vals = (target_vals - cont_mean) / cont_std

# ── Input normalization ──────────────────────────────────────────────────────
def compute_norm_stats(indices, n_vars=42, sample_every=24):
    sums  = torch.zeros(n_vars)
    sqsum = torch.zeros(n_vars)
    count = 0

    for i in indices[::sample_every]:
        path = input_path_for_index(i)
        if path is None:
            continue

        x = torch.load(path, weights_only=False).float()
        x = torch.nan_to_num(x, nan=0.0)

        x_mean = x.mean(dim=(0, 1))
        x_sq   = (x ** 2).mean(dim=(0, 1))

        sums  += x_mean
        sqsum += x_sq
        count += 1

    mean = sums / count
    var  = (sqsum / count - mean ** 2).clamp(min=1e-6)
    std  = torch.sqrt(var)

    return mean, std

if os.path.exists(NORM_CACHE):
    norm = torch.load(NORM_CACHE, weights_only=False)
    ch_mean, ch_std = norm['mean'], norm['std']
else:
    ch_mean, ch_std = compute_norm_stats(train_idx)
    torch.save({'mean': ch_mean, 'std': ch_std}, NORM_CACHE)

# ── Dataset ──────────────────────────────────────────────────────────────────
class WeatherDataset(Dataset):
    def __init__(self, indices):
        self.indices = [
            i for i in indices
            if input_path_for_index(i) is not None and i + FORECAST_HORIZON < n
        ]
        print(f"Dataset: {len(self.indices)}/{len(indices)} valid")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        path = FILE_INDEX[time_to_filename(times[i])]

        x = torch.load(path, weights_only=False).float()
        x = torch.nan_to_num(x, nan=0.0)

        x = (x - ch_mean) / ch_std
        x = x.permute(2, 0, 1)

        j = i + FORECAST_HORIZON
        y_cont = torch.nan_to_num(target_vals[j], nan=0.0)
        y_bin  = torch.nan_to_num(target_bin[j], nan=0.0)

        return x, y_cont, y_bin

# ── Model ────────────────────────────────────────────────────────────────────
# U-Net-style encoder–decoder with skips: good for gridded fields (multi-scale
# patterns, fronts, coastlines). Targets here are still *global* scalars, so we
# global-pool the decoded full-resolution map — not a dense weather map head.
class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class _Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = _DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            dy = skip.size(2) - x.size(2)
            dx = skip.size(3) - x.size(3)
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class WeatherCNN(nn.Module):
    def __init__(self, in_channels=42):
        super().__init__()
        self.inc = _DoubleConv(in_channels, 64)
        self.down1 = _Down(64, 128)
        self.down2 = _Down(128, 256)
        self.down3 = _Down(256, 256)
        self.up1 = _Up(256, 256, 128)
        self.up2 = _Up(128, 128, 64)
        self.up3 = _Up(64, 64, 64)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cont_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 6),
        )
        self.bin_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        u = self.up1(x3, x2)
        u = self.up2(u, x1)
        u = self.up3(u, x0)
        u = self.gap(u).flatten(1)
        return self.cont_head(u), self.bin_head(u).squeeze(1)

# ── Training ─────────────────────────────────────────────────────────────────
def main():
    train_loader = DataLoader(WeatherDataset(train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(WeatherDataset(val_idx),   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = WeatherCNN().to(DEVICE)

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0

        for x, y_cont, y_bin in train_loader:
            x, y_cont, y_bin = x.to(DEVICE), y_cont.to(DEVICE), y_bin.to(DEVICE)

            pred_cont, pred_bin = model(x)

            # ── Balanced loss ──
            loss_main   = mse_loss(pred_cont[:, :5], y_cont[:, :5])
            loss_precip = mse_loss(pred_cont[:, 5],  y_cont[:, 5])
            loss_bin    = bce_loss(pred_bin, y_bin)

            loss = loss_main + 0.5 * loss_precip + 0.1 * loss_bin

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x, y_cont, y_bin in val_loader:
                x, y_cont, y_bin = x.to(DEVICE), y_cont.to(DEVICE), y_bin.to(DEVICE)

                pred_cont, pred_bin = model(x)

                loss_main   = mse_loss(pred_cont[:, :5], y_cont[:, :5])
                loss_precip = mse_loss(pred_cont[:, 5],  y_cont[:, 5])
                loss_bin    = bce_loss(pred_bin, y_bin)

                loss = loss_main + 0.5 * loss_precip + 0.1 * loss_bin

                if not torch.isnan(loss):
                    val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val_loss and not np.isnan(val_loss):
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print("Saved best model")

if __name__ == "__main__":
    main()