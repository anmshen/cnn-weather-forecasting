#!/usr/bin/env python3
"""
sweep.py — Grid search over key hyperparameters for the weather CNN.

Each combination trains for SWEEP_EPOCHS epochs (short runs to rank configs).
Results are appended to sweep_results.csv after every trial so progress is
never lost if the job is killed early.

After the sweep finishes, re-train the best config for the full EPOCHS budget
using cnn.py.

Usage
-----
  python sweep.py
"""

import os
import csv
import random
import time
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy

# ── Fixed paths ───────────────────────────────────────────────────────────────
DATA_DIR         = '/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset'
INPUT_DIR        = os.path.join(DATA_DIR, 'inputs')
TARGET_PT        = os.path.join(DATA_DIR, 'targets.pt')
META_PT          = os.path.join(DATA_DIR, 'metadata.pt')
SAVE_DIR         = '/cluster/tufts/c26sp1cs0137/ashen05'
NORM_CACHE       = os.path.join(SAVE_DIR, 'norm_stats.pt')
RESULTS_CSV      = os.path.join(SAVE_DIR, 'sweep_results.csv')

DEVICE           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SPLIT_SEED       = 42
FORECAST_HORIZON = 24
NUM_WORKERS      = 4

# ── Sweep budget ──────────────────────────────────────────────────────────────
# Short runs to rank configs — increase if you have more time.
SWEEP_EPOCHS     = 5

# ── Hyperparameter grid ───────────────────────────────────────────────────────
# Add / remove values freely; every combination will be evaluated.
PARAM_GRID = {
    "lr":            [1e-3, 3e-4, 1e-4],
    "batch_size":    [8, 16],
    "bin_loss_w":    [0.1, 0.3, 0.5],   # weight of BCE vs MSE
    "crop_r0":       [30, 49, 70],       # top-left row of spatial crop
    "crop_c0":       [0, 24, 50],        # top-left col (shift west = smaller)
}

# Spatial crop size — fixed across all trials (change here if desired)
CROP_H, CROP_W = 352, 352

# cuDNN flags — set once for the whole process
torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

# ── File index ────────────────────────────────────────────────────────────────
def build_file_index(input_dir):
    index = {}
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith('.pt'):
                index[f] = os.path.join(root, f)
    return index

print("Building file index …")
FILE_INDEX = build_file_index(INPUT_DIR)
print(f"  Indexed {len(FILE_INDEX)} files.")

# ── Load metadata / targets ───────────────────────────────────────────────────
print("Loading metadata and targets …")
meta        = torch.load(META_PT,   weights_only=False)
target      = torch.load(TARGET_PT, weights_only=False)
times       = meta['times']
target_vals = target['values'].float()
target_bin  = target['binary_label'].float()
n           = len(times)

def time_to_filename(t):
    return "X_" + str(t)[:13].replace('T', '').replace('-', '').replace(':', '') + ".pt"

def input_path(i):
    return FILE_INDEX.get(time_to_filename(times[i]))

def is_trainval(i):
    p = input_path(i)
    if p is None:
        return False
    return any(os.path.join(INPUT_DIR, str(y)) in p for y in (2018, 2019, 2020))

def is_valid(i):
    j = i + FORECAST_HORIZON
    if j >= n:
        return False
    if (times[j] - times[i]) != np.timedelta64(FORECAST_HORIZON, 'h'):
        return False
    if input_path(i) is None:
        return False
    if torch.isnan(target_vals[j]).any() or torch.isnan(target_bin[j]):
        return False
    return True

# ── Build split once (shared across all trials) ───────────────────────────────
print("Building train/val split …")
pool = [i for i in range(n) if is_trainval(i) and is_valid(i)]
random.Random(SPLIT_SEED).shuffle(pool)
cut        = int(len(pool) * 0.80)
train_idx  = pool[:cut]
val_idx    = pool[cut:]
print(f"  pool={len(pool)}  train={len(train_idx)}  val={len(val_idx)}")

# ── Norm stats ────────────────────────────────────────────────────────────────
# Use the default crop (CROP_R0=49, CROP_C0=24) for norm stats.
# Re-computing per crop would be very expensive; the differences are small
# because the same channels are present everywhere in the region.
DEFAULT_R0, DEFAULT_C0 = 49, 24

def compute_norm_stats(indices, r0, c0, n_samples=2000):
    rng   = random.Random(0)
    picks = rng.sample(indices, min(n_samples, len(indices)))
    sum_x, sum_x2, count = None, None, 0
    for i in picks:
        p = input_path(i)
        x = torch.load(p, weights_only=True).float()
        if torch.isnan(x).any():
            continue
        crop = x[r0:r0+CROP_H, c0:c0+CROP_W, :].reshape(-1, x.shape[-1])
        if sum_x is None:
            sum_x  = torch.zeros(crop.shape[1])
            sum_x2 = torch.zeros(crop.shape[1])
        sum_x  += crop.mean(0)
        sum_x2 += (crop ** 2).mean(0)
        count  += 1
    mean = sum_x / count
    std  = (sum_x2 / count - mean ** 2).clamp(min=1e-8).sqrt()
    j_vals    = torch.stack([target_vals[i + FORECAST_HORIZON] for i in indices])
    cont_mean = j_vals.mean(0)
    cont_std  = j_vals.std(0).clamp(min=1e-8)
    return {"mean": mean, "std": std, "cont_mean": cont_mean, "cont_std": cont_std}

if os.path.exists(NORM_CACHE):
    print(f"Loading cached norm stats from {NORM_CACHE}")
    norm_stats = torch.load(NORM_CACHE, map_location='cpu', weights_only=False)
else:
    print("Computing norm stats …")
    norm_stats = compute_norm_stats(train_idx, DEFAULT_R0, DEFAULT_C0)
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(norm_stats, NORM_CACHE)
    print(f"  Saved → {NORM_CACHE}")

IN_CHANNELS = norm_stats['mean'].shape[0]
print(f"  in_channels = {IN_CHANNELS}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class WeatherDataset(Dataset):
    """Crop position is passed at construction so each trial uses its own."""
    def __init__(self, indices, r0, c0, inp_mean, inp_std, cont_mean, cont_std):
        self.indices   = indices
        self.r0, self.c0 = r0, c0
        # CPU tensors — safe for multiprocessing workers
        self.inp_mean  = inp_mean
        self.inp_std   = inp_std
        self.cont_mean = cont_mean
        self.cont_std  = cont_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        j = i + FORECAST_HORIZON
        p = input_path(i)
        x = torch.load(p, weights_only=True).float()
        if torch.isnan(x).any():
            return None
        x_crop = x[self.r0:self.r0+CROP_H, self.c0:self.c0+CROP_W, :]
        # guard against crop going out of bounds
        if x_crop.shape[0] != CROP_H or x_crop.shape[1] != CROP_W:
            return None
        x_crop = x_crop.permute(2, 0, 1)   # (C, H, W)
        x_norm = (x_crop - self.inp_mean[:, None, None]) / self.inp_std[:, None, None]
        y_cont = (target_vals[j] - self.cont_mean) / self.cont_std
        y_bin  = target_bin[j]
        return x_norm, y_cont, y_bin

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    x, yc, yb = zip(*batch)
    return torch.stack(x), torch.stack(yc), torch.stack(yb)

# ── Model ─────────────────────────────────────────────────────────────────────
class _ConvBnRelu(nn.Module):
    def __init__(self, ic, oc, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class _ResBlock(nn.Module):
    def __init__(self, ic, oc, stride=1):
        super().__init__()
        self.c1 = _ConvBnRelu(ic, oc, stride=stride)
        self.c2 = nn.Sequential(
            nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc))
        self.skip = (nn.Sequential(
            nn.Conv2d(ic, oc, 1, stride=stride, bias=False), nn.BatchNorm2d(oc))
            if stride != 1 or ic != oc else nn.Identity())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.c2(self.c1(x)) + self.skip(x))

class WeatherCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.stem   = _ConvBnRelu(in_channels, 64)
        self.stage1 = _ResBlock(64,  128, stride=2)
        self.stage2 = _ResBlock(128, 256, stride=2)
        self.stage3 = _ResBlock(256, 256, stride=2)
        self.stage4 = _ResBlock(256, 512, stride=2)
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.cont_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, 6))
        self.bin_head = nn.Sequential(
            nn.Linear(512, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
    def forward(self, x):
        for layer in [self.stem, self.stage1, self.stage2, self.stage3, self.stage4]:
            x = layer(x)
        x = self.gap(x).flatten(1)
        return self.cont_head(x), self.bin_head(x).squeeze(1)

# ── Single trial ──────────────────────────────────────────────────────────────
def run_trial(params):
    lr         = params['lr']
    batch_size = params['batch_size']
    bin_w      = params['bin_loss_w']
    r0         = params['crop_r0']
    c0         = params['crop_c0']

    # Clamp crop so it never goes out of bounds (450×449 grid)
    r0 = min(r0, 450 - CROP_H)
    c0 = min(c0, 449 - CROP_W)

    inp_mean_cpu  = norm_stats['mean'].float()
    inp_std_cpu   = norm_stats['std'].float()
    cont_mean_cpu = norm_stats['cont_mean'].float()
    cont_std_cpu  = norm_stats['cont_std'].float()

    train_ds = WeatherDataset(train_idx, r0, c0,
                              inp_mean_cpu, inp_std_cpu,
                              cont_mean_cpu, cont_std_cpu)
    val_ds   = WeatherDataset(val_idx,   r0, c0,
                              inp_mean_cpu, inp_std_cpu,
                              cont_mean_cpu, cont_std_cpu)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              collate_fn=collate_skip_none)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              collate_fn=collate_skip_none)

    model     = WeatherCNN(IN_CHANNELS).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=SWEEP_EPOCHS)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    best_val = float('inf')

    for epoch in range(1, SWEEP_EPOCHS + 1):
        # ── train ──
        model.train()
        for batch in train_loader:
            if batch is None:
                continue
            x, yc, yb = [t.to(DEVICE) for t in batch]
            pc, pb = model(x)
            loss = mse(pc, yc) + bin_w * bce(pb, yb)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
        scheduler.step()

        # ── val ──
        model.eval()
        tot_cont, tot_bin, nb = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                x, yc, yb = [t.to(DEVICE) for t in batch]
                pc, pb = model(x)
                tot_cont += mse(pc, yc).item()
                tot_bin  += bce(pb, yb).item()
                nb += 1
        if nb:
            val_loss = tot_cont / nb + bin_w * tot_bin / nb
            if val_loss < best_val:
                best_val = val_loss

    return best_val

# ── Grid search ───────────────────────────────────────────────────────────────
keys   = list(PARAM_GRID.keys())
values = list(PARAM_GRID.values())
combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
total  = len(combos)
print(f"\n{'='*60}")
print(f"Starting sweep: {total} combinations × {SWEEP_EPOCHS} epochs each")
print(f"Results → {RESULTS_CSV}")
print(f"{'='*60}\n")

# Load already-completed rows so we can skip them on restart
completed = set()
if os.path.exists(RESULTS_CSV):
    with open(RESULTS_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = tuple(row[k] for k in keys)
            completed.add(key)
    print(f"Resuming — {len(completed)} trials already done.\n")

os.makedirs(SAVE_DIR, exist_ok=True)
csv_exists = os.path.exists(RESULTS_CSV)
csv_file   = open(RESULTS_CSV, 'a', newline='')
writer     = csv.DictWriter(csv_file, fieldnames=keys + ['val_loss', 'duration_s'])
if not csv_exists:
    writer.writeheader()

best_overall = {'val_loss': float('inf'), 'params': None}

for trial_num, params in enumerate(combos, 1):
    combo_key = tuple(str(params[k]) for k in keys)
    if combo_key in completed:
        print(f"[{trial_num:>3}/{total}] SKIP (already done): {params}")
        continue

    print(f"[{trial_num:>3}/{total}] {params}")
    t0 = time.time()
    try:
        val_loss = run_trial(params)
        duration = time.time() - t0
        row = {**{k: params[k] for k in keys},
               'val_loss': f'{val_loss:.6f}',
               'duration_s': f'{duration:.1f}'}
        writer.writerow(row)
        csv_file.flush()
        print(f"         val_loss={val_loss:.4f}  ({duration:.0f}s)")

        if val_loss < best_overall['val_loss']:
            best_overall = {'val_loss': val_loss, 'params': deepcopy(params)}
            print(f"         ★ New best!")

    except Exception as e:
        print(f"         ERROR: {e}")
        row = {**{k: params[k] for k in keys},
               'val_loss': 'ERROR',
               'duration_s': f'{time.time()-t0:.1f}'}
        writer.writerow(row)
        csv_file.flush()

csv_file.close()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Sweep complete.")
print(f"Results saved to: {RESULTS_CSV}")
if best_overall['params']:
    print(f"\nBest val_loss : {best_overall['val_loss']:.4f}")
    print("Best params   :")
    for k, v in best_overall['params'].items():
        print(f"  {k:15s} = {v}")
    print("\nCopy these into cnn.py and run the full training.")
print(f"{'='*60}")