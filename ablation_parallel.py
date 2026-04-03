#!/usr/bin/env python3
import sys

ONLY_LABEL = None
if len(sys.argv) > 1:
    ONLY_LABEL = sys.argv[1]
    
"""
ablation.py — Part 3: Input Variable Ablation Study
=====================================================

Hypothesis
----------
Input features that are highly correlated (e.g. temperature at different
pressure levels, or wind components) will be redundant — a model trained
on a well-chosen subset can match the performance of the full-feature model.

Approach
--------
We define N variable groups based on physical similarity, then train one
WeatherCNN per configuration:
  - "all_features"         : every channel (baseline)
  - "drop_<group>"         : all channels except one group (leave-one-group-out)
  - "only_<group>"         : a single group in isolation

Each model is evaluated on the 2021 test year with the same metrics as
the main evaluation:
  - RMSE per continuous target
  - RMSE for APCP when true > 2 mm
  - AUC for binary precipitation label

Results are saved to ablation_results.json and ablation_results.csv.

Usage
-----
  python ablation.py
"""

import os
import json
import time
import random
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR         = '/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset'
INPUT_DIR        = os.path.join(DATA_DIR, 'inputs')
TARGET_PT        = os.path.join(DATA_DIR, 'targets.pt')
META_PT          = os.path.join(DATA_DIR, 'metadata.pt')

DEVICE           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE       = 8
LR               = 3e-4
EPOCHS           = 50
SPLIT_SEED       = 42
FORECAST_HORIZON = 24

CROP_H, CROP_W   = 352, 352
CROP_R0          = max(0, 225 - CROP_H // 2)
CROP_C0          = max(0, 200 - CROP_W // 2)

OUTPUT_DIR       = '/cluster/tufts/c26sp1cs0137/ashen05/ablation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_JSON = os.path.join(OUTPUT_DIR, f'{ONLY_LABEL}_results.json' if ONLY_LABEL else 'ablation_results.json')
RESULTS_CSV  = os.path.join(OUTPUT_DIR, f'{ONLY_LABEL}_results.csv' if ONLY_LABEL else 'ablation_results.csv')

# ── Load metadata / targets ───────────────────────────────────────────────────
meta       = torch.load(META_PT,   weights_only=False)
target     = torch.load(TARGET_PT, weights_only=False)

times       = meta['times']
var_names   = meta['variable_names']           # list of channel name strings
target_vals = target['values'].float()        # (T, 6)
target_bin  = target['binary_label'].float()  # (T,)
n           = len(times)

print(f"Total timesteps : {n}")
print(f"Input variables : {len(var_names)}")
print("Variable list:")
for idx, v in enumerate(var_names):
    print(f"  [{idx:3d}] {v}")

# ── Variable group definitions ────────────────────────────────────────────────
# Groups are defined by substring matching on the variable name.
# A variable is assigned to the FIRST group whose substrings match.
# Variables that match no group land in "other".
#
# Adjust the substrings below to match the exact names in your metadata.

GROUP_PATTERNS = {
    "temperature":   ["TMP", "T2M", "temp"],
    "humidity":      ["RH", "SPFH", "DPT", "dewpoint", "humidity"],
    "wind_u":        ["UGRD", "U10", "UWIND"],
    "wind_v":        ["VGRD", "V10", "VWIND"],
    "wind_gust":     ["GUST"],
    "precipitation": ["APCP", "PRATE", "precip"],
    "pressure":      ["PRES", "MSLMA", "HGT", "geopotential"],
    "clouds_rad":    ["TCDC", "DSWRF", "DLWRF", "USWRF", "ULWRF",
                      "cloud", "radiation"],
    "soil_sfc":      ["SOILW", "TSOIL", "WEASD", "SNOWC", "snow"],
}


def assign_groups(names, patterns):
    groups = {g: [] for g in patterns}
    groups["other"] = []
    for ch, name in enumerate(names):
        assigned = False
        for group, substrings in patterns.items():
            if any(s.lower() in name.lower() for s in substrings):
                groups[group].append(ch)
                assigned = True
                break
        if not assigned:
            groups["other"].append(ch)
    return {g: idxs for g, idxs in groups.items() if idxs}


GROUPS       = assign_groups(var_names, GROUP_PATTERNS)
ALL_CHANNELS = list(range(len(var_names)))

print("\nGroup assignments:")
for g, idxs in GROUPS.items():
    preview = idxs[:6]
    ellipsis = "..." if len(idxs) > 6 else ""
    print(f"  {g:20s}: {len(idxs):3d} channels  {preview}{ellipsis}")

# ── Experiment configurations ─────────────────────────────────────────────────
experiments = []

# 1. Full model — baseline
experiments.append(("all_features", ALL_CHANNELS))

# 2. Leave-one-group-out
for group_name, group_idxs in GROUPS.items():
    remaining = [c for c in ALL_CHANNELS if c not in group_idxs]
    if remaining:
        experiments.append((f"drop_{group_name}", remaining))

# 3. Single group only
for group_name, group_idxs in GROUPS.items():
    experiments.append((f"only_{group_name}", group_idxs))

print(f"\nTotal experiments to run: {len(experiments)}")
for label, ch in experiments:
    print(f"  {label:40s} ({len(ch)} channels)")

# ── File index ────────────────────────────────────────────────────────────────
def build_file_index(input_dir):
    idx = {}
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith('.pt'):
                idx[f] = os.path.join(root, f)
    return idx


FILE_INDEX = build_file_index(INPUT_DIR)
print(f"\nIndexed {len(FILE_INDEX)} input files.")


def time_to_filename(t):
    return "X_" + str(t)[:13].replace('T', '').replace('-', '').replace(':', '') + ".pt"


def input_path(i):
    return FILE_INDEX.get(time_to_filename(times[i]))


def is_from_year(i, year):
    p = input_path(i)
    return p is not None and os.path.join(INPUT_DIR, str(year)) in p


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


# ── Splits ────────────────────────────────────────────────────────────────────
print("Building splits …")
pool = [i for i in range(n)
        if (is_from_year(i, 2018) or is_from_year(i, 2019) or is_from_year(i, 2020))
        and is_valid(i)]
random.Random(SPLIT_SEED).shuffle(pool)
cut       = int(len(pool) * 0.80)
train_idx = pool[:cut]
val_idx   = pool[cut:]
test_idx  = [i for i in range(n) if is_from_year(i, 2021) and is_valid(i)]
print(f"  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

# ── Normalisation stats ───────────────────────────────────────────────────────
NORM_CACHE = os.path.join(OUTPUT_DIR, 'ablation_norm_stats.pt')


def compute_norm_stats(indices, n_samples=2000):
    print(f"  Computing norm stats from {n_samples} samples …")
    picks = random.Random(0).sample(indices, min(n_samples, len(indices)))
    sum_x = sum_x2 = None
    count = 0
    for i in picks:
        x = torch.load(input_path(i), weights_only=True).float()
        if torch.isnan(x).any():
            continue
        x_flat = x[CROP_R0:CROP_R0+CROP_H, CROP_C0:CROP_C0+CROP_W, :].reshape(-1, x.shape[-1])
        if sum_x is None:
            C = x_flat.shape[1]
            sum_x = torch.zeros(C)
            sum_x2 = torch.zeros(C)
        sum_x  += x_flat.mean(0)
        sum_x2 += (x_flat ** 2).mean(0)
        count  += 1
    mean = sum_x / count
    std  = (sum_x2 / count - mean ** 2).clamp(min=1e-8).sqrt()
    jv        = torch.stack([target_vals[i + FORECAST_HORIZON] for i in indices])
    cont_mean = jv.mean(0)
    cont_std  = jv.std(0).clamp(min=1e-8)
    return {"mean": mean, "std": std, "cont_mean": cont_mean, "cont_std": cont_std}


if os.path.exists(NORM_CACHE):
    print(f"Loading cached norm stats from {NORM_CACHE}")
    ns = torch.load(NORM_CACHE, map_location='cpu', weights_only=False)
else:
    ns = compute_norm_stats(train_idx)
    torch.save(ns, NORM_CACHE)
    print(f"Saved norm stats → {NORM_CACHE}")

inp_mean_all  = ns['mean'].float()       # (C_all,)
inp_std_all   = ns['std'].float()        # (C_all,)
cont_mean_cpu = ns['cont_mean'].float()  # (6,)
cont_std_cpu  = ns['cont_std'].float()   # (6,)
cont_mean_dev = cont_mean_cpu.to(DEVICE)
cont_std_dev  = cont_std_cpu.to(DEVICE)

TARGET_NAMES = [
    'TMP@2m_above_ground',
    'RH@2m_above_ground',
    'UGRD@10m_above_ground',
    'VGRD@10m_above_ground',
    'GUST@surface',
    'APCP_1hr_acc_fcst@surface',
]

# ── Dataset ───────────────────────────────────────────────────────────────────
class WeatherDataset(Dataset):
    def __init__(self, indices, channel_list, mean, std):
        self.indices      = indices
        self.channel_list = channel_list
        # Always keep on CPU — DataLoader worker processes have no CUDA context
        self.mean         = mean.cpu().float()
        self.std          = std.cpu().float()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        j = i + FORECAST_HORIZON
        x = torch.load(input_path(i), weights_only=True).float()
        if torch.isnan(x).any():
            return None
        x_crop = x[CROP_R0:CROP_R0+CROP_H, CROP_C0:CROP_C0+CROP_W, :]
        x_crop = x_crop[:, :, self.channel_list].permute(2, 0, 1)
        x_norm = (x_crop - self.mean[:, None, None]) / self.std[:, None, None]
        y_cont_norm = (target_vals[j] - cont_mean_cpu) / cont_std_cpu
        return x_norm, y_cont_norm, target_bin[j]


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    x, y_cont, y_bin = zip(*batch)
    return torch.stack(x), torch.stack(y_cont), torch.stack(y_bin)


# ── Model ─────────────────────────────────────────────────────────────────────
class _ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


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
            ) if (stride != 1 or in_ch != out_ch) else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)) + self.shortcut(x))


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
            nn.Dropout(0.3), nn.Linear(256, 6),
        )
        self.bin_head = nn.Sequential(
            nn.Linear(512, 64), nn.ReLU(inplace=True), nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).flatten(1)
        return self.cont_head(x), self.bin_head(x).squeeze(1)


# ── Training helpers ──────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()


def run_epoch(model, loader, opt=None):
    model.train() if opt is not None else model.eval()
    total_c = total_b = nb = 0
    ctx = torch.enable_grad() if opt is not None else torch.no_grad()
    with ctx:
        for batch in loader:
            if batch is None:
                continue
            x, y_cont, y_bin = [t.to(DEVICE) for t in batch]
            pc, pb = model(x)
            lc = mse_loss(pc, y_cont)
            lb = bce_loss(pb, y_bin)
            loss = lc + 0.1 * lb
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            total_c += lc.item()
            total_b += lb.item()
            nb      += 1
    return total_c / max(nb, 1), total_b / max(nb, 1)


def train_model(label, channel_list):
    print(f"\n{'='*62}")
    print(f"  Experiment : {label}  ({len(channel_list)} channels)")
    print(f"{'='*62}")

    ch_t     = torch.tensor(channel_list)
    mean_sub = inp_mean_all[ch_t]
    std_sub  = inp_std_all[ch_t]

    tl = DataLoader(WeatherDataset(train_idx, channel_list, mean_sub, std_sub),
                    BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True,
                    collate_fn=collate_skip_none)
    vl = DataLoader(WeatherDataset(val_idx,   channel_list, mean_sub, std_sub),
                    BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
                    collate_fn=collate_skip_none)

    model = WeatherCNN(in_channels=len(channel_list)).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_loss = float('inf')
    ckpt_path = os.path.join(OUTPUT_DIR, f'{label}.pt')

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tc, tb = run_epoch(model, tl, opt)
        vc, vb = run_epoch(model, vl)
        sched.step()
        vl_total = vc + 0.1 * vb
        print(f"  Ep {epoch:02d}/{EPOCHS}  "
              f"tr={tc:.4f}/{tb:.4f}  val={vc:.4f}/{vb:.4f}  "
              f"({time.time()-t0:.0f}s)")
        if vl_total < best_loss:
            best_loss = vl_total
            torch.save({'state_dict': model.state_dict(),
                        'channels': channel_list,
                        'mean': mean_sub, 'std': std_sub}, ckpt_path)
            print(f"    checkpointed (val={vl_total:.4f})")

    print(f"  Best val loss: {best_loss:.4f}")
    return ckpt_path


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_model(label, channel_list, ckpt_path):
    # Load checkpoint to CPU first — mean/std must stay on CPU for DataLoader workers
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = WeatherCNN(in_channels=len(channel_list)).to(DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    ds = WeatherDataset(test_idx, channel_list, ckpt['mean'], ckpt['std'])
    dl = DataLoader(ds, BATCH_SIZE, shuffle=False, num_workers=4,
                    pin_memory=True, collate_fn=collate_skip_none)

    pc_list, tc_list, pb_list, tb_list = [], [], [], []
    with torch.no_grad():
        for batch in dl:
            if batch is None:
                continue
            x, y_cn, y_b = [t.to(DEVICE) for t in batch]
            pcn, pbl = model(x)
            pc_list.append((pcn * cont_std_dev + cont_mean_dev).cpu())
            tc_list.append((y_cn * cont_std_dev + cont_mean_dev).cpu())
            pb_list.append(torch.sigmoid(pbl).cpu())
            tb_list.append(y_b.cpu())

    pred_c = torch.cat(pc_list).numpy()
    true_c = torch.cat(tc_list).numpy()
    pred_b = torch.cat(pb_list).numpy()
    true_b = torch.cat(tb_list).numpy()

    m = {'label': label, 'n_channels': len(channel_list)}

    for k, name in enumerate(TARGET_NAMES):
        m[f'rmse_{name}'] = float(np.sqrt(np.mean((pred_c[:, k] - true_c[:, k]) ** 2)))

    apcp_k     = TARGET_NAMES.index('APCP_1hr_acc_fcst@surface')
    heavy_mask = true_c[:, apcp_k] > 2.0
    if heavy_mask.sum() > 0:
        m['rmse_APCP_heavy'] = float(np.sqrt(np.mean(
            (pred_c[heavy_mask, apcp_k] - true_c[heavy_mask, apcp_k]) ** 2)))
    else:
        m['rmse_APCP_heavy'] = float('nan')
    m['n_heavy'] = int(heavy_mask.sum())

    try:
        m['auc_precip_binary'] = float(roc_auc_score(true_b, pred_b))
    except ValueError:
        m['auc_precip_binary'] = float('nan')
    m['n_pos']   = int(true_b.sum())
    m['n_total'] = len(true_b)
    return m


def print_metrics(m):
    print(f"\n  Results — {m['label']}  ({m['n_channels']} channels)")
    print(f"  {'─'*56}")
    for name in TARGET_NAMES:
        print(f"    {name:42s}  RMSE: {m[f'rmse_{name}']:8.4f}")
    print(f"    {'APCP (true>2mm)':42s}  RMSE: {m['rmse_APCP_heavy']:8.4f}"
          f"  [n={m['n_heavy']}]")
    print(f"    {'APCP binary':42s}   AUC: {m['auc_precip_binary']:8.4f}")


# ── Main loop ─────────────────────────────────────────────────────────────────
all_results: list = []
if os.path.exists(RESULTS_JSON):
    with open(RESULTS_JSON) as f:
        all_results = json.load(f)
    done = {r['label'] for r in all_results}
    print(f"\nResuming — {len(done)} experiments already done.")
else:
    done = set()

for label, channel_list in experiments:
    if ONLY_LABEL is not None and label != ONLY_LABEL:
        continue
    if label in done:
        print(f"  Skipping {label} (already complete)")
        continue

    ckpt_path = os.path.join(OUTPUT_DIR, f'{label}.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = train_model(label, channel_list)
    else:
        print(f"\n  Checkpoint exists for '{label}', skipping training.")

    metrics = evaluate_model(label, channel_list, ckpt_path)
    print_metrics(metrics)
    all_results.append(metrics)

    with open(RESULTS_JSON, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved → {RESULTS_JSON}")

# ── CSV ───────────────────────────────────────────────────────────────────────
if all_results:
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV saved → {RESULTS_CSV}")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"{'ABLATION RESULTS SUMMARY':^100}")
print(f"{'='*100}")
print(f"{'Experiment':<38} {'#ch':>4}  "
      f"{'TMP':>7} {'RH':>7} {'UGRD':>7} {'VGRD':>7} "
      f"{'GUST':>7} {'APCP>2':>8} {'AUC':>7}")
print('─' * 100)


def sort_key(r):
    lbl = r['label']
    if lbl == 'all_features': return (0, lbl)
    if lbl.startswith('drop_'): return (1, lbl)
    return (2, lbl)


for r in sorted(all_results, key=sort_key):
    rmses = [r.get(f'rmse_{name}', float('nan')) for name in TARGET_NAMES]
    print(f"  {r['label']:<36} {r['n_channels']:>4}  "
          f"{rmses[0]:>7.4f} {rmses[1]:>7.3f} {rmses[2]:>7.4f} {rmses[3]:>7.4f} "
          f"{rmses[4]:>7.4f} {r['rmse_APCP_heavy']:>8.4f} {r['auc_precip_binary']:>7.4f}")

print(f"\nOutputs in: {OUTPUT_DIR}")