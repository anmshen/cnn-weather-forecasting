#!/usr/bin/env python3
"""
Evaluate a weather forecasting model on the test set.

Usage:
    python evaluate.py

Configuration
-------------
Edit the two variables in the "Configuration" section below:

    MODEL_NAME : str
        Name of the model folder inside evaluation/.  The folder must contain
        a model.py file that exposes:
            get_model(metadata: dict) -> torch.nn.Module
        The model forward() takes (B, 450, 449, c) float32 and returns (B, 6).

    TEST_YEAR : int
        Calendar year of the test data.  All hours t in this year are used,
        provided that t+24h also falls within the dataset.

Metrics (per assignment2.md)
----------------------------
  - RMSE for each of the 5 continuous variables
  - RMSE for APCP_1hr_acc_fcst@surface restricted to samples where true APCP > 2 mm
  - AUC (area under ROC curve) for the binary label APCP > 2 mm,
    using the model's raw APCP prediction as the ranking score
"""

import sys
import importlib.util
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

# ============================================================
# Configuration — edit here to switch models or test periods
# ============================================================

MODEL_NAME = "model"   # folder name inside evaluation/  (e.g. "stub", "my_cnn")
TEST_YEAR  = 2021     # year whose hours form the test set

# ============================================================
# Paths (derived automatically — no need to edit)
# ============================================================

EVAL_DIR    = Path(__file__).parent          # evaluation/
ROOT        = EVAL_DIR.parent                # assignment2_data/
DATASET_DIR = ROOT / "data/assignment2_data/dataset"

# ============================================================
# Load dataset metadata and targets
# ============================================================

print("Loading dataset metadata and targets …")
metadata     = torch.load(DATASET_DIR / "metadata.pt", weights_only=False)
targets_data = torch.load(DATASET_DIR / "targets.pt",  weights_only=False)

times         = targets_data["time"]           # numpy datetime64[ns], shape (T,)
target_values = targets_data["values"]         # torch.Tensor (T, 6) float32
binary_labels = targets_data["binary_label"]   # torch.Tensor (T,) bool
target_vars   = list(targets_data["variable_names"])   # 6 variable names

print(f"  Full dataset : {times[0]}  →  {times[-1]}  ({len(times)} steps)")
print(f"  Target vars  : {target_vars}")

# ============================================================
# Identify test indices
#
#   - t (input time) must fall in TEST_YEAR
#   - target is at t+24h, so index t_idx+24 must be within bounds
# ============================================================

times_years  = times.astype("datetime64[Y]").astype(int) + 1970
in_year_mask = np.where(times_years == TEST_YEAR)[0]
test_indices = in_year_mask[in_year_mask + 24 < len(times)]

print(f"\nTest configuration")
print(f"  Model        : {MODEL_NAME}")
print(f"  Test year    : {TEST_YEAR}")
print(f"  Input  t     : {times[test_indices[0]]}  →  {times[test_indices[-1]]}")
print(f"  Target t+24h : {times[test_indices[0]+24]}  →  {times[test_indices[-1]+24]}")
print(f"  # samples    : {len(test_indices)}")

# ============================================================
# Load model
#
#   Dynamically loads evaluation/<MODEL_NAME>/model.py and calls get_model().
#   Any model folder dropped into evaluation/ works as long as its model.py
#   follows the same interface.
# ============================================================

model_path = EVAL_DIR / MODEL_NAME / "model.py"
if not model_path.exists():
    raise FileNotFoundError(
        f"No model file found at {model_path}\n"
        f"Create evaluation/{MODEL_NAME}/model.py with a get_model(metadata) function."
    )

# Add the model's own directory to sys.path so that any sibling .py files
# (submodules, helpers, etc.) inside the model folder can be imported normally.
model_dir = str(EVAL_DIR / MODEL_NAME)
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

spec   = importlib.util.spec_from_file_location("model", model_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model  = module.get_model(metadata)
model.eval()
print(f"\nLoaded model  : {MODEL_NAME}  ({model.__class__.__name__})")

# ============================================================
# Inference loop
# ============================================================

all_preds   = []
all_targets = []
all_binary  = []

print(f"\nRunning inference …")

with torch.no_grad():
    for step, t_idx in enumerate(test_indices):
        t24_idx = t_idx + 24

        # --- Load input tensor X_t ---
        dt = pd.Timestamp(times[t_idx])
        input_path = (DATASET_DIR / "inputs"
                      / str(dt.year)
                      / f"X_{dt.strftime('%Y%m%d%H')}.pt")
        x = torch.load(input_path, weights_only=True).float()   # (450, 449, c)

        if torch.isnan(x).any():
            continue

        # --- Forward pass (model expects a batch dimension) ---
        pred = model(x.unsqueeze(0)).squeeze(0)   # (6,)

        all_preds.append(pred)
        all_targets.append(target_values[t24_idx])
        all_binary.append(binary_labels[t24_idx])

        if (step + 1) % 500 == 0 or (step + 1) == len(test_indices):
            print(f"  {step+1:>5}/{len(test_indices)}")

preds   = torch.stack(all_preds).float()    # (N, 6)
targets = torch.stack(all_targets).float()  # (N, 6)
binary  = torch.stack(all_binary)           # (N,) bool

# ============================================================
# Metrics
# ============================================================

print("\n" + "=" * 65)
print(f"Results  —  model: {MODEL_NAME}   test year: {TEST_YEAR}")
print("=" * 65)

apcp_var  = "APCP_1hr_acc_fcst@surface"
apcp_idx  = target_vars.index(apcp_var)

for j, var in enumerate(target_vars):
    if var == apcp_var:
        # RMSE only on rainy samples (true APCP > 2 mm)
        rain_mask = targets[:, j] > 2.0
        n_rain = rain_mask.sum().item()
        if n_rain == 0:
            print(f"  {var:42s}  RMSE (true>2mm): N/A  (no rainy samples)")
        else:
            rmse = torch.sqrt(((preds[rain_mask, j] - targets[rain_mask, j]) ** 2).mean()).item()
            print(f"  {var:42s}  RMSE (true>2mm): {rmse:8.4f} mm   [n={n_rain}]")
    else:
        rmse = torch.sqrt(((preds[:, j] - targets[:, j]) ** 2).mean()).item()
        print(f"  {var:42s}  RMSE: {rmse:8.4f}")

# AUC: use raw APCP prediction as ranking score for binary label
scores = preds[:, apcp_idx].numpy()
labels = binary.numpy().astype(int)
n_pos  = labels.sum()
n_neg  = len(labels) - n_pos

if n_pos == 0 or n_neg == 0:
    print(f"\n  {'APCP > 2mm':42s}  AUC: N/A  (only one class present)")
else:
    auc = roc_auc_score(labels, scores)
    print(f"\n  {'APCP > 2mm (binary)':42s}  AUC:  {auc:.4f}   [pos={n_pos}, neg={n_neg}]")

print("=" * 65)
