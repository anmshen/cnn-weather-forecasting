import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR   = '/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset'
INPUT_DIR  = os.path.join(DATA_DIR, 'inputs')
TARGET_PT  = os.path.join(DATA_DIR, 'targets.pt')
META_PT    = os.path.join(DATA_DIR, 'metadata.pt')

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
LR         = 3e-4
EPOCHS     = 20

MODEL_PATH = '/cluster/tufts/c26sp1cs0137/ashen05/best_model.pt'
NORM_CACHE = '/cluster/tufts/c26sp1cs0137/ashen05/norm_stats.pt'

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

# ── Split ────────────────────────────────────────────────────────────────────
n = len(times)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

train_idx = list(range(0, train_end))
val_idx   = list(range(train_end, val_end))
test_idx  = list(range(val_end, n))

# ── Normalize targets (IMPORTANT) ────────────────────────────────────────────
cont_mean = target_vals[train_idx].mean(dim=0)
cont_std  = target_vals[train_idx].std(dim=0).clamp(min=1e-6)

target_vals = (target_vals - cont_mean) / cont_std

# ── Input normalization ──────────────────────────────────────────────────────
def compute_norm_stats(indices, n_vars=42, sample_every=24):
    sums  = torch.zeros(n_vars)
    sqsum = torch.zeros(n_vars)
    count = 0

    for i in indices[::sample_every]:
        path = FILE_INDEX.get(time_to_filename(times[i]))
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
        self.indices = [i for i in indices if FILE_INDEX.get(time_to_filename(times[i])) is not None]
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

        y_cont = torch.nan_to_num(target_vals[i], nan=0.0)
        y_bin  = torch.nan_to_num(target_bin[i], nan=0.0)

        return x, y_cont, y_bin

# ── Model ────────────────────────────────────────────────────────────────────
class WeatherCNN(nn.Module):
    def __init__(self, in_channels=42):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        self.cont_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 6),
        )

        self.bin_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        feat = self.encoder(x)
        return self.cont_head(feat), self.bin_head(feat).squeeze(1)

# ── Training ─────────────────────────────────────────────────────────────────
def main():
    train_loader = DataLoader(WeatherDataset(train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(WeatherDataset(val_idx),   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(WeatherDataset(test_idx),  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

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

    # ── Test (with denormalization) ─────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print("No saved model — skipping test")
        return

    print("Evaluating test set...")
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    sum_sq_error = torch.zeros(5, device=DEVICE)
    count = 0

    with torch.no_grad():
        for x, y_cont, _ in test_loader:
            x, y_cont = x.to(DEVICE), y_cont.to(DEVICE)
            pred, _ = model(x)

            # denormalize
            pred = pred * cont_std.to(DEVICE) + cont_mean.to(DEVICE)
            y_cont = y_cont * cont_std.to(DEVICE) + cont_mean.to(DEVICE)

            err = (pred[:, :5] - y_cont[:, :5]) ** 2
            sum_sq_error += err.sum(dim=0)
            count += err.shape[0]

    rmse = torch.sqrt(sum_sq_error / count)

    print("\nTest RMSE:")
    for i, v in enumerate(rmse):
        print(f"Var {i}: {v.item():.4f}")

if __name__ == "__main__":
    main()