#!/usr/bin/env python3
"""
Clean, improved CNN-BiLSTM training script for 9x81 sensor windows.
- Scales data once in `main()` (per-channel across samples & timesteps)
- Dataset only formats + augments
- Deeper conv encoder (Conv-Conv with stride-downsampling)
- LayerNorm before LSTM, Bi-LSTM with 2 layers + dropout, mean pooling
- OneCycleLR scheduler (per-batch)
- Useful augmentations: amplitude scale, time-warp (simple resample), temporal dropout, gaussian noise
- Weighted loss for class imbalance
- Gradient clipping, model checkpointing, early stopping, clean logging
"""
import os
import sys
import glob
import json
import math
import joblib
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# -------------------------
# Data augmentations
# -------------------------
def random_amplitude_scale(x: np.ndarray, min_scale=0.9, max_scale=1.1):
    """Scale each channel by a random factor (per-sample). x shape (C, T)"""
    scales = np.random.uniform(min_scale, max_scale, size=(x.shape[0], 1)).astype(np.float32)
    return x * scales

def random_temporal_dropout(x: np.ndarray, drop_prob=0.05):
    """Randomly dropout (zero) some timesteps across all channels."""
    mask = np.random.rand(x.shape[1]) > drop_prob
    x = x.copy()
    x[:, ~mask] = 0.0
    return x

def random_time_warp(x: np.ndarray, max_warp=0.08):
    """Simple time-warp via linear resampling (stretch/squeeze)."""
    c, t = x.shape
    warp = 1.0 + np.random.uniform(-max_warp, max_warp)
    new_t = max(2, int(round(t * warp)))
    new_idx = np.linspace(0, t - 1, new_t)
    orig_idx = np.arange(t)
    warped = np.stack([np.interp(new_idx, orig_idx, x[ch]) for ch in range(c)], axis=0)
    # if warped is longer, center-crop; if shorter, pad
    if new_t > t:
        start = (new_t - t) // 2
        warped = warped[:, start:start + t]
    elif new_t < t:
        pad = np.zeros((c, t), dtype=warped.dtype)
        start = (t - new_t) // 2
        pad[:, start:start + new_t] = warped
        warped = pad
    return warped.astype(np.float32)

# -------------------------
# Dataset
# -------------------------
class SensorDataset(Dataset):
    """
    Expects data shaped (N, C, T) as np.float32 and labels (N,)
    Does NOT fit/transform scaler. Augmentations applied on-the-fly.
    """
    def __init__(self, data: np.ndarray, labels: np.ndarray, augment: bool = False):
        assert data.ndim == 3 and data.shape[1] == 9 and data.shape[2] == 81
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].copy()  # shape (C, T)
        y = int(self.labels[idx])

        if self.augment:
            # apply augmentations with some probabilities
            if np.random.rand() < 0.5:
                x = random_amplitude_scale(x, 0.9, 1.1)
            if np.random.rand() < 0.4:
                x = random_time_warp(x, max_warp=0.06)
            if np.random.rand() < 0.5:
                x = random_temporal_dropout(x, drop_prob=0.03)
            if np.random.rand() < 0.5:
                noise = np.random.normal(0, 0.01, size=x.shape).astype(np.float32)
                x = x + noise

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

# -------------------------
# Model
# -------------------------
class ConvBlock(nn.Module):
    """Conv1D block: Conv -> BN -> ReLU"""
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ConvBiLSTM(nn.Module):
    """
    Encoder: stack of convolutional blocks with stride-downsampling (no MaxPool)
    BiLSTM: 2-layer bidirectional with dropout, LayerNorm before LSTM
    Classification head: FC layers with dropout
    """
    def __init__(self, in_channels=9, conv_channels=128, lstm_hidden=128, dropout=0.3, num_classes=10):
        super().__init__()
        # conv encoder: (C=9, T=81) -> downsample twice (T/4 ~ 20)
        self.enc = nn.Sequential(
            ConvBlock(in_channels, conv_channels//2, kernel_size=7, stride=1),
            ConvBlock(conv_channels//2, conv_channels, kernel_size=5, stride=2),   # downsample x2
            ConvBlock(conv_channels, conv_channels, kernel_size=5, stride=1),
            ConvBlock(conv_channels, conv_channels, kernel_size=3, stride=2),      # downsample x2 -> total x4
        )
        self.dropout = nn.Dropout(dropout)

        # LayerNorm applied over channel dimension (after transpose to [B, T, C])
        self.layer_norm = None  # will initialize lazily when forward sees shape

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=0.25,
            batch_first=True,
            bidirectional=True
        )

        fc_input = lstm_hidden * 2  # bidirectional

        self.head = nn.Sequential(
            nn.Linear(fc_input, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [B, C=9, T=81]
        x = x.float()
        x = self.enc(x)  # -> [B, C_enc, T_enc]
        # transpose to [B, T_enc, C_enc] for LSTM
        x = x.transpose(1, 2)
        if self.layer_norm is None:
            # initialize LayerNorm to normalize over channel dim (C_enc)
            self.layer_norm = nn.LayerNorm(x.size(-1)).to(x.device)
        x = self.layer_norm(x)
        x, _ = self.lstm(x)  # x: [B, T_enc, 2*lstm_hidden]
        # mean pooling over time
        x = x.mean(dim=1)
        x = self.head(x)
        return x

# -------------------------
# Data loading / preprocessing
# -------------------------
def load_and_stack_csvs(data_folder: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load CSVs, skip empty files, return (data, labels).
    Assumes sensor columns are 3:732 (729 cols) and label is column 0.
    """
    csv_files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_folder}")

    all_data = []
    all_labels = []
    skipped = 0
    for f in csv_files:
        df = pd.read_csv(f)
        if len(df) == 0:
            skipped += 1
            continue
        # read sensor window (729 cols)
        sensor_data = df.iloc[:, 3:732].values  # shape (n_samples, 729)
        labels = df.iloc[:, 0].astype(int).values
        all_data.append(sensor_data)
        all_labels.append(labels)

    if skipped:
        print(f"Skipped {skipped} empty CSV files")

    data = np.vstack(all_data)
    labels = np.hstack(all_labels)
    return data, labels

def scale_data_per_channel(data, scaler=None, fit=False):
    N = data.shape[0]
    reshaped = data.reshape(N, 9, 81).transpose(0, 2, 1).reshape(-1, 9)

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        scaler.fit(reshaped)

    scaled = scaler.transform(reshaped)
    scaled = scaled.reshape(N, 81, 9).transpose(0, 2, 1)

    return scaled, scaler


# -------------------------
# Training & validation loops
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            # OneCycleLR expects step per batch
            scheduler.step()
        running_loss += loss.item() * xb.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            running_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

# -------------------------
# Main
# -------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python train_model.py <data_folder>")
        sys.exit(1)
    data_folder = sys.argv[1]

    # Hyperparameters
    hyper = {
        "conv_channels": 128,
        "lstm_hidden": 128,
        "dropout": 0.3,
        "lr": 1e-3,           # nominal lr (used by optimizer)
        "max_lr": 1e-2,       # max lr for OneCycle
        "weight_decay": 1e-4,
        "batch_size": 128,
        "num_epochs": 100,
        "patience": 15,
        "num_workers": 4,
    }
    print("Hyperparameters:")
    for k, v in hyper.items():
        print(f"  {k}: {v}")

    # Load raw data
    print("Loading CSV files...")
    raw_data, raw_labels = load_and_stack_csvs(data_folder)
    print(f"Total samples: {raw_data.shape[0]}, raw shape: {raw_data.shape}")

    # cast
    raw_data = raw_data.astype(np.float32)

    # train/val split (stratified-ish by random permutation)
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(raw_data))
    split = int(0.8 * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]
    train_raw, val_raw = raw_data[train_idx], raw_data[val_idx]
    train_labels, val_labels = raw_labels[train_idx], raw_labels[val_idx]

    # Fit scaler on train only and transform both
    print("Fitting scaler on training data...")
    scaler = StandardScaler()
    train_scaled, scaler = scale_data_per_channel(train_raw, scaler=scaler, fit=True)
    val_scaled, _ = scale_data_per_channel(val_raw, scaler=scaler, fit=False)  # reuse fitted scaler
    joblib.dump(scaler, "sensor_scaler.pkl")
    print("Scaler saved to sensor_scaler.pkl")

    # Create datasets
    train_ds = SensorDataset(train_scaled, train_labels, augment=True)
    val_ds = SensorDataset(val_scaled, val_labels, augment=False)
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Compute class weights
    classes = np.unique(train_labels)
    cw = class_weight.compute_class_weight("balanced", classes=classes, y=train_labels)
    class_weights = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
    print(f"Class weights: {cw}")

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=hyper["batch_size"], shuffle=True,
                              num_workers=hyper["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=hyper["batch_size"], shuffle=False,
                            num_workers=hyper["num_workers"], pin_memory=True)

    # Model: infer num_classes from labels
    num_classes = int(np.max(raw_labels) + 1)
    model = ConvBiLSTM(in_channels=9, conv_channels=hyper["conv_channels"],
                       lstm_hidden=hyper["lstm_hidden"], dropout=hyper["dropout"],
                       num_classes=num_classes).to(DEVICE)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Loss, optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=hyper["lr"], weight_decay=hyper["weight_decay"])

    # OneCycleLR requires total_steps: num_epochs * steps_per_epoch
    steps_per_epoch = max(1, len(train_loader))
    total_steps = hyper["num_epochs"] * steps_per_epoch
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                              max_lr=hyper["max_lr"],
                                              total_steps=total_steps,
                                              pct_start=0.1,
                                              anneal_strategy="cos",
                                              final_div_factor=1e4)

    # Training state
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    print("Starting training...")
    for epoch in range(hyper["num_epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, DEVICE)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, DEVICE)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch+1}/{hyper['num_epochs']}]  LR={lr:.6e}  Train: {train_loss:.4f}, {train_acc:.1f}%  Val: {val_loss:.4f}, {val_acc:.1f}%")

        # Checkpoint best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "hyper": hyper,
            }, "best_model.pth")
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= hyper["patience"]:
            print(f"Early stopping at epoch {epoch+1} (patience {hyper['patience']} reached)")
            break

    # Final metrics
    print("Training complete.")
    if best_epoch >= 0:
        print(f"Best epoch: {best_epoch+1}  Best val loss: {best_val_loss:.4f}  Best val acc: {history['val_acc'][best_epoch]:.2f}%")

    # Save history and plots
    with open("training_history.json", "w") as fh:
        json.dump(history, fh, indent=2)
    print("Saved training_history.json")

    # Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    if best_epoch >= 0:
        plt.axvline(best_epoch, color="r", linestyle="--")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    if best_epoch >= 0:
        plt.axvline(best_epoch, color="r", linestyle="--")
    plt.xlabel("epoch"); plt.ylabel("accuracy (%)"); plt.legend(); plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_results.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved training_results.pdf")

    # Save final model (best already saved)
    print("Done. Best model: best_model.pth, scaler: sensor_scaler.pkl, history: training_history.json")

if __name__ == "__main__":
    main()
