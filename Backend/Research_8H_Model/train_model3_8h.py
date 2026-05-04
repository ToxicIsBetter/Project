"""
MODEL 3 — 8H Dual-Head Transformer Training Pipeline
Trains on 8-hourly data instead of daily
"""

import time
import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from boruta import BorutaPy
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.bool = bool
np.int = int
np.float = float
warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────
DATA_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "btc_8h_full.csv"
)  # Path to 8H data
MODEL_DIR = "processed_model3_8h"
PLOTS_DIR = "plots_8h"
START_DATE = "2018-02-01"
TRAIN_END = "2022-12-31"
VAL_END = "2023-12-31"
SEQ_LEN = 5

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 60)
print("MODEL 3 — 8H Dual-Head Transformer Training")
print("=" * 60)

# ────────────────────────────────────────────────────────────────
# LOAD DATA
# ────────────────────────────────────────────────────────────────
print("\nLoading 8H data...")
df = pd.read_csv(DATA_FILE, parse_dates=["date"], index_col="date")
print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
print(f"Date range: {df.index[0]} → {df.index[-1]}")

# Keep date as column for splitting
df = df.reset_index()
df.rename(columns={"index": "date"}, inplace=True)
df["date"] = pd.to_datetime(df["date"])

# Ensure we have price column
if "price" not in df.columns and "close" in df.columns:
    df["price"] = df["close"]
if "price" not in df.columns:
    raise ValueError("No price column found")

# ────────────────────────────────────────────────────────────────
# TARGET VARIABLE
# ────────────────────────────────────────────────────────────────
print("\nCreating target variable...")
df["target"] = (df["price"].shift(-1) > df["price"]).astype(int)
df = df.iloc[:-1].reset_index(drop=True)

# ────────────────────────────────────────────────────────────────
# DATE RANGE FILTER
# ────────────────────────────────────────────────────────────────
print(f"\nFiltering date range: {START_DATE} onwards")
df = df[df["date"] >= START_DATE].reset_index(drop=True)

# ────────────────────────────────────────────────────────────────
# FEATURE SELECTION
# ────────────────────────────────────────────────────────────────
print("\nSelecting features...")

# Identify feature columns (exclude metadata and target)
exclude_cols = ["target", "date", "close_time"]
feature_cols = [
    c
    for c in df.columns
    if c not in exclude_cols
    and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
]

# Handle missing values - keep date column for splitting
df = df[feature_cols + ["target", "date"]].copy()
df = df.ffill().bfill().dropna()

print(f"Using {len(feature_cols)} features")

# ────────────────────────────────────────────────────────────────
# TRAIN/VAL/TEST SPLIT
# ────────────────────────────────────────────────────────────────
print("\nSplitting data...")
# Use date column for splitting
train = df[df["date"] <= TRAIN_END].reset_index(drop=True)
val = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].reset_index(drop=True)
test = df[df["date"] > VAL_END].reset_index(drop=True)

print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

# ────────────────────────────────────────────────────────────────
# FEATURE SCALING
# ────────────────────────────────────────────────────────────────
print("\nScaling features...")
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train[feature_cols].values)
val_scaled = scaler.transform(val[feature_cols].values)
test_scaled = scaler.transform(test[feature_cols].values)

joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

# ────────────────────────────────────────────────────────────────
# SEQUENCES
# ────────────────────────────────────────────────────────────────
print(f"\nCreating sequences (len={SEQ_LEN})...")


def create_sequences(data, targets, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len : i])
        y.append(targets[i])
    return np.array(X), np.array(y)


y_train = train["target"].values
y_val = val["target"].values
y_test = test["target"].values

X_train, y_train_seq = create_sequences(train_scaled, y_train, SEQ_LEN)
X_val, y_val_seq = create_sequences(val_scaled, y_val, SEQ_LEN)
X_test, y_test_seq = create_sequences(test_scaled, y_test, SEQ_LEN)

print(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

# Save sequences
np.save(f"{MODEL_DIR}/X_train.npy", X_train)
np.save(f"{MODEL_DIR}/y_train.npy", y_train_seq)
np.save(f"{MODEL_DIR}/X_val.npy", X_val)
np.save(f"{MODEL_DIR}/y_val.npy", y_val_seq)
np.save(f"{MODEL_DIR}/X_test.npy", X_test)
np.save(f"{MODEL_DIR}/y_test.npy", y_test_seq)

# ────────────────────────────────────────────────────────────────
# DUAL-HEAD TRANSFORMER MODEL
# ────────────────────────────────────────────────────────────────
print("\nBuilding Dual-Head Transformer...")


class DualHeadTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head1 = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.head2 = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # Take last timestep
        h1 = self.dropout(self.head1(x))
        h2 = self.dropout(self.head2(x))
        return h1 + h2  # Combine heads


model = DualHeadTransformer(
    input_dim=len(feature_cols), d_model=64, nhead=8, num_layers=2, dropout=0.2
)

print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# ────────────────────────────────────────────────────────────────
# TRAINING
# ────────────────────────────────────────────────────────────────
print("\nTraining...")
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5
)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train_seq)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.FloatTensor(y_val_seq)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

best_val_loss = float("inf")
patience_counter = 0
EPOCHS = 100
PATIENCE = 15

train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
for batch_x, batch_y in train_loader:
    optimizer.zero_grad()
    out = model(batch_x)
    loss = criterion(out.squeeze(), batch_y)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        val_out = model(X_val_t).squeeze()
        val_loss = criterion(val_out, y_val_t)

    train_losses.append(epoch_loss / len(train_loader))
    val_losses.append(val_loss.item())

    scheduler.step(val_loss.item())

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        torch.save(model.state_dict(), f"{MODEL_DIR}/model_best.pth")
        patience_counter = 0
        print(
            f"Epoch {epoch + 1:3d} | Train: {epoch_loss / len(train_loader):.4f} | Val: {val_loss.item():.4f} ★"
        )
    else:
        patience_counter += 1
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d} | Train: {epoch_loss / len(train_loader):.4f} | Val: {val_loss.item():.4f}"
            )

    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch + 1}")
        break

# Load best model
model.load_state_dict(torch.load(f"{MODEL_DIR}/model_best.pth"))
print(f"\nBest model saved (val_loss: {best_val_loss:.4f})")

# ────────────────────────────────────────────────────────────────
# EVALUATION
# ────────────────────────────────────────────────────────────────
print("\nEvaluating on test set...")
model.eval()
with torch.no_grad():
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test_seq)
    test_out = model(X_test_t).squeeze()
    test_probs = torch.sigmoid(test_out).numpy()
    test_preds = (test_probs > 0.5).astype(int)

    accuracy = accuracy_score(y_test_seq, test_preds)
    f1 = f1_score(y_test_seq, test_preds)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test_seq, test_preds)}")
    print(f"\nClassification Report:\n{classification_report(y_test_seq, test_preds)}")

# Save metrics
metrics = {
    "accuracy": float(accuracy),
    "f1": float(f1),
    "best_val_loss": float(best_val_loss),
    "epochs_trained": epoch + 1,
    "feature_cols": feature_cols,
}
with open(f"{MODEL_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ────────────────────────────────────────────────────────────────
# PLOTS
# ────────────────────────────────────────────────────────────────
print("\nGenerating plots...")

# Training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curves")
plt.legend()
plt.savefig(f"{PLOTS_DIR}/training_curves.png", dpi=150)
plt.close()

# ROC Curve
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test_seq, test_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(f"{PLOTS_DIR}/roc_curve.png", dpi=150)
plt.close()

print(f"\n✅ Training complete!")
print(f"   Model: {MODEL_DIR}/model_best.pth")
print(f"   Metrics: {MODEL_DIR}/metrics.json")
print(f"   Plots: {PLOTS_DIR}/")
