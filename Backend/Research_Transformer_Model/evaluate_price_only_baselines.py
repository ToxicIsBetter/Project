"""
NeuralEdge — Price-Only Baseline Evaluation
============================================
Trains and evaluates price-only models (Logistic Regression, LSTM)
on the EXACT same train/val/test split as the Dual-Head Transformer,
then produces a proper comparative evaluation table.

Price-only features = OHLCV-derived only (no on-chain, no sentiment).
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report,
)
from datetime import datetime

np.bool = bool
np.int = int
np.float = float
warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)
os.makedirs("price_only_results", exist_ok=True)

print("=" * 70)
print("NEURALEDGE — Price-Only Baseline Evaluation")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ── 1. Load and Merge Data ─────────────────────────────────────────────────
print("\n[1] Loading data...")

ohlcv = pd.read_csv("../../Backend/Core_API_Service/data/clean_ohlcv.csv", parse_dates=["Date"])
onchain = pd.read_csv("../../Backend/Core_API_Service/data/clean_onchain.csv", parse_dates=["Date"])
sentiment = pd.read_csv("../../Backend/Core_API_Service/data/clean_sentiment.csv", parse_dates=["Date"])
trends = pd.read_csv("../../Backend/Core_API_Service/data/clean_google.csv", parse_dates=["Date"])

for frame in [ohlcv, onchain, sentiment, trends]:
    frame["Date"] = frame["Date"].dt.normalize()

df = ohlcv.merge(onchain, on="Date", how="left")
df = df.merge(sentiment, on="Date", how="left")
df = df.merge(trends, on="Date", how="left")
df = df.sort_values("Date").reset_index(drop=True)

START_DATE = "2018-02-01"
df = df[df["Date"] >= START_DATE].reset_index(drop=True)

# Fix Fear & Greed placeholder 50s
fg_cols = ["fear_greed","fg_ma7","fg_ma14","fg_change","fg_change7",
           "fg_extreme_fear","fg_extreme_greed"]
for col in fg_cols:
    if col in df.columns:
        df[col] = df[col].replace(50.0, np.nan).ffill(limit=3).bfill(limit=3).fillna(0.0)

# ── 2. Build Price-Only Features ──────────────────────────────────────────
print("\n[2] Engineering price-only features...")

# Returns at multiple horizons
df["return_1d"] = df["Close"].pct_change(1)
df["return_3d"] = df["Close"].pct_change(3)
df["return_7d"] = df["Close"].pct_change(7)
df["return_14d"] = df["Close"].pct_change(14)

# Moving averages
df["ma_5"] = df["Close"].rolling(5).mean()
df["ma_14"] = df["Close"].rolling(14).mean()
df["ma_50"] = df["Close"].rolling(50).mean()
df["ma_ratio_5"] = df["Close"] / (df["ma_5"] + 1e-8)
df["ma_ratio_14"] = df["Close"] / (df["ma_14"] + 1e-8)
df["ma_ratio_50"] = df["Close"] / (df["ma_50"] + 1e-8)
df["ma_cross"] = (df["ma_5"] > df["ma_14"]).astype(int)

# Volatility
df["vol_5d"] = df["Close"].pct_change().rolling(5).std()
df["vol_14d"] = df["Close"].pct_change().rolling(14).std()
df["vol_ratio"] = df["vol_5d"] / (df["vol_14d"] + 1e-8)

# RSI (14-day)
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / (loss + 1e-8)
df["rsi_14"] = 100 - (100 / (1 + rs))

# Bollinger Band position
bb_std = df["Close"].rolling(20).std()
bb_ma = df["Close"].rolling(20).mean()
df["bb_position"] = (df["Close"] - bb_ma) / (2 * bb_std + 1e-8)

# Price momentum
df["momentum_7d"] = df["Close"].pct_change(7)
df["momentum_14d"] = df["Close"].pct_change(14)
df["momentum_30d"] = df["Close"].pct_change(30)

# Volume ratio
df["vol_ratio_5_30"] = df["Volume"].rolling(5).mean() / (df["Volume"].rolling(30).mean() + 1e-8)

# High-Low range
df["hl_range"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-8)

PRICE_ONLY_FEATURES = [
    "return_1d", "return_3d", "return_7d", "return_14d",
    "ma_ratio_5", "ma_ratio_14", "ma_ratio_50", "ma_cross",
    "vol_5d", "vol_14d", "vol_ratio",
    "rsi_14", "bb_position",
    "momentum_7d", "momentum_14d", "momentum_30d",
    "vol_ratio_5_30", "hl_range",
]

# Filter to available columns
PRICE_ONLY_FEATURES = [c for c in PRICE_ONLY_FEATURES if c in df.columns]
print(f"  Using {len(PRICE_ONLY_FEATURES)} price-only features: {PRICE_ONLY_FEATURES}")

# ── 3. Target Variable ────────────────────────────────────────────────────
print("\n[3] Constructing target variable...")
df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df = df.iloc[:-1].reset_index(drop=True)
df = df.ffill().bfill()
df = df.dropna(subset=PRICE_ONLY_FEATURES + ["target"]).reset_index(drop=True)
print(f"  Final dataset: {len(df)} rows")

class_dist = df["target"].value_counts()
print(f"  Up (1):   {class_dist.get(1,0):,} ({100*class_dist.get(1,0)/len(df):.1f}%)")
print(f"  Down (0): {class_dist.get(0,0):,} ({100*class_dist.get(0,0)/len(df):.1f}%)")

# ── 4. Chronological Split (same as Dual-Head Transformer) ────────────────
print("\n[4] Chronological split...")
TRAIN_END = "2022-12-31"
VAL_END = "2023-12-31"
train = df[df["Date"] <= TRAIN_END].copy().reset_index(drop=True)
val = df[(df["Date"] > TRAIN_END) & (df["Date"] <= VAL_END)].copy().reset_index(drop=True)
test = df[df["Date"] > VAL_END].copy().reset_index(drop=True)

print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

# ── 5. Scale price-only features ──────────────────────────────────────────
scaler_price = StandardScaler()
X_train_price = scaler_price.fit_transform(train[PRICE_ONLY_FEATURES].values)
X_val_price = scaler_price.transform(val[PRICE_ONLY_FEATURES].values)
X_test_price = scaler_price.transform(test[PRICE_ONLY_FEATURES].values)

y_train = train["target"].values
y_val = val["target"].values
y_test = test["target"].values

print(f"  Price-only feature matrix: {X_train_price.shape[1]} features")

# ── 6. Logistic Regression Baseline ──────────────────────────────────────
print("\n[5] Training Logistic Regression (price-only)...")
best_lr_c = 0.001
best_lr_f1 = 0
for c in [0.0001, 0.001, 0.01, 0.1, 1.0]:
    lr_temp = LogisticRegression(penalty="l2", C=c, random_state=42, max_iter=2000)
    lr_temp.fit(X_train_price, y_train)
    preds = lr_temp.predict(X_val_price)
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_lr_f1:
        best_lr_f1 = f1
        best_lr_c = c

lr_model = LogisticRegression(penalty="l2", C=best_lr_c, random_state=42, max_iter=2000)
lr_model.fit(X_train_price, y_train)

lr_val_probs = lr_model.predict_proba(X_val_price)[:, 1]
lr_val_preds = (lr_val_probs > 0.5).astype(int)
lr_val_f1 = f1_score(y_val, lr_val_preds)

# Threshold sweep on validation
best_lr_thresh = 0.5
best_lr_thresh_f1 = lr_val_f1
for t in np.arange(0.25, 0.75, 0.01):
    preds_t = (lr_val_probs > t).astype(int)
    f1_t = f1_score(y_val, preds_t, zero_division=0)
    if f1_t > best_lr_thresh_f1:
        best_lr_thresh_f1 = f1_t
        best_lr_thresh = t

lr_test_probs = lr_model.predict_proba(X_test_price)[:, 1]
lr_test_preds = (lr_test_probs > best_lr_thresh).astype(int)

lr_acc = accuracy_score(y_test, lr_test_preds)
lr_f1 = f1_score(y_test, lr_test_preds, zero_division=0)
lr_precision = precision_score(y_test, lr_test_preds, zero_division=0)
lr_recall = recall_score(y_test, lr_test_preds, zero_division=0)
lr_auc = roc_auc_score(y_test, lr_test_probs)

print(f"  LR Test Accuracy: {lr_acc:.4f} | F1: {lr_f1:.4f} | AUC: {lr_auc:.4f}")
print(f"  Optimal threshold: {best_lr_thresh:.2f}")

# Save LR predictions
np.save("price_only_results/lr_probs_test.npy", lr_test_probs)
np.save("price_only_results/lr_preds_test.npy", lr_test_preds)
np.save("price_only_results/y_test_price.npy", y_test)

# ── 7. LSTM Baseline ───────────────────────────────────────────────────────
print("\n[6] Training LSTM (price-only)...")

SEQ_LEN = 7

def build_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len : i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = build_sequences(X_train_price, y_train, SEQ_LEN)
X_val_seq, y_val_seq = build_sequences(X_val_price, y_val, SEQ_LEN)
X_test_seq, y_test_seq = build_sequences(X_test_price, y_test, SEQ_LEN)

print(f"  LSTM train: {X_train_seq.shape} | val: {X_val_seq.shape} | test: {X_test_seq.shape}")

class PriceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = h[-1]
        return self.fc(h).squeeze(-1)

lstm_model = PriceLSTM(input_dim=len(PRICE_ONLY_FEATURES), hidden_dim=64,
                       num_layers=2, dropout=0.3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model = lstm_model.to(device)

# Class weights
n_up = y_train_seq.sum()
n_down = len(y_train_seq) - n_up
pos_weight = torch.tensor([n_down / (n_up + 1e-8)], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

X_train_t = torch.FloatTensor(X_train_seq)
y_train_t = torch.FloatTensor(y_train_seq)
X_val_t = torch.FloatTensor(X_val_seq)
y_val_t = torch.FloatTensor(y_val_seq)

train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

EPOCHS = 150
PATIENCE = 25
best_val_loss = float("inf")
patience_ctr = 0
best_lstm_state = None

for epoch in range(EPOCHS):
    lstm_model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = lstm_model(xb)
        loss = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
        optimizer.step()

    lstm_model.eval()
    with torch.no_grad():
        val_out = lstm_model(X_val_t.to(device)).squeeze()
        val_probs_lstm_raw = torch.sigmoid(val_out).cpu().numpy()
        val_loss = criterion(val_out, y_val_t.to(device)).item()

    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_lstm_state = {k: v.cpu().clone() for k, v in lstm_model.state_dict().items()}
        patience_ctr = 0
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d} | Val Loss: {val_loss:.4f}")

lstm_model.load_state_dict({k: v.to(device) for k, v in best_lstm_state.items()})
lstm_model.eval()

# LSTM threshold sweep on validation
with torch.no_grad():
    val_out_lstm = lstm_model(X_val_t.to(device)).squeeze()
    val_probs_lstm = torch.sigmoid(val_out_lstm).cpu().numpy()

best_lstm_thresh = 0.5
best_lstm_val_f1 = f1_score(y_val_seq, (val_probs_lstm > 0.5).astype(int), zero_division=0)
for t in np.arange(0.25, 0.75, 0.01):
    preds_t = (val_probs_lstm > t).astype(int)
    f1_t = f1_score(y_val_seq, preds_t, zero_division=0)
    if f1_t > best_lstm_val_f1:
        best_lstm_val_f1 = f1_t
        best_lstm_thresh = t

# Evaluate LSTM on test
X_test_t = torch.FloatTensor(X_test_seq)
with torch.no_grad():
    lstm_out = lstm_model(X_test_t.to(device)).squeeze()
    lstm_test_probs = torch.sigmoid(lstm_out).cpu().numpy()

lstm_test_preds = (lstm_test_probs > best_lstm_thresh).astype(int)

lstm_acc = accuracy_score(y_test_seq, lstm_test_preds)
lstm_f1 = f1_score(y_test_seq, lstm_test_preds, zero_division=0)
lstm_precision = precision_score(y_test_seq, lstm_test_preds, zero_division=0)
lstm_recall = recall_score(y_test_seq, lstm_test_preds, zero_division=0)
lstm_auc = roc_auc_score(y_test_seq, lstm_test_probs)

print(f"  LSTM Test Accuracy: {lstm_acc:.4f} | F1: {lstm_f1:.4f} | AUC: {lstm_auc:.4f}")
print(f"  Optimal threshold: {best_lstm_thresh:.2f}")

np.save("price_only_results/lstm_probs_test.npy", lstm_test_probs)
np.save("price_only_results/lstm_preds_test.npy", lstm_test_preds)

# ── 8. Load Dual-Head Transformer results ─────────────────────────────────
print("\n[7] Loading Dual-Head Transformer results...")

DUALHEAD_RESULTS = {
    "accuracy": 0.6396,
    "f1": 0.62,
    "auc": 0.654,
    "threshold": 0.51,
    "model": "Dual-Head Transformer",
}

# Note: The 63.96% result is from the Grid Search winner (Run 82)
# evaluated on the same test set. If saved predictions exist, use them.
dualhead_pred_path = "processed_model3/model3_probs_test.npy"
dualhead_true_path = "processed_model3/model3_ytrue_test.npy"

if os.path.exists(dualhead_pred_path) and os.path.exists(dualhead_true_path):
    dh_probs = np.load(dualhead_pred_path)
    dh_ytrue = np.load(dualhead_true_path)
    if len(dh_ytrue) == len(y_test_seq):
        DUALHEAD_RESULTS["accuracy"] = accuracy_score(dh_ytrue, (dh_probs > 0.51).astype(int))
        DUALHEAD_RESULTS["f1"] = f1_score(dh_ytrue, (dh_probs > 0.51).astype(int), zero_division=0)
        DUALHEAD_RESULTS["auc"] = roc_auc_score(dh_ytrue, dh_probs)
        print(f"  Loaded from saved predictions — Accuracy: {DUALHEAD_RESULTS['accuracy']:.4f} | F1: {DUALHEAD_RESULTS['f1']:.4f}")

# ── 9. Build Comparison Table ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("[8] BASELINE COMPARISON RESULTS (Test Set: Jan 2024 – Dec 2025)")
print("=" * 70)

models_results = {
    "Majority-Class Baseline": {
        "accuracy": float(np.sum(y_test == 0) / len(y_test)),
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "auc": 0.5,
    },
    "Logistic Regression (Price-Only)": {
        "accuracy": float(lr_acc),
        "f1": float(lr_f1),
        "precision": float(lr_precision),
        "recall": float(lr_recall),
        "auc": float(lr_auc),
    },
    "LSTM (Price-Only)": {
        "accuracy": float(lstm_acc),
        "f1": float(lstm_f1),
        "precision": float(lstm_precision),
        "recall": float(lstm_recall),
        "auc": float(lstm_auc),
    },
    "Dual-Head Transformer (Full)": {
        "accuracy": float(DUALHEAD_RESULTS["accuracy"]),
        "f1": float(DUALHEAD_RESULTS["f1"]),
        "precision": None,
        "recall": None,
        "auc": float(DUALHEAD_RESULTS["auc"]),
    },
}

header = f"{'Model':<35} {'Accuracy':>9} {'F1':>7} {'AUC':>7} {'Precision':>10} {'Recall':>7}"
print("\n" + header)
print("-" * 80)
for name, m in models_results.items():
    prec_s = f"{m['precision']:.4f}" if m["precision"] is not None else "   N/A  "
    rec_s  = f"{m['recall']:.4f}"  if m["recall"]  is not None else "   N/A  "
    print(f"{name:<35} {m['accuracy']:>9.4f} {m['f1']:>7.4f} {m['auc']:>7.4f} {prec_s:>10} {rec_s:>7}")

# ── 10. Save Results ──────────────────────────────────────────────────────
results_json = {
    "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "test_period": "January 2024 onwards",
    "train_samples": int(len(train)),
    "val_samples": int(len(val)),
    "test_samples": int(len(test)),
    "price_only_features": PRICE_ONLY_FEATURES,
    "lstm_optimal_threshold": float(best_lstm_thresh),
    "lr_optimal_threshold": float(best_lr_thresh),
    "models": models_results,
}

with open("price_only_results/baseline_comparison_results.json", "w") as f:
    json.dump(results_json, f, indent=2)

print("\n" + "=" * 70)
print("✅ Price-only baseline evaluation complete!")
print(f"Results saved to price_only_results/baseline_comparison_results.json")
print(f"Test samples: {len(test)} rows | Majority-class acc: {models_results['Majority-Class Baseline']['accuracy']:.4f}")
print(f"Dual-Head Transformer accuracy: {models_results['Dual-Head Transformer (Full)']['accuracy']:.4f}")
print(f"LSTM (price-only) accuracy:     {models_results['LSTM (Price-Only)']['accuracy']:.4f}")
print(f"Logistic Regression (price-only) accuracy: {models_results['Logistic Regression (Price-Only)']['accuracy']:.4f}")
print("=" * 70)