"""
transformer_model.py — PyTorch Transformer for BTC Direction Prediction
Industry-standard Transformer encoder classifier with positional encoding.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
SEQ_LEN = 30
D_MODEL = 64          # Transformer embedding dimension
NHEAD = 4             # Multi-head attention heads
NUM_LAYERS = 2        # Transformer encoder layers
DIM_FF = 128          # Feed-forward dimension
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
PATIENCE = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = Path('data')

print(f"🔧 Device: {DEVICE}")

# ============================================================
# 1. LOAD DATA
# ============================================================
print("📊 Loading data...")
df = pd.read_csv(DATA_DIR / 'btc_features.csv', parse_dates=['date'], index_col='date')
with open(DATA_DIR / 'feature_cols.txt') as f:
    feature_cols = f.read().strip().split('\n')

# ============================================================
# 2. PREPARE SEQUENCES
# ============================================================
X_raw = df[feature_cols].values
y_raw = df['target'].values
dates = df.index

train_mask = dates < '2024-07-01'
scaler = StandardScaler()
scaler.fit(X_raw[train_mask])
X_scaled = scaler.transform(X_raw)
X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

def make_sequences(X, y, dates, seq_len):
    xs, ys, ds = [], [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
        ds.append(dates[i+seq_len])
    return np.array(xs), np.array(ys), ds

X_seq, y_seq, d_seq = make_sequences(X_scaled, y_raw, dates, SEQ_LEN)
d_seq = pd.DatetimeIndex(d_seq)

train_idx = d_seq < '2024-07-01'
val_idx = (d_seq >= '2024-07-01') & (d_seq < '2025-01-01')
test_idx = d_seq >= '2025-01-01'

X_train, y_train = X_seq[train_idx], y_seq[train_idx]
X_val, y_val = X_seq[val_idx], y_seq[val_idx]
X_test, y_test = X_seq[test_idx], y_seq[test_idx]

print(f"   Sequences — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

def to_loader(X, y, batch_size, shuffle=False):
    ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = to_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
val_loader = to_loader(X_val, y_val, BATCH_SIZE)
test_loader = to_loader(X_test, y_test, BATCH_SIZE)

# ============================================================
# 3. TRANSFORMER MODEL
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_ff, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling over sequence
        return self.classifier(x).squeeze(-1)


model = TransformerClassifier(
    input_dim=len(feature_cols),
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_ff=DIM_FF,
    dropout=DROPOUT
).to(DEVICE)

print(f"\n🏗️  Transformer Model: {sum(p.numel() for p in model.parameters()):,} parameters")

# ============================================================
# 4. TRAINING
# ============================================================
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_loss = float('inf')
best_epoch = 0
train_losses, val_losses = [], []

print(f"\n🔥 Training Transformer ({EPOCHS} epochs max, patience={PATIENCE})...\n")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
    train_loss = epoch_loss / len(train_loader)
    train_losses.append(train_loss)
    scheduler.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            pred = model(X_batch)
            val_loss += criterion(pred, y_batch).item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), DATA_DIR / 'transformer_best.pt')
    elif epoch - best_epoch >= PATIENCE:
        print(f"   ⏹ Early stopping at epoch {epoch+1} (best: {best_epoch+1})")
        break

# ============================================================
# 5. EVALUATE ON TEST SET
# ============================================================
model.load_state_dict(torch.load(DATA_DIR / 'transformer_best.pt', weights_only=True))
model.eval()

all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        probs = model(X_batch).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

y_pred = np.array(all_preds)
y_prob = np.array(all_probs)
y_true = np.array(all_labels)

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"\n{'='*50}")
print(f"  TRANSFORMER TEST RESULTS")
print(f"{'='*50}")
print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")

result = {
    'name': 'Transformer',
    'accuracy': float(acc),
    'precision': float(prec),
    'recall': float(rec),
    'f1': float(f1),
    'probabilities': y_prob.tolist(),
    'predictions': y_pred.tolist(),
    'labels': y_true.tolist(),
    'train_losses': train_losses,
    'val_losses': val_losses,
}
with open(DATA_DIR / 'transformer_results.json', 'w') as f:
    json.dump(result, f)

print(f"\n✅ Transformer results saved to {DATA_DIR / 'transformer_results.json'}")
