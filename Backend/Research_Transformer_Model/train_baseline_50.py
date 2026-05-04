"""
BASELINE MODEL - Price + On-chain ONLY (no sentiment)
Target: 50%+ accuracy without bias
"""
import os, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

os.makedirs('processed_model3', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("="*60)
print("BASELINE MODEL - Price + On-chain Only")
print("Target: 50%+ Accuracy")
print("="*60)

# Load data - NO SENTIMENT
print("\nLoading data...")
ohlcv = pd.read_csv("../../data/train_ohlcv.csv", parse_dates=["Date"])
onchain = pd.read_csv("../../data/train_onchain.csv", parse_dates=["Date"])

print(f"  OHLCV: {len(ohlcv)} rows")
print(f"  On-chain: {len(onchain)} rows")

# Merge
df = ohlcv.merge(onchain, on="Date", how="left")
print(f"\n  Merged: {df.shape}")

# Filter date
START_DATE = '2018-02-01'
df = df[df['Date'] >= START_DATE].reset_index(drop=True)
print(f"  After {START_DATE}: {len(df)} rows")

# Create target
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.iloc[:-1].reset_index(drop=True)

# Select STRONG features only (no sentiment)
feature_cols = [
    'Close', 'Volume',  # Price
    'AdrActCnt', 'TxCnt', 'HashRate', 'BlkCnt', 'CapMVRVCur'  # On-chain
]
feature_cols = [c for c in feature_cols if c in df.columns]

# Drop NaN
df = df[feature_cols + ['target']].dropna()
print(f"  After dropping NaN: {len(df)} rows")

if len(df) < 100:
    print("ERROR: Not enough data!")
    exit(1)

# Class balance
class_dist = df['target'].value_counts()
print(f"\n  Class distribution:")
print(f"    Up (1):   {class_dist.get(1, 0):,} ({100*class_dist.get(1, 0)/len(df):.1f}%)")
print(f"    Down (0): {class_dist.get(0, 0):,} ({100*class_dist.get(0, 0)/len(df):.1f}%)")

# Prepare data
print("\n" + "="*60)
print("Preparing Data")
print("="*60)

X = df[feature_cols].values
y = df['target'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'processed_model3/scaler_baseline.pkl')

# Split 70/15/15
n = len(X_scaled)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, X_val, X_test = X_scaled[:train_end], X_scaled[train_end:val_end], X_scaled[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Sequences
SEQ_LEN = 3

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LEN)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, SEQ_LEN)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, SEQ_LEN)

print(f"  Sequences - Train: {len(X_train_seq)} | Val: {len(X_val_seq)} | Test: {len(X_test_seq)}")

# Model
print("\n" + "="*60)
print("Building Model")
print("="*60)

class BaselineModel(nn.Module):
    def __init__(self, input_dim, d_model=32, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = nn.Linear(d_model, 1)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.classifier(x).squeeze()

model = BaselineModel(input_dim=len(feature_cols))
print(f"  Features: {len(feature_cols)}")
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

# Train
print("\n" + "="*60)
print("Training")
print("="*60)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

X_train_t = torch.FloatTensor(X_train_seq)
y_train_t = torch.FloatTensor(y_train_seq)
X_val_t = torch.FloatTensor(X_val_seq)
y_val_t = torch.FloatTensor(y_val_seq)

train_loader = DataLoader(list(zip(X_train_t, y_train_t)), batch_size=32, shuffle=True)

best_val_acc = 0
best_state = None
patience_counter = 0

for epoch in range(100):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    model.eval()
    with torch.no_grad():
        val_out = model(X_val_t)
        val_loss = criterion(val_out, y_val_t)
        val_preds = (torch.sigmoid(val_out) > 0.5).numpy()
        val_acc = accuracy_score(y_val_seq, val_preds)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
        print(f"  Epoch {epoch+1:3d} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc:.4f} ★")
    else:
        patience_counter += 1
        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc:.4f}")
    
    if patience_counter >= 20:
        break

if best_state:
    model.load_state_dict(best_state)
    torch.save(best_state, 'processed_model3/baseline_best.pt')

print(f"\n  Best Val Acc: {best_val_acc:.4f}")

# Test
print("\n" + "="*60)
print("Test Results")
print("="*60)

model.eval()
with torch.no_grad():
    X_test_t = torch.FloatTensor(X_test_seq)
    test_out = model(X_test_t)
    test_preds = (torch.sigmoid(test_out) > 0.5).numpy()
    
    acc = accuracy_score(y_test_seq, test_preds)
    f1 = f1_score(y_test_seq, test_preds)
    
    print(f"\n  Test Accuracy: {acc:.4f}")
    print(f"  Test F1: {f1:.4f}")
    
    cm = confusion_matrix(y_test_seq, test_preds)
    print(f"\n  Confusion Matrix:")
    print(f"    [[{cm[0,0]} {cm[0,1]}]")
    print(f"     [{cm[1,0]} {cm[1,1]}]]")
    
    # Save
    metrics = {
        'accuracy': float(acc),
        'f1': float(f1),
        'best_val_acc': float(best_val_acc),
        'features': feature_cols
    }
    with open('processed_model3/baseline_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

print("\n✅ BASELINE COMPLETE!")
