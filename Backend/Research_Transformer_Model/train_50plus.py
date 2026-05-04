"""
MODEL3_OnChain - 50%+ Accuracy Version
- Fewer, stronger features only
- No class weighting (creates bias)
- Proper regularization
- Early stopping on validation
"""
import os, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)
os.makedirs('processed_model3', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("="*60)
print("MODEL3_OnChain - 50%+ Accuracy Training")
print("="*60)

# ────────────────────────────────────────────────────────────────
# LOAD CLEAN DATA
# ────────────────────────────────────────────────────────────────
print("\nLoading data...")

ohlcv = pd.read_csv("../../data/train_ohlcv.csv", parse_dates=["Date"])
onchain = pd.read_csv("../../data/train_onchain.csv", parse_dates=["Date"])
sentiment = pd.read_csv("../../data/train_sentiment_fixed.csv", parse_dates=["Date"])
google = pd.read_csv("../../data/train_google.csv", parse_dates=["Date"])

print(f"  OHLCV: {len(ohlcv)} rows")
print(f"  On-chain: {len(onchain)} rows")
print(f"  Sentiment: {len(sentiment)} rows")
print(f"  Google: {len(google)} rows")

# Merge
df = ohlcv.merge(onchain, on="Date", how="left")
df = df.merge(sentiment, on="Date", how="left")
df = df.merge(google, on="Date", how="left")

print(f"\n  Merged: {df.shape}")
print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")

# Filter date
START_DATE = '2018-02-01'
df = df[df['Date'] >= START_DATE].reset_index(drop=True)
print(f"  After {START_DATE}: {len(df)} rows")

# Create target BEFORE dropping NaN
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.iloc[:-1].reset_index(drop=True)

# Select ONLY strong, non-leaky features
feature_cols = [
    # Price features (6)
    'Open', 'High', 'Low', 'Close', 'Volume',
    # On-chain (6) - only the most reliable
    'AdrActCnt', 'TxCnt', 'HashRate', 'BlkCnt', 'CapMVRVCur', 'FeeTotNtv',
    # Sentiment (3) - only core metrics
    'fear_greed', 'fg_ma7', 'fg_ma14'
]

# Keep only columns that exist
feature_cols = [c for c in feature_cols if c in df.columns]
print(f"\n  Using {len(feature_cols)} features: {feature_cols}")

# Drop rows with NaN in features OR target
df = df[feature_cols + ['target']].dropna()
print(f"  After dropping NaN: {len(df)} rows")

# Check class balance
class_dist = df['target'].value_counts()
print(f"\n  Class distribution:")
print(f"    Up (1):   {class_dist.get(1, 0):,} ({100*class_dist.get(1, 0)/len(df):.1f}%)")
print(f"    Down (0): {class_dist.get(0, 0):,} ({100*class_dist.get(0, 0)/len(df):.1f}%)")

# ────────────────────────────────────────────────────────────────
# PREPARE DATA
# ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Preparing Data")
print("="*60)

X = df[feature_cols].values
y = df['target'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'processed_model3/scaler.pkl')

# Chronological split (70/15/15)
n = len(X_scaled)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, X_val, X_test = X_scaled[:train_end], X_scaled[train_end:val_end], X_scaled[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Create sequences (seq_len=3 for simplicity)
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

# Save
np.save('processed_model3/X_train.npy', X_train_seq)
np.save('processed_model3/y_train.npy', y_train_seq)
np.save('processed_model3/X_val.npy', X_val_seq)
np.save('processed_model3/y_val.npy', y_val_seq)
np.save('processed_model3/X_test.npy', X_test_seq)
np.save('processed_model3/y_test.npy', y_test_seq)

# ────────────────────────────────────────────────────────────────
# BUILD SIMPLE MODEL
# ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Building Model")
print("="*60)

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, d_model=32, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # Last timestep
        return self.classifier(x)

model = SimpleTransformer(input_dim=len(feature_cols))
print(f"  Input: {len(feature_cols)} features")
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

# ────────────────────────────────────────────────────────────────
# TRAIN
# ────────────────────────────────────────────────────────────────
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

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

best_val_acc = 0
best_model_state = None
patience_counter = 0
EPOCHS = 100
PATIENCE = 20

train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        out = model(batch_x).squeeze()
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_out = model(X_val_t).squeeze()
        val_loss = criterion(val_out, y_val_t)
        val_preds = (torch.sigmoid(val_out) > 0.5).numpy()
        val_acc = accuracy_score(y_val_seq, val_preds)
    
    train_losses.append(epoch_loss / len(train_loader))
    val_losses.append(val_loss.item())
    
    # Track accuracy
    train_out = model(X_train_t).squeeze()
    train_preds = (torch.sigmoid(train_out) > 0.5).numpy()
    train_acc = accuracy_score(y_train_seq, train_preds)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    scheduler.step(val_loss.item())
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        print(f"  Epoch {epoch+1:3d} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc:.4f} ★ BEST")
    else:
        patience_counter += 1
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc:.4f}")
    
    if patience_counter >= PATIENCE:
        print(f"  Early stopping at epoch {epoch+1}")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, 'processed_model3/model3_best.pt')

print(f"\n  Best validation accuracy: {best_val_acc:.4f}")

# ────────────────────────────────────────────────────────────────
# EVALUATE
# ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Test Evaluation")
print("="*60)

model.eval()
with torch.no_grad():
    X_test_t = torch.FloatTensor(X_test_seq)
    test_out = model(X_test_t).squeeze()
    test_probs = torch.sigmoid(test_out).numpy()
    test_preds = (test_probs > 0.5).astype(int)
    
    acc = accuracy_score(y_test_seq, test_preds)
    f1 = f1_score(y_test_seq, test_preds)
    
    print(f"\n  Test Accuracy: {acc:.4f}")
    print(f"  Test F1 Score: {f1:.4f}")
    
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_test_seq, test_preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"    [[{tn} {fp}]")
    print(f"     [{fn} {tp}]]")
    print(f"\n  Classification Report:")
    print(classification_report(y_test_seq, test_preds, digits=4))
    
    # Save metrics
    metrics = {
        'accuracy': float(acc),
        'f1': float(f1),
        'best_val_acc': float(best_val_acc),
        'epochs_trained': epoch + 1,
        'features': feature_cols,
        'threshold': 0.5
    }
    with open('processed_model3/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

# Plots
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('plots/training_curves.png', dpi=150)
plt.close()

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print(f"\nModel: processed_model3/model3_best.pt")
print(f"Metrics: processed_model3/metrics.json")
print(f"Plots: plots/")
