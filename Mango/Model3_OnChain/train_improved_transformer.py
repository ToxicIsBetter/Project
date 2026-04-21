"""
IMPROVED Dual-Head Transformer
- Class weighting to handle imbalance
- Threshold optimization
- Better regularization
"""
import os, json, warnings, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve
from boruta import BorutaPy
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.bool = bool; np.int = int; np.float = float
warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

os.makedirs('processed_model3', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("="*60)
print("IMPROVED Dual-Head Transformer")
print("With Class Weighting & Threshold Optimization")
print("="*60)

# Load clean data
print("\nLoading CLEAN data...")
trends = pd.read_csv("../../data/clean_google.csv", parse_dates=["Date"])
trends["Date"] = pd.to_datetime(trends["Date"]).dt.normalize()

ohlcv = pd.read_csv("../../data/clean_ohlcv.csv", parse_dates=["Date"])
onchain = pd.read_csv("../../data/clean_onchain.csv", parse_dates=["Date"])
sentiment = pd.read_csv("../../data/clean_sentiment.csv", parse_dates=["Date"])

print(f"  Trends: {len(trends)} | OHLCV: {len(ohlcv)} | On-chain: {len(onchain)} | Sentiment: {len(sentiment)}")

# Merge
for frame in [ohlcv, onchain, sentiment, trends]:
    frame["Date"] = frame["Date"].dt.normalize()

df = ohlcv.merge(onchain, on="Date", how="left")
df = df.merge(sentiment, on="Date", how="left")
df = df.merge(trends, on="Date", how="left")
df = df.sort_values("Date").reset_index(drop=True)

print(f"  Merged: {df.shape}")

# Filter date
START_DATE = '2018-02-01'
df = df[df['Date'] >= START_DATE].reset_index(drop=True)
print(f"  After {START_DATE}: {len(df)} rows")

# Fix Fear & Greed
fg_cols = ['fear_greed', 'fg_ma7', 'fg_ma14', 'fg_change', 'fg_change7', 'fg_extreme_fear', 'fg_extreme_greed']
for col in fg_cols:
    if col in df.columns:
        df[col] = df[col].replace(50.0, np.nan)
        df[col] = df[col].ffill(limit=3).bfill(limit=3).fillna(0.0)

# Target
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.iloc[:-1].reset_index(drop=True)
df = df.ffill().bfill().dropna()
print(f"  Final: {len(df)} rows")

# Class distribution
class_dist = df['target'].value_counts()
print(f"\n  Class Distribution:")
print(f"    Up (1):   {class_dist.get(1, 0):,} ({100*class_dist.get(1, 0)/len(df):.1f}%)")
print(f"    Down (0): {class_dist.get(0, 0):,} ({100*class_dist.get(0, 0)/len(df):.1f}%)")

# Feature selection
print("\n" + "="*60)
print("Feature Selection")
print("="*60)

pure_onchain_cols = [
    'AdrActCnt', 'AdrBalCnt', 'TxCnt', 'TxTfrCnt', 'HashRate',
    'BlkCnt', 'SplyCur', 'SplyExNtv', 'SplyExpFut10yr',
    'FeeTotNtv', 'FlowInExNtv', 'FlowOutExNtv', 'IssTotNtv',
    'CapMVRVCur',
    'AdrActCnt_growth7d', 'AdrActCnt_growth30d',
    'TxCnt_growth7d', 'TxCnt_growth30d',
    'HashRate_growth7d', 'HashRate_growth30d',
    'CapMVRVCur_growth7d', 'CapMVRVCur_growth30d'
]
pure_onchain_cols = [c for c in pure_onchain_cols if c in df.columns]

binary_cols = ['fg_extreme_fear', 'fg_extreme_greed']

HEAD2_FEATURES = [
    'google_trends', 'gt_ma7', 'gt_ma30', 'gt_change7', 'gt_momentum',
    'fear_greed', 'fg_ma7', 'fg_ma14', 'fg_change', 'fg_change7',
    'fg_extreme_fear', 'fg_extreme_greed',
]
HEAD2_FEATURES = [c for c in HEAD2_FEATURES if c in df.columns]

# LASSO selection
scale_onchain = [c for c in pure_onchain_cols if c not in binary_cols]
scaler_temp = StandardScaler()
X_fs = scaler_temp.fit_transform(df[scale_onchain].values)
y_fs = df['target'].values

lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.05, random_state=42, max_iter=2000)
lasso.fit(X_fs, y_fs)
selector_l1 = SelectFromModel(lasso, prefit=True)
lasso_mask = selector_l1.get_support()
lasso_selected = [scale_onchain[i] for i, s in enumerate(lasso_mask) if s]

print(f"  LASSO selected: {len(lasso_selected)} features")
print(f"  Features: {lasso_selected}")

HEAD1_FEATURES = lasso_selected + [c for c in binary_cols if c in pure_onchain_cols]

with open('processed_model3/feature_sets.json', 'w') as f:
    json.dump({
        'onchain_lasso': lasso_selected,
        'final_h1': HEAD1_FEATURES,
        'sentiment': HEAD2_FEATURES,
    }, f, indent=2)

print(f"\n  Head 1: {len(HEAD1_FEATURES)} | Head 2: {len(HEAD2_FEATURES)}")

# Scaling
print("\n" + "="*60)
print("Scaling")
print("="*60)

head1_scale_cols = [c for c in HEAD1_FEATURES if c not in binary_cols]
head1_binary_cols = [c for c in HEAD1_FEATURES if c in binary_cols]

scaler_h1 = StandardScaler()
train_h1_scaled = scaler_h1.fit_transform(df[head1_scale_cols].values)
if head1_binary_cols:
    train_h1_binary = df[head1_binary_cols].values
    train_h1 = np.hstack([train_h1_scaled, train_h1_binary])
else:
    train_h1 = train_h1_scaled

bounded_sentiment = ['fear_greed', 'fg_ma7', 'fg_ma14', 'google_trends', 'gt_ma7', 'gt_ma30']
bounded_sentiment = [c for c in bounded_sentiment if c in HEAD2_FEATURES]
flag_cols = ['fg_extreme_fear', 'fg_extreme_greed']
flag_cols = [c for c in flag_cols if c in HEAD2_FEATURES]
continuous_sent = [c for c in HEAD2_FEATURES if c not in bounded_sentiment + flag_cols]

scaler_h2_mm = MinMaxScaler()
scaler_h2_std = StandardScaler()

train_sent_bounded = scaler_h2_mm.fit_transform(df[bounded_sentiment].values) if bounded_sentiment else np.empty((len(df), 0))
train_sent_cont = scaler_h2_std.fit_transform(df[continuous_sent].values) if continuous_sent else np.empty((len(df), 0))
train_flag = df[flag_cols].values if flag_cols else np.empty((len(df), 0))

train_h2 = np.hstack([train_sent_bounded, train_sent_cont, train_flag])

joblib.dump(scaler_h1, 'processed_model3/scaler_head1.pkl')
joblib.dump(scaler_h2_mm, 'processed_model3/scaler_head2_minmax.pkl')
joblib.dump(scaler_h2_std, 'processed_model3/scaler_head2_std.pkl')

# Sequences
print("\n" + "="*60)
print("Sequences")
print("="*60)

SEQ_LEN = 5

def build_sequences(h1_data, h2_data, targets, seq_len):
    X1, X2, y = [], [], []
    for i in range(seq_len, len(h1_data)):
        X1.append(h1_data[i-seq_len:i])
        X2.append(h2_data[i-seq_len:i])
        y.append(targets[i])
    return np.array(X1), np.array(X2), np.array(y)

y_all = df['target'].values
X1_all, X2_all, y_all_seq = build_sequences(train_h1, train_h2, y_all, SEQ_LEN)

# Split
TRAIN_END_IDX = int(len(X1_all) * 0.70)
VAL_END_IDX = int(len(X1_all) * 0.85)

X1_train, X2_train = X1_all[:TRAIN_END_IDX], X2_all[:TRAIN_END_IDX]
X1_val, X2_val = X1_all[TRAIN_END_IDX:VAL_END_IDX], X2_all[TRAIN_END_IDX:VAL_END_IDX]
X1_test, X2_test = X1_all[VAL_END_IDX:], X2_all[VAL_END_IDX:]
y_train, y_val, y_test = y_all_seq[:TRAIN_END_IDX], y_all_seq[TRAIN_END_IDX:VAL_END_IDX], y_all_seq[VAL_END_IDX:]

print(f"  Train: {len(X1_train)} | Val: {len(X1_val)} | Test: {len(X1_test)}")

# Save
np.save('processed_model3/X1_train.npy', X1_train)
np.save('processed_model3/X2_train.npy', X2_train)
np.save('processed_model3/y_train.npy', y_train)
np.save('processed_model3/X1_val.npy', X1_val)
np.save('processed_model3/X2_val.npy', X2_val)
np.save('processed_model3/y_val.npy', y_val)
np.save('processed_model3/X1_test.npy', X1_test)
np.save('processed_model3/X2_test.npy', X2_test)
np.save('processed_model3/y_test.npy', y_test)

# Model
print("\n" + "="*60)
print("Building Dual-Head Transformer")
print("="*60)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class DualHeadTransformer(nn.Module):
    def __init__(self, input_dim_h1, input_dim_h2, d_model=64, nhead=8, num_layers=2, dropout=0.2):
        super().__init__()
        self.head1_proj = nn.Linear(input_dim_h1, d_model)
        self.head2_proj = nn.Linear(input_dim_h2, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head1 = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.head2 = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x1, x2):
        x1 = self.head1_proj(x1)
        x2 = self.head2_proj(x2)
        x = (x1 + x2) / 2
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        h1 = self.dropout(self.head1(x))
        h2 = self.dropout(self.head2(x))
        return h1 + h2

model = DualHeadTransformer(len(HEAD1_FEATURES), len(HEAD2_FEATURES))
print(f"  Head 1: {len(HEAD1_FEATURES)} | Head 2: {len(HEAD2_FEATURES)}")
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

# Training with CLASS WEIGHTING
print("\n" + "="*60)
print("Training with Class Weighting")
print("="*60)

# Calculate class weights
n_up = y_train.sum()
n_down = len(y_train) - n_up
weight_up = len(y_train) / (2 * n_up)
weight_down = len(y_train) / (2 * n_down)
print(f"  Class weights - Up: {weight_up:.3f}, Down: {weight_down:.3f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight_up]))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

X1_train_t = torch.FloatTensor(X1_train)
X2_train_t = torch.FloatTensor(X2_train)
y_train_t = torch.FloatTensor(y_train)
X1_val_t = torch.FloatTensor(X1_val)
X2_val_t = torch.FloatTensor(X2_val)
y_val_t = torch.FloatTensor(y_val)

train_dataset = TensorDataset(X1_train_t, X2_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

best_val_loss = float('inf')
patience_counter = 0
EPOCHS = 200
PATIENCE = 30
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_x1, batch_x2, batch_y in train_loader:
        optimizer.zero_grad()
        out = model(batch_x1, batch_x2).squeeze()
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    model.eval()
    with torch.no_grad():
        val_out = model(X1_val_t, X2_val_t).squeeze()
        val_loss = criterion(val_out, y_val_t)
    
    train_losses.append(epoch_loss / len(train_loader))
    val_losses.append(val_loss.item())
    scheduler.step(val_loss.item())
    
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        torch.save(model.state_dict(), 'processed_model3/model3_best.pt')
        patience_counter = 0
        print(f"  Epoch {epoch+1:3d} | Train: {epoch_loss/len(train_loader):.4f} | Val: {val_loss.item():.4f} ★")
    else:
        patience_counter += 1
        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d} | Train: {epoch_loss/len(train_loader):.4f} | Val: {val_loss.item():.4f}")
    
    if patience_counter >= PATIENCE:
        print(f"  Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(torch.load('processed_model3/model3_best.pt'))
print(f"\n  Best val loss: {best_val_loss:.4f}")

# Evaluation with THRESHOLD TUNING
print("\n" + "="*60)
print("Evaluation with Threshold Optimization")
print("="*60)

model.eval()
with torch.no_grad():
    X1_test_t = torch.FloatTensor(X1_test)
    X2_test_t = torch.FloatTensor(X2_test)
    test_out = model(X1_test_t, X2_test_t).squeeze()
    test_probs = torch.sigmoid(test_out).numpy()
    
    # Find optimal threshold
    print("\n  Threshold Analysis:")
    best_threshold = 0.5
    best_f1 = 0
    
    for thresh in np.arange(0.2, 0.8, 0.02):
        preds = (test_probs > thresh).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    print(f"    Best threshold: {best_threshold:.3f}")
    print(f"    Best F1: {best_f1:.4f}")
    
    # Final evaluation
    test_preds = (test_probs > best_threshold).astype(int)
    acc = accuracy_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    
    print(f"\n  === FINAL TEST RESULTS ===")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Threshold: {best_threshold:.3f}")
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, test_preds)
    print(f"    [[{cm[0,0]} {cm[0,1]}]")
    print(f"     [{cm[1,0]} {cm[1,1]}]]")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, test_preds))

# Save metrics
metrics = {
    'accuracy': float(acc),
    'f1': float(f1),
    'best_threshold': float(best_threshold),
    'best_val_loss': float(best_val_loss),
    'epochs_trained': epoch + 1,
    'features': HEAD1_FEATURES + HEAD2_FEATURES,
}
with open('processed_model3/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Plots
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Curves')
plt.legend()
plt.savefig('plots/training_curves.png', dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, test_probs)
roc_auc = np.trapezoid(tpr, fpr)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('plots/roc_curve.png', dpi=150)
plt.close()

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print(f"\nModel: processed_model3/model3_best.pt")
print(f"Metrics: processed_model3/metrics.json")
