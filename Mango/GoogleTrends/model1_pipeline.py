"""
MODEL 1 — Dual-Head Transformer: Baseline Sentiment (2018-2026)
Fear & Greed + Google Trends Only
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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from boruta import BorutaPy
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.bool = bool; np.int = int; np.float = float
warnings.filterwarnings('ignore')

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)
os.makedirs('processed_model1', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# PHASE 2 — Check Google Trends File
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("PHASE 2 — Check Google Trends")
print("=" * 60)

if not os.path.exists('google_trends_bitcoin.csv'):
    raise FileNotFoundError("google_trends_bitcoin.csv is missing.")

trends = pd.read_csv('google_trends_bitcoin.csv', parse_dates=['Date'])
trends['Date'] = trends['Date'].dt.normalize()

print(f"  Google Trends rows: {len(trends)}")
print(f"  Date range: {trends['Date'].min().date()} to {trends['Date'].max().date()}")

feb_2018 = trends[trends['Date'] >= '2018-02-01']
print(f"  Rows from 2018-02-01 onwards: {len(feb_2018)}")

# ═══════════════════════════════════════════════════════════════
# PHASE 3 — Load All Files
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 3 — Loading All Files")
print("=" * 60)

ohlcv     = pd.read_csv('ohlcv_2010_to_now.csv', parse_dates=['Date'])
onchain   = pd.read_csv('onchain_and_technicals_2010_to_now-2.csv', parse_dates=['Date'])
sentiment = pd.read_csv('sentiment_2010_to_now-3.csv', parse_dates=['Date'])

for frame in [ohlcv, onchain, sentiment, trends]:
    frame['Date'] = frame['Date'].dt.normalize()

df = ohlcv.merge(onchain, on='Date', how='left')
df = df.merge(sentiment, on='Date', how='left')
df = df.merge(trends, on='Date', how='left')
df = df.sort_values('Date').reset_index(drop=True)

print(f"  Merged shape: {df.shape}")
print(f"  Full date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# ═══════════════════════════════════════════════════════════════
# PHASE 4 — Drop Leaky & Redundant Columns
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 4 — Dropping Leaky & Redundant Columns")
print("=" * 60)

drop_cols = [
    'ROI1yr', 'CapMrktCurUSD', 'CapMrktEstUSD',
    'ReferenceRate', 'ReferenceRateUSD', 'ReferenceRateEUR', 'ReferenceRateETH',
    'AssetCompletionTime', 'AssetEODCompletionTime',
    'SplyExUSD',
    'SMA_7', 'SMA_14', 'SMA_21', 'SMA_50', 'SMA_100', 'SMA_200',
    'EMA_12', 'EMA_26', 'EMA_50',
    'BB_upper', 'BB_lower',
]

drop_cols = [c for c in drop_cols if c in df.columns]
df.drop(columns=drop_cols, inplace=True)
print(f"  Shape after dropping: {df.shape}")

# ═══════════════════════════════════════════════════════════════
# PHASE 5 — Set Date Range (2018-02-01)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 5 — Set Date Range")
print("=" * 60)

START_DATE = '2018-02-01'
df = df[df['Date'] >= START_DATE].reset_index(drop=True)

print(f"  Rows after date filter: {len(df)}")
print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# ═══════════════════════════════════════════════════════════════
# PHASE 6 — Fix Fear & Greed Column
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 6 — Fix Fear & Greed Column")
print("=" * 60)

fg_cols = ['fear_greed', 'fg_ma7', 'fg_ma14', 'fg_change', 'fg_change7',
           'fg_extreme_fear', 'fg_extreme_greed']

for col in fg_cols:
    if col in df.columns:
        df[col] = df[col].replace(50.0, np.nan)
        df[col] = df[col].ffill(limit=3)
        df[col] = df[col].bfill(limit=3)
        df[col] = df[col].fillna(0.0)

print("  Fear & Greed columns cleaned.")
print(f"  Remaining NaNs in fg columns: {df[fg_cols].isnull().sum().sum()}")

# ═══════════════════════════════════════════════════════════════
# PHASE 7 — Define Target Variable
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 7 — Define Target Variable")
print("=" * 60)

df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.iloc[:-1].reset_index(drop=True)

print("  Target distribution:")
print(df['target'].value_counts(normalize=True).round(3).to_string())

# ═══════════════════════════════════════════════════════════════
# PHASE 8 — Handle Remaining Missing Values
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 8 — Handle Remaining Missing Values")
print("=" * 60)

df = df.ffill()
df = df.bfill()

null_count = df.isnull().sum().sum()
assert null_count == 0, f"Still {null_count} nulls — investigate before continuing"
print("  Zero nulls confirmed. Safe to proceed.")

# ═══════════════════════════════════════════════════════════════
# PHASE 9 — Log-Transform Skewed Features
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 9 — Log Transforms")
print("=" * 60)

log_pos_cols = [
    'AdrActCnt', 'AdrBalCnt', 'TxCnt', 'TxTfrCnt', 'HashRate',
    'BlkCnt', 'SplyCur', 'SplyExNtv', 'SplyExpFut10yr',
    'FeeTotNtv', 'FlowInExNtv', 'FlowOutExNtv', 'IssTotNtv',
    'Volume', 'volume_reported_spot_usd_1d', 'vol_sma7', 'vol_sma30',
    'TR', 'ATR', 'CapMVRVCur',
]
log_pos_cols = [c for c in log_pos_cols if c in df.columns]
for col in log_pos_cols:
    df[col] = np.log1p(df[col].clip(lower=0))

signed_log_cols = [
    'OBV', 'OBV_sma7',
    'TxCnt_growth7d', 'TxCnt_growth30d',
    'HashRate_growth7d', 'HashRate_growth30d',
    'AdrActCnt_growth7d', 'AdrActCnt_growth30d',
    'CapMVRVCur_growth7d', 'CapMVRVCur_growth30d',
    'return_30d', 'ROI30d',
    'gt_change7', 'gt_momentum',
]
signed_log_cols = [c for c in signed_log_cols if c in df.columns]
for col in signed_log_cols:
    df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

print("  Log transforms applied.")

# ═══════════════════════════════════════════════════════════════
# PHASE 10 — Chronological Train / Val / Test Split
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 10 — Chronological Split")
print("=" * 60)

TRAIN_END = '2022-12-31'
VAL_END   = '2023-12-31'

train = df[df['Date'] <= TRAIN_END].copy().reset_index(drop=True)
val   = df[(df['Date'] > TRAIN_END) & (df['Date'] <= VAL_END)].copy().reset_index(drop=True)
test  = df[df['Date'] > VAL_END].copy().reset_index(drop=True)

print(f"  Train: {len(train)} rows ({train['Date'].min().date()} to {train['Date'].max().date()})")
print(f"  Val:   {len(val)} rows ({val['Date'].min().date()} to {val['Date'].max().date()})")
print(f"  Test:  {len(test)} rows ({test['Date'].min().date()} to {test['Date'].max().date()})")

# ═══════════════════════════════════════════════════════════════
# PHASE 11 — Define Feature Sets
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 11 — Feature Set Definitions")
print("=" * 60)

exclude_always = ['Date', 'target', 'Open', 'High', 'Low', 'Close']
binary_cols = ['sma7_sma21_cross', 'sma21_sma50_cross', 'sma50_sma200_cross',
               'fg_extreme_fear', 'fg_extreme_greed']

onchain_technical_cols = [
    'price_sma50_ratio', 'price_sma200_ratio', 'price_sma21_ratio',
    'RSI_14', 'RSI_21', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'ROC_5', 'ROC_10', 'ROC_20', 'Stoch_K', 'Stoch_D',
    'BB_width', 'BB_pct', 'ATR', 'TR',
    'vol_7d', 'vol_14d', 'vol_21d', 'vol_30d', 'vol_60d', 'vol_ratio',
    'return_1d', 'return_3d', 'return_5d', 'return_7d',
    'return_10d', 'return_14d', 'return_21d', 'return_30d',
    'Volume', 'OBV', 'OBV_sma7', 'vol_sma7', 'vol_sma30',
    'volume_reported_spot_usd_1d',
    'AdrActCnt', 'AdrBalCnt', 'TxCnt', 'TxTfrCnt', 'HashRate',
    'BlkCnt', 'SplyCur', 'SplyExNtv', 'SplyExpFut10yr',
    'FeeTotNtv', 'FlowInExNtv', 'FlowOutExNtv', 'IssTotNtv',
    'CapMVRVCur', 'ROI30d',
    'AdrActCnt_growth7d', 'AdrActCnt_growth30d',
    'TxCnt_growth7d', 'TxCnt_growth30d',
    'HashRate_growth7d', 'HashRate_growth30d',
    'CapMVRVCur_growth7d', 'CapMVRVCur_growth30d',
    'sma7_sma21_cross', 'sma21_sma50_cross', 'sma50_sma200_cross',
]
onchain_technical_cols = [c for c in onchain_technical_cols if c in df.columns]

HEAD2_FEATURES = [
    'google_trends', 'gt_ma7', 'gt_ma30', 'gt_change7', 'gt_momentum',
    'fear_greed', 'fg_ma7', 'fg_ma14', 'fg_change', 'fg_change7',
    'fg_extreme_fear', 'fg_extreme_greed',
]
HEAD2_FEATURES = [c for c in HEAD2_FEATURES if c in df.columns]

print(f"  Head 1 candidate features: {len(onchain_technical_cols)}")
print(f"  Head 2 sentiment features: {len(HEAD2_FEATURES)}")

# ═══════════════════════════════════════════════════════════════
# PHASE 12 — Feature Selection on Head 1 (FRESH for Model 1)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 12 — Boruta + LASSO Feature Selection")
print("=" * 60)

scale_onchain = [c for c in onchain_technical_cols if c not in binary_cols]
scaler_temp   = StandardScaler()
X_fs = scaler_temp.fit_transform(train[scale_onchain].values)
y_fs = train['target'].values

rf = RandomForestClassifier(n_jobs=-1, max_depth=7, random_state=42, n_estimators=200)
boruta = BorutaPy(estimator=rf, n_estimators='auto', random_state=42, verbose=0, max_iter=100)
boruta.fit(X_fs, y_fs)
boruta_selected = [scale_onchain[i] for i, s in enumerate(boruta.support_) if s]
boruta_selected += [c for c in binary_cols if c in onchain_technical_cols]
print(f"  Boruta selected {len(boruta_selected)} features")

lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.05, random_state=42, max_iter=2000)
lasso.fit(X_fs, y_fs)
selector_l1 = SelectFromModel(lasso, prefit=True)
lasso_mask = selector_l1.get_support()
lasso_selected = [scale_onchain[i] for i, s in enumerate(lasso_mask) if s]
lasso_selected += [c for c in binary_cols if c in onchain_technical_cols]
print(f"  LASSO selected {len(lasso_selected)} features")

HEAD1_FEATURES = boruta_selected

with open('processed_model1/feature_sets.json', 'w') as f:
    json.dump({
        'onchain_boruta': boruta_selected,
        'onchain_lasso':  lasso_selected,
        'sentiment':      HEAD2_FEATURES,
    }, f, indent=2)

print(f"\n  ✅ Head 1 final features: {len(HEAD1_FEATURES)}")
print(f"  ✅ Head 2 final features: {len(HEAD2_FEATURES)}")

# ═══════════════════════════════════════════════════════════════
# PHASE 13 — Scale Features (Fit on Train Only)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 13 — Feature Scaling")
print("=" * 60)

head1_scale_cols  = [c for c in HEAD1_FEATURES if c not in binary_cols]
head1_binary_cols = [c for c in HEAD1_FEATURES if c in binary_cols]

scaler_h1 = StandardScaler()
if head1_scale_cols:
    train_h1_scaled = scaler_h1.fit_transform(train[head1_scale_cols].values)
    val_h1_scaled   = scaler_h1.transform(val[head1_scale_cols].values)
    test_h1_scaled  = scaler_h1.transform(test[head1_scale_cols].values)
else:
    train_h1_scaled = np.zeros((len(train), 0))
    val_h1_scaled   = np.zeros((len(val), 0))
    test_h1_scaled  = np.zeros((len(test), 0))

train_h1 = np.hstack([train_h1_scaled, train[head1_binary_cols].values]) if head1_binary_cols else train_h1_scaled
val_h1   = np.hstack([val_h1_scaled,   val[head1_binary_cols].values])   if head1_binary_cols else val_h1_scaled
test_h1  = np.hstack([test_h1_scaled,  test[head1_binary_cols].values])  if head1_binary_cols else test_h1_scaled

bounded_sentiment  = ['fear_greed', 'fg_ma7', 'fg_ma14', 'google_trends', 'gt_ma7', 'gt_ma30']
bounded_sentiment  = [c for c in bounded_sentiment if c in HEAD2_FEATURES]
flag_cols          = ['fg_extreme_fear', 'fg_extreme_greed']
flag_cols          = [c for c in flag_cols if c in HEAD2_FEATURES]
continuous_sent    = [c for c in HEAD2_FEATURES if c not in bounded_sentiment + flag_cols]

scaler_h2_mm  = MinMaxScaler()
scaler_h2_std = StandardScaler()

train_sent_bounded = scaler_h2_mm.fit_transform(train[bounded_sentiment].values)
val_sent_bounded   = scaler_h2_mm.transform(val[bounded_sentiment].values)
test_sent_bounded  = scaler_h2_mm.transform(test[bounded_sentiment].values)

if continuous_sent:
    train_sent_cont = scaler_h2_std.fit_transform(train[continuous_sent].values)
    val_sent_cont   = scaler_h2_std.transform(val[continuous_sent].values)
    test_sent_cont  = scaler_h2_std.transform(test[continuous_sent].values)
else:
    train_sent_cont, val_sent_cont, test_sent_cont = np.zeros((len(train),0)), np.zeros((len(val),0)), np.zeros((len(test),0))

train_h2 = np.hstack([train_sent_bounded, train_sent_cont, train[flag_cols].values])
val_h2   = np.hstack([val_sent_bounded,   val_sent_cont,   val[flag_cols].values])
test_h2  = np.hstack([test_sent_bounded,  test_sent_cont,  test[flag_cols].values])

joblib.dump(scaler_h1,     'processed_model1/scaler_head1.pkl')
joblib.dump(scaler_h2_mm,  'processed_model1/scaler_head2_minmax.pkl')
joblib.dump(scaler_h2_std, 'processed_model1/scaler_head2_std.pkl')

print(f"  Head 1 array shape: {train_h1.shape[1]} features")
print(f"  Head 2 array shape: {train_h2.shape[1]} features")

# ═══════════════════════════════════════════════════════════════
# PHASE 14 — Build Sliding Window Sequences
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 14 — Sliding Window Sequences")
print("=" * 60)

SEQ_LEN = 5

def build_sequences(h1_data, h2_data, targets, seq_len):
    X1, X2, y = [], [], []
    for i in range(seq_len, len(h1_data)):
        X1.append(h1_data[i - seq_len:i])
        X2.append(h2_data[i - seq_len:i])
        y.append(targets[i])
    return np.array(X1), np.array(X2), np.array(y)

y_train = train['target'].values
y_val   = val['target'].values
y_test  = test['target'].values

X1_train, X2_train, y_train_seq = build_sequences(train_h1, train_h2, y_train, SEQ_LEN)
X1_val,   X2_val,   y_val_seq   = build_sequences(val_h1,   val_h2,   y_val,   SEQ_LEN)
X1_test,  X2_test,  y_test_seq  = build_sequences(test_h1,  test_h2,  y_test,  SEQ_LEN)

print(f"  X1_train (on-chain):  {X1_train.shape}")
print(f"  X2_train (sentiment): {X2_train.shape}")
print(f"  y_train:              {y_train_seq.shape}")

np.save('processed_model1/X1_train.npy', X1_train)
np.save('processed_model1/X2_train.npy', X2_train)
np.save('processed_model1/y_train.npy',  y_train_seq)
np.save('processed_model1/X1_val.npy',   X1_val)
np.save('processed_model1/X2_val.npy',   X2_val)
np.save('processed_model1/y_val.npy',    y_val_seq)
np.save('processed_model1/X1_test.npy',  X1_test)
np.save('processed_model1/X2_test.npy',  X2_test)
np.save('processed_model1/y_test.npy',   y_test_seq)
print("  All arrays saved to processed_model1/")

# ═══════════════════════════════════════════════════════════════
# PHASE 15 — Build the Dual-Head Transformer Model
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 15 — Building Dual-Head Transformer")
print("=" * 60)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class DualHeadTransformer(nn.Module):
    def __init__(self, n_onchain, n_sentiment,
                 d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.3):
        super().__init__()
        self.proj_onchain = nn.Linear(n_onchain, d_model)
        self.pos_enc_h1   = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer_h1  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer_h1 = nn.TransformerEncoder(encoder_layer_h1, num_layers=num_layers)

        self.proj_sentiment = nn.Linear(n_sentiment, d_model)
        self.pos_enc_h2     = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer_h2    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer_h2 = nn.TransformerEncoder(encoder_layer_h2, num_layers=num_layers)

        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x_onchain, x_sentiment):
        h1 = self.proj_onchain(x_onchain)
        h1 = self.pos_enc_h1(h1)
        h1 = self.transformer_h1(h1)
        h1 = h1[:, -1, :]

        h2 = self.proj_sentiment(x_sentiment)
        h2 = self.pos_enc_h2(h2)
        h2 = self.transformer_h2(h2)
        h2 = h2[:, -1, :]

        fused = torch.cat([h1, h2], dim=-1)
        return self.fusion(fused).squeeze(-1)

N_ONCHAIN   = X1_train.shape[2]
N_SENTIMENT = X2_train.shape[2]

model = DualHeadTransformer(
    n_onchain=N_ONCHAIN, n_sentiment=N_SENTIMENT,
    d_model=64, nhead=4, num_layers=2,
    dim_feedforward=128, dropout=0.3)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total trainable parameters: {total_params:,}")

# ═══════════════════════════════════════════════════════════════
# PHASE 16 — Train the Model
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 16 — Training")
print("=" * 60)

pos_rate = y_train_seq.mean()
neg_rate = 1 - pos_rate
print(f"  Class balance — Up: {pos_rate:.3f} | Down: {neg_rate:.3f}")
pos_weight = torch.tensor([neg_rate / pos_rate], dtype=torch.float32)

def make_loader(X1, X2, y, batch_size=32, shuffle=False):
    dataset = TensorDataset(
        torch.tensor(X1, dtype=torch.float32),
        torch.tensor(X2, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X1_train, X2_train, y_train_seq, batch_size=32, shuffle=False)
val_loader   = make_loader(X1_val,   X2_val,   y_val_seq,   batch_size=32, shuffle=False)

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = model.to(device)
criterion = nn.BCELoss(weight=pos_weight.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=10, factor=0.5)

EPOCHS   = 200
PATIENCE = 30
best_val_loss    = float('inf')
patience_ctr     = 0
best_model_state = None
train_losses, val_losses = [], []

print(f"  Device: {device}")
print(f"  Training for max {EPOCHS} epochs (patience={PATIENCE})...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0
    for X1b, X2b, yb in train_loader:
        X1b, X2b, yb = X1b.to(device), X2b.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(X1b, X2b)
        loss  = criterion(preds, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    avg_train_loss = epoch_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X1b, X2b, yb in val_loader:
            X1b, X2b, yb = X1b.to(device), X2b.to(device), yb.to(device)
            preds    = model(X1b, X2b)
            val_loss += criterion(preds, yb).item()
    avg_val_loss = val_loss / len(val_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    scheduler.step(avg_val_loss)

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss    = avg_val_loss
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_ctr     = 0
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"\n  ⏹ Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
            break

model.load_state_dict(best_model_state)
torch.save(best_model_state, 'processed_model1/model1_best.pt')
print("  Best model saved.")

# ═══════════════════════════════════════════════════════════════
# PHASE 17 — Evaluate on Test Set
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 17 — Test Set Evaluation")
print("=" * 60)

model.eval()
all_preds, all_probs = [], []
test_loader = make_loader(X1_test, X2_test, y_test_seq, batch_size=32, shuffle=False)

with torch.no_grad():
    for X1b, X2b, yb in test_loader:
        X1b, X2b = X1b.to(device), X2b.to(device)
        probs = model(X1b, X2b).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_probs.extend(probs)
        all_preds.extend(preds)

y_true = y_test_seq[:len(all_preds)]

print(f"\n  === MODEL 1 TEST RESULTS ===")
print(f"  Accuracy: {accuracy_score(y_true, all_preds):.4f}")
print(f"  F1 Score: {f1_score(y_true, all_preds):.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_true, all_preds, target_names=['Down', 'Up']))
print(f"  Confusion Matrix:")
print(confusion_matrix(y_true, all_preds))

# ═══════════════════════════════════════════════════════════════
# PHASE 18 — Save Everything for Comparison
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 18 — Saving All Outputs")
print("=" * 60)

np.save('processed_model1/model1_probs_test.npy',  np.array(all_probs))
np.save('processed_model1/model1_preds_test.npy',  np.array(all_preds))
np.save('processed_model1/model1_ytrue_test.npy',  y_true)

with open('processed_model1/model1_losses.json', 'w') as f:
    json.dump({'train': train_losses, 'val': val_losses}, f)

config = {
    'model': 'DualHeadTransformer',
    'name': 'Model 1 — Baseline Sentiment (2018-2026)',
    'start_date': '2018-02-01',
    'train_end': TRAIN_END,
    'val_end': VAL_END,
    'seq_len': SEQ_LEN,
    'd_model': 64,
    'nhead': 4,
    'num_layers': 2,
    'dropout': 0.3,
    'n_head1_features': int(N_ONCHAIN),
    'n_head2_features': int(N_SENTIMENT),
    'sentiment_sources': ['Google Trends (2018+)', 'Fear & Greed (2018+)'],
    'feature_selection': 'Boruta (primary), LASSO (secondary)',
    'note': 'No availability masks — all sentiment sources real from start date',
    'accuracy': float(accuracy_score(y_true, all_preds)),
    'f1': float(f1_score(y_true, all_preds)),
}
with open('processed_model1/model1_config.json', 'w') as f:
    json.dump(config, f, indent=2)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train_losses, label='Train Loss', color='#2196F3')
ax.plot(val_losses, label='Val Loss', color='#FF5722')
ax.set_xlabel('Epoch')
ax.set_ylabel('BCE Loss')
ax.set_title('Model 1 (Baseline) — Training Curves')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('plots/model1_training_curves.png', dpi=150)
plt.close(fig)

print("\n  ✅ All Model 1 outputs saved to processed_model1/")
print("  ✅ Training curves saved to plots/model1_training_curves.png")
print("\n  🎉 MODEL 1 PIPELINE COMPLETE!")
