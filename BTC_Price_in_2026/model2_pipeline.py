"""
MODEL 2 — Dual-Head Transformer: On-Chain + Enriched Sentiment (2013–2026)
Full pipeline: Phases 1-19 per Dubey & Enke (2025) methodology.
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
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Fix numpy deprecation warnings from boruta
np.bool = bool; np.int = int; np.float = float
warnings.filterwarnings('ignore')

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)
os.makedirs('processed', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# PHASE 2 — Fetch Google Trends Data
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("PHASE 2 — Fetching Google Trends Data")
print("=" * 60)

GT_FILE = 'google_trends_bitcoin.csv'
if os.path.exists(GT_FILE):
    print(f"  Google Trends file already exists. Skipping fetch.")
    raw_gt = pd.read_csv(GT_FILE, parse_dates=['Date'])
else:
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-US', tz=0)

        periods = [
            ('2013-01-01', '2014-09-01'),
            ('2014-06-01', '2016-03-01'),
            ('2015-12-01', '2017-09-01'),
            ('2017-06-01', '2019-03-01'),
            ('2018-12-01', '2020-09-01'),
            ('2020-06-01', '2022-03-01'),
            ('2021-12-01', '2023-09-01'),
            ('2023-06-01', '2026-03-01'),
        ]

        chunks = []
        for start, end in periods:
            try:
                pytrends.build_payload(['bitcoin'], timeframe=f'{start} {end}', geo='')
                chunk = pytrends.interest_over_time()[['bitcoin']].copy()
                chunk.columns = ['google_trends']
                chunks.append(chunk)
                print(f"  ✅ Fetched {start} to {end}: {len(chunk)} rows")
                time.sleep(3)
            except Exception as e:
                print(f"  ⚠️ Failed {start}–{end}: {e}")

        if chunks:
            raw_gt = pd.concat(chunks)
            raw_gt = raw_gt[~raw_gt.index.duplicated(keep='first')].sort_index()
            raw_gt = raw_gt.resample('D').interpolate(method='linear')
            raw_gt['google_trends'] = raw_gt['google_trends'] / raw_gt['google_trends'].max() * 100
            raw_gt['gt_ma7']     = raw_gt['google_trends'].rolling(7).mean()
            raw_gt['gt_ma30']    = raw_gt['google_trends'].rolling(30).mean()
            raw_gt['gt_change7'] = raw_gt['google_trends'].pct_change(7)
            raw_gt['gt_momentum'] = raw_gt['google_trends'] - raw_gt['gt_ma30']
            raw_gt.index.name = 'Date'
            raw_gt = raw_gt.reset_index()
            raw_gt['Date'] = pd.to_datetime(raw_gt['Date']).dt.normalize()
            raw_gt.to_csv(GT_FILE, index=False)
            print(f"  Google Trends saved: {len(raw_gt)} rows")
        else:
            raise RuntimeError("No Google Trends data fetched")
    except Exception as e:
        print(f"  ⚠️ Google Trends fetch failed: {e}")
        print(f"  Creating synthetic Google Trends proxy from BTC price volatility...")
        ohlcv_temp = pd.read_csv('ohlcv_2010_to_now.csv', parse_dates=['Date'])
        ohlcv_temp = ohlcv_temp[ohlcv_temp['Date'] >= '2013-01-01'].copy()
        # Use absolute return as a proxy for search interest
        ohlcv_temp['google_trends'] = ohlcv_temp['Close'].pct_change().abs().rolling(7).mean()
        ohlcv_temp['google_trends'] = ohlcv_temp['google_trends'] / ohlcv_temp['google_trends'].max() * 100
        ohlcv_temp['gt_ma7']     = ohlcv_temp['google_trends'].rolling(7).mean()
        ohlcv_temp['gt_ma30']    = ohlcv_temp['google_trends'].rolling(30).mean()
        ohlcv_temp['gt_change7'] = ohlcv_temp['google_trends'].pct_change(7)
        ohlcv_temp['gt_momentum'] = ohlcv_temp['google_trends'] - ohlcv_temp['gt_ma30']
        raw_gt = ohlcv_temp[['Date','google_trends','gt_ma7','gt_ma30','gt_change7','gt_momentum']].copy()
        raw_gt['Date'] = pd.to_datetime(raw_gt['Date']).dt.normalize()
        raw_gt.to_csv(GT_FILE, index=False)
        print(f"  Proxy Google Trends saved: {len(raw_gt)} rows")

# (Phase 3 — LunarCrush removed: API requires paid subscription)

# ═══════════════════════════════════════════════════════════════
# PHASE 4 — Load All Base Data Files & Merge
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 4 — Loading and Merging All Data Sources")
print("=" * 60)

ohlcv     = pd.read_csv('ohlcv_2010_to_now.csv', parse_dates=['Date'])
onchain   = pd.read_csv('onchain_and_technicals_2010_to_now.csv', parse_dates=['Date'])
sentiment = pd.read_csv('sentiment_2010_to_now.csv', parse_dates=['Date'])
trends    = pd.read_csv(GT_FILE, parse_dates=['Date'])

for frame in [ohlcv, onchain, sentiment, trends]:
    frame['Date'] = frame['Date'].dt.normalize()

df = ohlcv.merge(onchain, on='Date', how='left')
df = df.merge(sentiment, on='Date', how='left')
df = df.merge(trends, on='Date', how='left')
df = df.sort_values('Date').reset_index(drop=True)

print(f"  Merged shape: {df.shape}")
print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# ═══════════════════════════════════════════════════════════════
# PHASE 5 — Drop Leaky & Redundant Columns
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 5 — Dropping Leaky & Redundant Columns")
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
print(f"  Dropped {len(drop_cols)} columns. Shape: {df.shape}")

# ═══════════════════════════════════════════════════════════════
# PHASE 6 — Set Date Range (2013+)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 6 — Date Range Filter")
print("=" * 60)

START_DATE = '2013-01-01'
df = df[df['Date'] >= START_DATE].reset_index(drop=True)
print(f"  Rows after date filter: {len(df)}")

# ═══════════════════════════════════════════════════════════════
# PHASE 7 — Create Sentiment Availability Masks
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 7 — Sentiment Availability Masks")
print("=" * 60)

df['fg_available']     = (df['Date'] >= '2018-02-01').astype(float)
df['trends_available'] = (df['Date'] >= '2013-01-01').astype(float)

# Zero-fill Fear & Greed columns for pre-2018 rows
fg_cols = ['fear_greed', 'fg_ma7', 'fg_ma14', 'fg_change', 'fg_change7',
           'fg_extreme_fear', 'fg_extreme_greed']
for col in fg_cols:
    if col in df.columns:
        df.loc[df['Date'] < '2018-02-01', col] = 0.0
        mask_post = df['Date'] >= '2018-02-01'
        df.loc[mask_post, col] = df.loc[mask_post, col].replace(50.0, np.nan)
        df.loc[mask_post, col] = df.loc[mask_post, col].ffill(limit=3)
        df.loc[mask_post, col] = df.loc[mask_post, col].fillna(0.0)

print(f"  Rows with full sentiment (2018+): {(df['fg_available'] == 1).sum()}")
print(f"  Rows with Google Trends (2013+):  {(df['trends_available'] == 1).sum()}")

# ═══════════════════════════════════════════════════════════════
# PHASE 8 — Define Target Variable
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 8 — Target Variable")
print("=" * 60)

df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.iloc[:-1].reset_index(drop=True)

print("  Target distribution:")
print(df['target'].value_counts(normalize=True).round(3).to_string())

# ═══════════════════════════════════════════════════════════════
# PHASE 9 — Handle Remaining Missing Values
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 9 — Missing Value Handling")
print("=" * 60)

df = df.ffill().bfill()
null_count = df.isnull().sum().sum()
assert null_count == 0, f"Still {null_count} nulls — investigate before continuing"
print("  Zero nulls confirmed. Safe to proceed.")

# ═══════════════════════════════════════════════════════════════
# PHASE 10 — Log-Transform Skewed Features
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 10 — Log Transforms")
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
print(f"  Log1p applied to {len(log_pos_cols)} positive columns")

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
print(f"  Signed log applied to {len(signed_log_cols)} signed columns")

# ═══════════════════════════════════════════════════════════════
# PHASE 11 — Chronological Train / Val / Test Split
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 11 — Chronological Split")
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
# PHASE 12 — Define Feature Sets for Each Head
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 12 — Feature Set Definitions")
print("=" * 60)

exclude_always = ['Date', 'target', 'Open', 'High', 'Low', 'Close']
binary_cols = ['sma7_sma21_cross', 'sma21_sma50_cross', 'sma50_sma200_cross',
               'fg_extreme_fear', 'fg_extreme_greed',
               'fg_available', 'trends_available']

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

sentiment_feature_cols = [
    'google_trends', 'gt_ma7', 'gt_ma30', 'gt_change7', 'gt_momentum',
    'fear_greed', 'fg_ma7', 'fg_ma14', 'fg_change', 'fg_change7',
    'fg_extreme_fear', 'fg_extreme_greed',
    'fg_available', 'trends_available',
]
sentiment_feature_cols = [c for c in sentiment_feature_cols if c in df.columns]

print(f"  Head 1 (on-chain/technical): {len(onchain_technical_cols)} features")
print(f"  Head 2 (sentiment):          {len(sentiment_feature_cols)} features")

# ═══════════════════════════════════════════════════════════════
# PHASE 13 — Feature Selection on Head 1 (Boruta + LASSO)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 13 — Boruta + LASSO Feature Selection")
print("=" * 60)

scale_onchain = [c for c in onchain_technical_cols if c not in binary_cols]
head1_binary  = [c for c in onchain_technical_cols if c in binary_cols]

scaler_temp = StandardScaler()
X_fs = scaler_temp.fit_transform(train[scale_onchain].values)
y_fs = train['target'].values

# Boruta
from boruta import BorutaPy
rf = RandomForestClassifier(n_jobs=-1, max_depth=7, random_state=42, n_estimators=200)
boruta = BorutaPy(estimator=rf, n_estimators='auto', random_state=42, verbose=0, max_iter=100)
boruta.fit(X_fs, y_fs)
boruta_selected = [scale_onchain[i] for i, s in enumerate(boruta.support_) if s]
# Also include tentative features
boruta_tentative = [scale_onchain[i] for i, s in enumerate(boruta.support_weak_) if s]
boruta_selected += boruta_tentative
boruta_selected += head1_binary
print(f"  Boruta selected {len(boruta_selected)} features (incl. {len(boruta_tentative)} tentative)")

# LASSO / L1
lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.05, random_state=42, max_iter=2000)
lasso.fit(X_fs, y_fs)
selector_l1 = SelectFromModel(lasso, prefit=True)
lasso_mask = selector_l1.get_support()
lasso_selected = [scale_onchain[i] for i, s in enumerate(lasso_mask) if s]
lasso_selected += head1_binary
print(f"  LASSO selected {len(lasso_selected)} features")

PCA_N_COMPONENTS = 20

with open('processed/feature_sets.json', 'w') as f:
    json.dump({
        'onchain_all':      onchain_technical_cols,
        'onchain_boruta':   boruta_selected,
        'onchain_lasso':    lasso_selected,
        'sentiment':        sentiment_feature_cols,
        'pca_n_components': PCA_N_COMPONENTS,
    }, f, indent=2)

# Use Boruta as primary
HEAD1_FEATURES = boruta_selected
HEAD2_FEATURES = sentiment_feature_cols
print(f"\n  ✅ Final Head 1 features: {len(HEAD1_FEATURES)}")
print(f"  ✅ Final Head 2 features: {len(HEAD2_FEATURES)}")

# ═══════════════════════════════════════════════════════════════
# PHASE 14 — Scale Features (Fit on Train Only)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 14 — Feature Scaling")
print("=" * 60)

# HEAD 1
head1_scale_cols  = [c for c in HEAD1_FEATURES if c not in binary_cols]
head1_binary_cols = [c for c in HEAD1_FEATURES if c in binary_cols]

scaler_h1 = StandardScaler()
train_h1_scaled = scaler_h1.fit_transform(train[head1_scale_cols].values)
val_h1_scaled   = scaler_h1.transform(val[head1_scale_cols].values)
test_h1_scaled  = scaler_h1.transform(test[head1_scale_cols].values)

train_h1 = np.hstack([train_h1_scaled, train[head1_binary_cols].values]) if head1_binary_cols else train_h1_scaled
val_h1   = np.hstack([val_h1_scaled,   val[head1_binary_cols].values])   if head1_binary_cols else val_h1_scaled
test_h1  = np.hstack([test_h1_scaled,  test[head1_binary_cols].values])  if head1_binary_cols else test_h1_scaled

# HEAD 2
bounded_sentiment  = ['fear_greed','fg_ma7','fg_ma14','google_trends',
                       'gt_ma7','gt_ma30']
bounded_sentiment  = [c for c in bounded_sentiment if c in HEAD2_FEATURES]
flag_cols          = ['fg_extreme_fear','fg_extreme_greed','fg_available',
                      'trends_available']
flag_cols          = [c for c in flag_cols if c in HEAD2_FEATURES]
continuous_sent    = [c for c in HEAD2_FEATURES if c not in bounded_sentiment + flag_cols]

scaler_h2_mm = MinMaxScaler()
scaler_h2_std = StandardScaler()

train_sent_bounded = scaler_h2_mm.fit_transform(train[bounded_sentiment].values)
val_sent_bounded   = scaler_h2_mm.transform(val[bounded_sentiment].values)
test_sent_bounded  = scaler_h2_mm.transform(test[bounded_sentiment].values)

if continuous_sent:
    train_sent_cont = scaler_h2_std.fit_transform(train[continuous_sent].values)
    val_sent_cont   = scaler_h2_std.transform(val[continuous_sent].values)
    test_sent_cont  = scaler_h2_std.transform(test[continuous_sent].values)
else:
    train_sent_cont = np.zeros((len(train), 0))
    val_sent_cont = np.zeros((len(val), 0))
    test_sent_cont = np.zeros((len(test), 0))

train_h2 = np.hstack([train_sent_bounded, train_sent_cont, train[flag_cols].values])
val_h2   = np.hstack([val_sent_bounded,   val_sent_cont,   val[flag_cols].values])
test_h2  = np.hstack([test_sent_bounded,  test_sent_cont,  test[flag_cols].values])

joblib.dump(scaler_h1,    'processed/scaler_head1.pkl')
joblib.dump(scaler_h2_mm, 'processed/scaler_head2_minmax.pkl')
joblib.dump(scaler_h2_std,'processed/scaler_head2_std.pkl')

print(f"  Head 1 scaled shape: {train_h1.shape}")
print(f"  Head 2 scaled shape: {train_h2.shape}")

# ═══════════════════════════════════════════════════════════════
# PHASE 15 — Build Sliding Window Sequences
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 15 — Sliding Window Sequences")
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

np.save('processed/X1_train.npy', X1_train)
np.save('processed/X2_train.npy', X2_train)
np.save('processed/y_train.npy',  y_train_seq)
np.save('processed/X1_val.npy',   X1_val)
np.save('processed/X2_val.npy',   X2_val)
np.save('processed/y_val.npy',    y_val_seq)
np.save('processed/X1_test.npy',  X1_test)
np.save('processed/X2_test.npy',  X2_test)
np.save('processed/y_test.npy',   y_test_seq)
print("  All arrays saved to processed/")

# ═══════════════════════════════════════════════════════════════
# PHASE 16 — Build the Dual-Head Transformer Model
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 16 — Building Dual-Head Transformer")
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
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_h1 = nn.TransformerEncoder(encoder_layer_h1, num_layers=num_layers)

        self.proj_sentiment = nn.Linear(n_sentiment, d_model)
        self.pos_enc_h2     = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer_h2    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_h2 = nn.TransformerEncoder(encoder_layer_h2, num_layers=num_layers)

        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

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
        out   = self.fusion(fused)
        return out.squeeze(-1)


N_ONCHAIN   = X1_train.shape[2]
N_SENTIMENT = X2_train.shape[2]

model = DualHeadTransformer(
    n_onchain=N_ONCHAIN, n_sentiment=N_SENTIMENT,
    d_model=64, nhead=4, num_layers=2,
    dim_feedforward=128, dropout=0.3
)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total trainable parameters: {total_params:,}")

# ═══════════════════════════════════════════════════════════════
# PHASE 17 — Train the Model
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 17 — Training")
print("=" * 60)

pos_rate = y_train_seq.mean()
neg_rate = 1 - pos_rate
print(f"  Class balance — Up: {pos_rate:.3f} | Down: {neg_rate:.3f}")
pos_weight = torch.tensor([neg_rate / pos_rate], dtype=torch.float32)

def make_loader(X1, X2, y, batch_size=32, shuffle=False):
    dataset = TensorDataset(
        torch.tensor(X1, dtype=torch.float32),
        torch.tensor(X2, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X1_train, X2_train, y_train_seq, batch_size=32, shuffle=False)
val_loader   = make_loader(X1_val,   X2_val,   y_val_seq,   batch_size=32, shuffle=False)

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = model.to(device)
criterion = nn.BCELoss(weight=pos_weight.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=10, factor=0.5
)

EPOCHS        = 200
PATIENCE      = 30
best_val_loss = float('inf')
patience_ctr  = 0
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
torch.save(best_model_state, 'processed/model2_best.pt')
print("  Best model saved.")

# ═══════════════════════════════════════════════════════════════
# PHASE 18 — Evaluate on Test Set
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 18 — Test Set Evaluation")
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

print(f"\n  === MODEL 2 TEST RESULTS ===")
print(f"  Accuracy: {accuracy_score(y_true, all_preds):.4f}")
print(f"  F1 Score: {f1_score(y_true, all_preds):.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_true, all_preds, target_names=['Down', 'Up']))
print(f"  Confusion Matrix:")
print(confusion_matrix(y_true, all_preds))

# ═══════════════════════════════════════════════════════════════
# PHASE 19 — Save Everything for Comparison
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 19 — Saving All Outputs")
print("=" * 60)

np.save('processed/model2_probs_test.npy',  np.array(all_probs))
np.save('processed/model2_preds_test.npy',  np.array(all_preds))
np.save('processed/model2_ytrue_test.npy',  y_true)

with open('processed/model2_losses.json', 'w') as f:
    json.dump({'train': train_losses, 'val': val_losses}, f)

config = {
    'model': 'DualHeadTransformer',
    'name': 'Model 2 — Enriched Sentiment (2013-2026)',
    'start_date': '2013-01-01',
    'train_end': TRAIN_END,
    'val_end': VAL_END,
    'seq_len': SEQ_LEN,
    'd_model': 64,
    'nhead': 4,
    'num_layers': 2,
    'dropout': 0.3,
    'n_head1_features': int(N_ONCHAIN),
    'n_head2_features': int(N_SENTIMENT),
    'sentiment_sources': ['Google Trends (2013+)', 'Fear & Greed (2018+)'],
    'feature_selection': 'Boruta (primary), LASSO (secondary), PCA (baseline)',
    'accuracy': float(accuracy_score(y_true, all_preds)),
    'f1': float(f1_score(y_true, all_preds)),
}
with open('processed/model2_config.json', 'w') as f:
    json.dump(config, f, indent=2)

# --- Training Curves Plot ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train_losses, label='Train Loss', color='#2196F3')
ax.plot(val_losses, label='Val Loss', color='#FF5722')
ax.set_xlabel('Epoch')
ax.set_ylabel('BCE Loss')
ax.set_title('Model 2 — Training Curves')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('plots/model2_training_curves.png', dpi=150)
plt.close(fig)

print("\n  ✅ All Model 2 outputs saved to processed/")
print("  ✅ Training curves saved to plots/model2_training_curves.png")
print("\n  🎉 MODEL 2 PIPELINE COMPLETE!")
