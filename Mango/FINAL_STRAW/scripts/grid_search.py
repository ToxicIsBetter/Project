"""
GRID SEARCH — Exhaustive Hyperparameter Optimisation
=====================================================
Searches over: d_model, nhead, num_layers, dropout, lr, batch_size, SEQ_LEN
Auto-calibrates the decision threshold on the validation set per run.
Logs every result to grid_search_results.csv and saves the best model.

Usage (from FINAL_STRAW/):
    source ../../.venv/bin/activate
    python scripts/grid_search.py
"""
import os, sys, json, time, warnings, itertools, csv
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing  import StandardScaler, MinMaxScaler
from sklearn.metrics        import accuracy_score, f1_score, classification_report

import matplotlib.pyplot as plt
import random

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)          # FINAL_STRAW/
DATA_DIR   = os.path.join(ROOT_DIR, 'data')
OUT_DIR    = os.path.join(ROOT_DIR, 'grid_search_output')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Feature Lists (from feature_sets.json) ────────────────────────────
HEAD1_FEATURES = [
    'AdrActCnt', 'AdrBalCnt', 'TxCnt', 'HashRate', 'BlkCnt',
    'SplyExNtv', 'FlowInExNtv', 'CapMVRVCur',
    'AdrActCnt_growth7d', 'AdrActCnt_growth30d',
    'CapMVRVCur_growth7d', 'CapMVRVCur_growth30d',
    'momentum_7d',
]

HEAD2_FEATURES = [
    'google_trends', 'gt_ma7', 'gt_ma30', 'gt_change7', 'gt_momentum',
    'fear_greed', 'fg_ma7', 'fg_ma14', 'fg_change', 'fg_change7',
    'fg_extreme_fear', 'fg_extreme_greed',
]

BOUNDED_SENTIMENT = ['fear_greed', 'fg_ma7', 'fg_ma14', 'google_trends', 'gt_ma7', 'gt_ma30']
FLAG_COLS         = ['fg_extreme_fear', 'fg_extreme_greed']
CONTINUOUS_SENT   = [c for c in HEAD2_FEATURES if c not in BOUNDED_SENTIMENT + FLAG_COLS]

# Columns that need log1p transforms
LOG_POS = [
    'AdrActCnt', 'AdrBalCnt', 'TxCnt', 'TxTfrCnt', 'HashRate',
    'BlkCnt', 'SplyCur', 'SplyExNtv', 'SplyExpFut10yr',
    'FeeTotNtv', 'FlowInExNtv', 'FlowOutExNtv', 'IssTotNtv', 'CapMVRVCur',
]
SIGNED_LOG = [
    'TxCnt_growth7d', 'TxCnt_growth30d', 'HashRate_growth7d',
    'HashRate_growth30d', 'AdrActCnt_growth7d', 'AdrActCnt_growth30d',
    'CapMVRVCur_growth7d', 'CapMVRVCur_growth30d', 'gt_change7', 'gt_momentum',
]

# ══════════════════════════════════════════════════════════════════════
#  HYPERPARAMETER GRID
# ══════════════════════════════════════════════════════════════════════
PARAM_GRID = {
    'd_model':     [32, 64],
    'nhead':       [2, 4, 8],
    'num_layers':  [1, 2],
    'dropout':     [0.1, 0.2],
    'lr':          [5e-4, 1e-3, 2e-3],
    'batch_size':  [32, 64],
    'seq_len':     [5, 7],
}

# ── Prune invalid nhead/d_model combos ────────────────────────────────
def valid_combo(combo):
    """nhead must evenly divide d_model."""
    return combo['d_model'] % combo['nhead'] == 0

def generate_combos(grid):
    """Generate all valid grid combinations as list of dicts."""
    keys = list(grid.keys())
    values = list(grid.values())
    combos = []
    for vals in itertools.product(*values):
        combo = dict(zip(keys, vals))
        if valid_combo(combo):
            combos.append(combo)
    return combos

# ══════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════
def load_and_merge():
    """Load CSV files, merge, clean, engineer features, return DataFrame."""
    print("  Loading data...")
    ohlcv     = pd.read_csv(os.path.join(DATA_DIR, 'clean_ohlcv.csv'),     parse_dates=['Date'])
    onchain   = pd.read_csv(os.path.join(DATA_DIR, 'clean_onchain.csv'),   parse_dates=['Date'])
    sentiment = pd.read_csv(os.path.join(DATA_DIR, 'clean_sentiment.csv'), parse_dates=['Date'])
    google    = pd.read_csv(os.path.join(DATA_DIR, 'clean_google.csv'),    parse_dates=['Date'])

    for frame in [ohlcv, onchain, sentiment, google]:
        frame['Date'] = frame['Date'].dt.normalize()

    df = ohlcv.merge(onchain,   on='Date', how='left')
    df = df.merge(sentiment, on='Date', how='left')
    df = df.merge(google,    on='Date', how='left')
    df = df.sort_values('Date').reset_index(drop=True)

    # Drop leaky columns
    drop_cols = ['ROI1yr', 'CapMrktCurUSD', 'CapMrktEstUSD', 'ReferenceRate', 'SplyExUSD']
    drop_cols = [c for c in drop_cols if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # Date filter
    df = df[df['Date'] >= '2018-02-01'].reset_index(drop=True)

    # Fix Fear & Greed
    fg_cols = ['fear_greed', 'fg_ma7', 'fg_ma14', 'fg_change', 'fg_change7', 'fg_extreme_fear', 'fg_extreme_greed']
    for col in fg_cols:
        if col in df.columns:
            df[col] = df[col].replace(50.0, np.nan).ffill(limit=3).bfill(limit=3).fillna(0.0)

    # Add momentum features
    df['momentum_7d']    = df['Close'].pct_change(7)
    df['momentum_14d']   = df['Close'].pct_change(14)
    df['momentum_30d']   = df['Close'].pct_change(30)
    df['volatility_14d'] = df['Close'].pct_change().rolling(14).std() / (df['Close'].pct_change().rolling(14).mean() + 1e-6)
    df['ma_ratio']       = df['Close'] / (df['Close'].rolling(50).mean() + 1e-6)

    # Target
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.iloc[:-1].reset_index(drop=True)

    # Fill remaining NaNs
    df = df.ffill().bfill()

    # Log transforms
    for col in [c for c in LOG_POS if c in df.columns]:
        df[col] = np.log1p(df[col].clip(lower=0))
    for col in [c for c in SIGNED_LOG if c in df.columns]:
        df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

    print(f"  Data loaded: {len(df)} rows, Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


# ══════════════════════════════════════════════════════════════════════
#  SPLITTING & SCALING
# ══════════════════════════════════════════════════════════════════════
def split_and_scale(df):
    """Chronological 70/15/15 split, scale features, return numpy arrays."""
    n = len(df)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    train = df.iloc[:n_train].copy().reset_index(drop=True)
    val   = df.iloc[n_train:n_train + n_val].copy().reset_index(drop=True)
    test  = df.iloc[n_train + n_val:].copy().reset_index(drop=True)

    # Head 1 scaling (StandardScaler)
    scaler_h1 = StandardScaler()
    train_h1 = scaler_h1.fit_transform(train[HEAD1_FEATURES].values)
    val_h1   = scaler_h1.transform(val[HEAD1_FEATURES].values)
    test_h1  = scaler_h1.transform(test[HEAD1_FEATURES].values)

    # Head 2 scaling (MinMax for bounded, Standard for continuous, pass-through for flags)
    scaler_h2_mm  = MinMaxScaler()
    scaler_h2_std = StandardScaler()

    bounded_cols = [c for c in BOUNDED_SENTIMENT if c in HEAD2_FEATURES]
    flag_cols_   = [c for c in FLAG_COLS if c in HEAD2_FEATURES]
    cont_cols    = [c for c in CONTINUOUS_SENT if c in HEAD2_FEATURES]

    train_h2_b = scaler_h2_mm.fit_transform(train[bounded_cols].values)
    val_h2_b   = scaler_h2_mm.transform(val[bounded_cols].values)
    test_h2_b  = scaler_h2_mm.transform(test[bounded_cols].values)

    if cont_cols:
        train_h2_c = scaler_h2_std.fit_transform(train[cont_cols].values)
        val_h2_c   = scaler_h2_std.transform(val[cont_cols].values)
        test_h2_c  = scaler_h2_std.transform(test[cont_cols].values)
    else:
        train_h2_c = np.zeros((len(train), 0))
        val_h2_c   = np.zeros((len(val), 0))
        test_h2_c  = np.zeros((len(test), 0))

    train_h2 = np.hstack([train_h2_b, train_h2_c, train[flag_cols_].values])
    val_h2   = np.hstack([val_h2_b,   val_h2_c,   val[flag_cols_].values])
    test_h2  = np.hstack([test_h2_b,  test_h2_c,  test[flag_cols_].values])

    y_train = train['target'].values
    y_val   = val['target'].values
    y_test  = test['target'].values

    return (train_h1, train_h2, y_train,
            val_h1,   val_h2,   y_val,
            test_h1,  test_h2,  y_test,
            scaler_h1, scaler_h2_mm, scaler_h2_std)


# ══════════════════════════════════════════════════════════════════════
#  SEQUENCE BUILDER
# ══════════════════════════════════════════════════════════════════════
def build_sequences(h1, h2, y, seq_len):
    X1, X2, yy = [], [], []
    for i in range(seq_len, len(h1)):
        X1.append(h1[i - seq_len:i])
        X2.append(h2[i - seq_len:i])
        yy.append(y[i])
    return np.array(X1), np.array(X2), np.array(yy)


# ══════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE (matching FINAL_STRAW predict.py)
# ══════════════════════════════════════════════════════════════════════
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
    def __init__(self, input_dim_h1, input_dim_h2,
                 d_model=64, nhead=8, num_layers=2, dropout=0.3):
        super().__init__()
        self.head1_proj = nn.Linear(input_dim_h1, d_model)
        self.head2_proj = nn.Linear(input_dim_h2, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head1 = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.head2 = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        x1 = self.head1_proj(x1)
        x2 = self.head2_proj(x2)
        x  = (x1 + x2) / 2
        x  = self.pos_encoder(x)
        x  = self.transformer(x)
        x  = x[:, -1, :]
        h1 = self.dropout(self.head1(x))
        h2 = self.dropout(self.head2(x))
        return h1 + h2


# ══════════════════════════════════════════════════════════════════════
#  THRESHOLD CALIBRATION
# ══════════════════════════════════════════════════════════════════════
def calibrate_threshold(model, val_loader, device):
    """Find threshold that maximises balanced score on validation set."""
    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for X1b, X2b, yb in val_loader:
            X1b, X2b = X1b.to(device), X2b.to(device)
            logits = model(X1b, X2b)
            probs  = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_true.extend(yb.numpy().flatten())

    all_probs = np.array(all_probs)
    all_true  = np.array(all_true)

    best_t, best_f1, best_score = 0.5, 0.0, -1.0
    best_stats = {}
    for t in np.arange(0.10, 0.90, 0.01):
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(all_true, preds, zero_division=0)
        report = classification_report(all_true, preds, output_dict=True, zero_division=0)
        
        recall_down = report.get('0.0', report.get('0', {})).get('recall', 0.0)
        recall_up = report.get('1.0', report.get('1', {})).get('recall', 0.0)
        balance_penalty = abs(recall_up - recall_down)
        score = f1 - (0.5 * balance_penalty)

        if score > best_score:
            best_score = score
            best_f1 = f1
            best_t = t
            best_stats = {
                'balanced_val_score': score,
                'recall_down': recall_down,
                'recall_up': recall_up,
                'balance_penalty': balance_penalty
            }

    return round(best_t, 2), best_f1, best_stats


# ══════════════════════════════════════════════════════════════════════
#  SINGLE TRAINING RUN
# ══════════════════════════════════════════════════════════════════════
def train_one(params, train_h1, train_h2, y_train,
              val_h1, val_h2, y_val, device):
    """Train one model with given hyperparams. Returns metrics dict."""
    seq_len    = params['seq_len']
    batch_size = params['batch_size']

    # Build sequences
    X1_tr, X2_tr, y_tr = build_sequences(train_h1, train_h2, y_train, seq_len)
    X1_vl, X2_vl, y_vl = build_sequences(val_h1,   val_h2,   y_val,   seq_len)

    if len(X1_tr) == 0 or len(X1_vl) == 0:
        return None  # seq_len too large for data

    # Class weighting (sqrt scaling as per FINAL_STRAW)
    pos_rate = y_tr.mean()
    neg_rate = 1 - pos_rate
    weight_up   = np.sqrt(neg_rate / (pos_rate + 1e-8))
    weight_down = np.sqrt(pos_rate / (neg_rate + 1e-8))
    class_weights = torch.tensor([weight_down, weight_up], dtype=torch.float32).to(device)

    # DataLoaders
    def make_loader(X1, X2, y, bs, shuffle=False):
        ds = TensorDataset(
            torch.tensor(X1, dtype=torch.float32),
            torch.tensor(X2, dtype=torch.float32),
            torch.tensor(y,  dtype=torch.float32))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle)

    train_loader = make_loader(X1_tr, X2_tr, y_tr, batch_size, shuffle=True)
    val_loader   = make_loader(X1_vl, X2_vl, y_vl, batch_size, shuffle=False)

    # Model
    n_h1, n_h2 = X1_tr.shape[2], X2_tr.shape[2]
    model = DualHeadTransformer(
        input_dim_h1=n_h1, input_dim_h2=n_h2,
        d_model=params['d_model'], nhead=params['nhead'],
        num_layers=params['num_layers'], dropout=params['dropout']
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([weight_up], dtype=torch.float32).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=8, factor=0.5)

    # Training loop
    EPOCHS   = 150
    PATIENCE = 20
    best_val_loss    = float('inf')
    patience_ctr     = 0
    best_model_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        for X1b, X2b, yb in train_loader:
            X1b, X2b, yb = X1b.to(device), X2b.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(X1b, X2b).squeeze(-1)
            loss   = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_train = epoch_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X1b, X2b, yb in val_loader:
                X1b, X2b, yb = X1b.to(device), X2b.to(device), yb.to(device)
                logits   = model(X1b, X2b).squeeze(-1)
                val_loss += criterion(logits, yb).item()
        avg_val = val_loss / len(val_loader)

        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr     = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    if best_model_state is None:
        return None

    model.load_state_dict(best_model_state)

    # Calibrate threshold on val set
    threshold, val_f1, val_stats = calibrate_threshold(model, val_loader, device)

    return {
        'model':         model,
        'state_dict':    best_model_state,
        'best_val_loss': best_val_loss,
        'val_f1':        val_f1,
        'val_stats':     val_stats,
        'threshold':     threshold,
        'epochs_trained': epoch,
    }


# ══════════════════════════════════════════════════════════════════════
#  TEST SET EVALUATION
# ══════════════════════════════════════════════════════════════════════
def evaluate_test(model, test_h1, test_h2, y_test, seq_len, batch_size, threshold, device):
    """Evaluate model on the test set with the given threshold."""
    X1_te, X2_te, y_te = build_sequences(test_h1, test_h2, y_test, seq_len)
    if len(X1_te) == 0:
        return None

    ds = TensorDataset(
        torch.tensor(X1_te, dtype=torch.float32),
        torch.tensor(X2_te, dtype=torch.float32),
        torch.tensor(y_te,  dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for X1b, X2b, yb in loader:
            X1b, X2b = X1b.to(device), X2b.to(device)
            probs = torch.sigmoid(model(X1b, X2b)).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_true.extend(yb.numpy().flatten())

    all_probs = np.array(all_probs)
    all_true  = np.array(all_true)
    preds     = (all_probs >= threshold).astype(int)

    acc = accuracy_score(all_true, preds)
    f1  = f1_score(all_true, preds, zero_division=0)

    return {'accuracy': acc, 'f1': f1, 'preds': preds, 'probs': all_probs, 'y_true': all_true}


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  GRID SEARCH — Dual-Head Transformer Hyperparameter Optimisation")
    print("=" * 70)

    device = torch.device('cpu')  # CPU is sufficient for this model size
    print(f"  Device: {device}")

    # Load data once
    df = load_and_merge()
    (train_h1, train_h2, y_train,
     val_h1,   val_h2,   y_val,
     test_h1,  test_h2,  y_test,
     scaler_h1, scaler_h2_mm, scaler_h2_std) = split_and_scale(df)

    print(f"  Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    # Generate combos
    combos = generate_combos(PARAM_GRID)
    total  = len(combos)
    print(f"\n  Total valid hyperparameter combinations: {total}")
    print(f"  Estimated time: ~{total * 0.5:.0f}–{total * 2:.0f} minutes\n")

    # CSV log
    csv_path = os.path.join(OUT_DIR, 'grid_search_results.csv')
    csv_fields = [
        'rank', 'd_model', 'nhead', 'num_layers', 'dropout', 'lr', 'batch_size', 'seq_len',
        'threshold', 'balanced_val_score', 'val_f1', 'recall_down', 'recall_up', 'balance_penalty', 'val_loss',
        'test_accuracy', 'test_f1',
        'epochs_trained', 'time_seconds',
    ]
    csv_file = open(csv_path, 'w', newline='')
    writer   = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()

    # Track best
    best_val_f1   = 0.0
    best_result   = None
    best_params   = None
    results       = []

    for idx, params in enumerate(combos, 1):
        tag = (f"d={params['d_model']} nh={params['nhead']} nl={params['num_layers']} "
               f"do={params['dropout']} lr={params['lr']} bs={params['batch_size']} "
               f"seq={params['seq_len']}")
        print(f"  [{idx:4d}/{total}] {tag}", end=" ", flush=True)

        t0 = time.time()
        try:
            result = train_one(
                params, train_h1, train_h2, y_train,
                val_h1, val_h2, y_val, device)
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            continue

        if result is None:
            print("  ⏭ SKIP (no valid sequences)")
            continue

        elapsed = time.time() - t0

        # Evaluate on test set
        test_metrics = evaluate_test(
            result['model'], test_h1, test_h2, y_test,
            params['seq_len'], params['batch_size'],
            result['threshold'], device)

        if test_metrics is None:
            print("  ⏭ SKIP (test sequences empty)")
            continue

        # Log
        vstats = result['val_stats']
        row = {
            'rank':               0,
            'd_model':            params['d_model'],
            'nhead':              params['nhead'],
            'num_layers':         params['num_layers'],
            'dropout':            params['dropout'],
            'lr':                 params['lr'],
            'batch_size':         params['batch_size'],
            'seq_len':            params['seq_len'],
            'threshold':          result['threshold'],
            'balanced_val_score': round(vstats['balanced_val_score'], 4),
            'val_f1':             round(result['val_f1'], 4),
            'recall_down':        round(vstats['recall_down'], 4),
            'recall_up':          round(vstats['recall_up'], 4),
            'balance_penalty':    round(vstats['balance_penalty'], 4),
            'val_loss':           round(result['best_val_loss'], 4),
            'test_accuracy':      round(test_metrics['accuracy'], 4),
            'test_f1':            round(test_metrics['f1'], 4),
            'epochs_trained':     result['epochs_trained'],
            'time_seconds':       round(elapsed, 1),
        }
        writer.writerow(row)
        csv_file.flush()
        results.append((params, result, test_metrics, row))

        score = vstats['balanced_val_score']
        marker = "🏆 NEW BEST!" if score > best_val_f1 else ""
        print(f"  score={score:.4f}  f1={result['val_f1']:.4f}  "
              f"R_Dn={vstats['recall_down']:.2f}  R_Up={vstats['recall_up']:.2f}  "
              f"t={result['threshold']}  ({elapsed:.1f}s) {marker}")

        if score > best_val_f1:
            best_val_f1  = score
            best_result  = result
            best_params  = params
            best_test    = test_metrics

    csv_file.close()

    # ── Rank results and rewrite CSV ──────────────────────────────────
    results.sort(key=lambda x: x[3]['balanced_val_score'], reverse=True)
    ranked_rows = []
    for rank, (_, _, _, row) in enumerate(results, 1):
        row['rank'] = rank
        ranked_rows.append(row)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(ranked_rows)

    # ── Save best model ───────────────────────────────────────────────
    if best_result is not None:
        torch.save(best_result['state_dict'],
                   os.path.join(OUT_DIR, 'best_model.pt'))

        best_info = {
            'params':        best_params,
            'threshold':     best_result['threshold'],
            'val_f1':        best_result['val_f1'],
            'val_loss':      best_result['best_val_loss'],
            'test_accuracy': best_test['accuracy'],
            'test_f1':       best_test['f1'],
            'epochs_trained': best_result['epochs_trained'],
            'features_h1':   HEAD1_FEATURES,
            'features_h2':   HEAD2_FEATURES,
        }
        with open(os.path.join(OUT_DIR, 'best_params.json'), 'w') as f:
            json.dump(best_info, f, indent=2)

        # ── Print final report ────────────────────────────────────────
        print("\n" + "=" * 70)
        print("  🏆  GRID SEARCH COMPLETE — BEST HYPERPARAMETERS")
        print("=" * 70)
        print(f"  d_model:    {best_params['d_model']}")
        print(f"  nhead:      {best_params['nhead']}")
        print(f"  num_layers: {best_params['num_layers']}")
        print(f"  dropout:    {best_params['dropout']}")
        print(f"  lr:         {best_params['lr']}")
        print(f"  batch_size: {best_params['batch_size']}")
        print(f"  seq_len:    {best_params['seq_len']}")
        print(f"  threshold:  {best_result['threshold']}")
        print(f"")
        print(f"  Balanced Score: {best_result['val_stats']['balanced_val_score']:.4f}")
        print(f"  Val F1:         {best_result['val_f1']:.4f}")
        print(f"  Recall Down:    {best_result['val_stats']['recall_down']:.4f}")
        print(f"  Recall Up:      {best_result['val_stats']['recall_up']:.4f}")
        print(f"  Test Accuracy:  {best_test['accuracy']:.4f}")
        print(f"  Test F1:        {best_test['f1']:.4f}")
        print(f"")
        print(f"  Saved to: {OUT_DIR}/")
        print(f"    best_model.pt        — model weights")
        print(f"    best_params.json     — winning hyperparameters")
        print(f"    grid_search_results.csv — all {len(results)} runs ranked")
        print("=" * 70)

        # ── Plot top-20 combos ────────────────────────────────────────
        top_n = min(20, len(ranked_rows))
        labels  = [f"d{r['d_model']}_nh{r['nhead']}_nl{r['num_layers']}_do{r['dropout']}_lr{r['lr']}_bs{r['batch_size']}_s{r['seq_len']}"
                    for r in ranked_rows[:top_n]]
        val_f1s = [r['balanced_val_score'] for r in ranked_rows[:top_n]]
        test_f1s = [r['test_f1'] for r in ranked_rows[:top_n]]

        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(top_n)
        width = 0.35
        ax.barh(x - width/2, val_f1s, width, label='Balanced Val Score', color='#2196F3')
        ax.barh(x + width/2, test_f1s, width, label='Test F1', color='#FF5722')
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel('F1 Score')
        ax.set_title(f'Grid Search — Top {top_n} Hyperparameter Combinations')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, 'grid_search_top20.png'), dpi=150)
        plt.close(fig)
        print(f"\n  📊 Plot saved: grid_search_top20.png")

    else:
        print("\n  ❌ No valid runs completed.")

    print("\n  🎉 Grid search finished!")


if __name__ == '__main__':
    main()
