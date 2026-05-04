"""
RETRAIN RANK 3 — Production Model
===================================
Hyperparams: d=32, nh=8, nl=1, do=0.1, lr=1e-3, bs=32, seq=7, threshold=0.31
Target: 59.01% accuracy, 0.616 F1

Saves all artifacts into FINAL_STRAW/models/ for deployment.
"""
import os, json, random, warnings, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Set seed for reproducibility
SEED = 128
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(False)  # We don't need strict determinism as long as initial weights are seeded

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, 'data')
MODEL_DIR  = os.path.join(ROOT_DIR, 'models')
PLOT_DIR   = os.path.join(ROOT_DIR, 'plots')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── RANK 3 HYPERPARAMETERS ────────────────────────────────────────────
D_MODEL    = 32
NHEAD      = 8
NUM_LAYERS = 1
DROPOUT    = 0.1
LR         = 1e-3
BATCH_SIZE = 32
SEQ_LEN    = 7
THRESHOLD  = 0.31
EPOCHS     = 200
PATIENCE   = 25

# ── Feature Lists ─────────────────────────────────────────────────────
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
#  MODEL ARCHITECTURE
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
                 d_model=32, nhead=8, num_layers=1, dropout=0.1):
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
#  DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════
def load_and_merge():
    print("  Loading data...")
    ohlcv     = pd.read_csv(os.path.join(DATA_DIR, 'clean_ohlcv.csv'),     parse_dates=['Date'])
    onchain   = pd.read_csv(os.path.join(DATA_DIR, 'clean_onchain.csv'),   parse_dates=['Date'])
    sentiment = pd.read_csv(os.path.join(DATA_DIR, 'clean_sentiment.csv'), parse_dates=['Date'])
    google    = pd.read_csv(os.path.join(DATA_DIR, 'clean_google.csv'),    parse_dates=['Date'])

    for frame in [ohlcv, onchain, sentiment, google]:
        frame['Date'] = frame['Date'].dt.normalize()

    df = ohlcv.merge(onchain, on='Date', how='left')
    df = df.merge(sentiment, on='Date', how='left')
    df = df.merge(google, on='Date', how='left')
    df = df.sort_values('Date').reset_index(drop=True)

    drop_cols = ['ROI1yr', 'CapMrktCurUSD', 'CapMrktEstUSD', 'ReferenceRate', 'SplyExUSD']
    drop_cols = [c for c in drop_cols if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    df = df[df['Date'] >= '2018-02-01'].reset_index(drop=True)

    fg_cols = ['fear_greed', 'fg_ma7', 'fg_ma14', 'fg_change', 'fg_change7', 'fg_extreme_fear', 'fg_extreme_greed']
    for col in fg_cols:
        if col in df.columns:
            df[col] = df[col].replace(50.0, np.nan).ffill(limit=3).bfill(limit=3).fillna(0.0)

    df['momentum_7d']    = df['Close'].pct_change(7)
    df['momentum_14d']   = df['Close'].pct_change(14)
    df['momentum_30d']   = df['Close'].pct_change(30)
    df['volatility_14d'] = df['Close'].pct_change().rolling(14).std() / (df['Close'].pct_change().rolling(14).mean() + 1e-6)
    df['ma_ratio']       = df['Close'] / (df['Close'].rolling(50).mean() + 1e-6)

    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.iloc[:-1].reset_index(drop=True)
    df = df.ffill().bfill()

    for col in [c for c in LOG_POS if c in df.columns]:
        df[col] = np.log1p(df[col].clip(lower=0))
    for col in [c for c in SIGNED_LOG if c in df.columns]:
        df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

    print(f"  Data loaded: {len(df)} rows")
    return df


def build_sequences(h1, h2, y, seq_len):
    X1, X2, yy = [], [], []
    for i in range(seq_len, len(h1)):
        X1.append(h1[i - seq_len:i])
        X2.append(h2[i - seq_len:i])
        yy.append(y[i])
    return np.array(X1), np.array(X2), np.array(yy)


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  RETRAINING RANK 3 — Production Model")
    print("  d=32, nh=8, nl=1, do=0.1, lr=1e-3, bs=32, seq=7, t=0.31")
    print("=" * 70)

    device = torch.device('cpu')
    df = load_and_merge()

    # ── Split ─────────────────────────────────────────────────────────
    n = len(df)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    train = df.iloc[:n_train].copy().reset_index(drop=True)
    val   = df.iloc[n_train:n_train + n_val].copy().reset_index(drop=True)
    test  = df.iloc[n_train + n_val:].copy().reset_index(drop=True)
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # ── Scale Head 1 ──────────────────────────────────────────────────
    scaler_h1 = StandardScaler()
    train_h1 = scaler_h1.fit_transform(train[HEAD1_FEATURES].values)
    val_h1   = scaler_h1.transform(val[HEAD1_FEATURES].values)
    test_h1  = scaler_h1.transform(test[HEAD1_FEATURES].values)

    # ── Scale Head 2 ──────────────────────────────────────────────────
    bounded = [c for c in BOUNDED_SENTIMENT if c in HEAD2_FEATURES]
    flags   = [c for c in FLAG_COLS if c in HEAD2_FEATURES]
    cont    = [c for c in CONTINUOUS_SENT if c in HEAD2_FEATURES]

    scaler_h2_mm  = MinMaxScaler()
    scaler_h2_std = StandardScaler()

    train_h2_b = scaler_h2_mm.fit_transform(train[bounded].values)
    val_h2_b   = scaler_h2_mm.transform(val[bounded].values)
    test_h2_b  = scaler_h2_mm.transform(test[bounded].values)

    if cont:
        train_h2_c = scaler_h2_std.fit_transform(train[cont].values)
        val_h2_c   = scaler_h2_std.transform(val[cont].values)
        test_h2_c  = scaler_h2_std.transform(test[cont].values)
    else:
        train_h2_c = np.zeros((len(train), 0))
        val_h2_c   = np.zeros((len(val), 0))
        test_h2_c  = np.zeros((len(test), 0))

    train_h2 = np.hstack([train_h2_b, train_h2_c, train[flags].values])
    val_h2   = np.hstack([val_h2_b,   val_h2_c,   val[flags].values])
    test_h2  = np.hstack([test_h2_b,  test_h2_c,  test[flags].values])

    y_train = train['target'].values
    y_val   = val['target'].values
    y_test  = test['target'].values

    # ── Sequences ─────────────────────────────────────────────────────
    X1_tr, X2_tr, y_tr = build_sequences(train_h1, train_h2, y_train, SEQ_LEN)
    X1_vl, X2_vl, y_vl = build_sequences(val_h1,   val_h2,   y_val,   SEQ_LEN)
    X1_te, X2_te, y_te = build_sequences(test_h1,  test_h2,  y_test,  SEQ_LEN)

    print(f"  Sequences — Train: {len(y_tr)} | Val: {len(y_vl)} | Test: {len(y_te)}")

    # ── Class weights (sqrt scaling) ──────────────────────────────────
    pos_rate = y_tr.mean()
    neg_rate = 1 - pos_rate
    weight_up = np.sqrt(neg_rate / (pos_rate + 1e-8))

    # ── DataLoaders ───────────────────────────────────────────────────
    def make_loader(X1, X2, y, bs, shuffle=False):
        ds = TensorDataset(
            torch.tensor(X1, dtype=torch.float32),
            torch.tensor(X2, dtype=torch.float32),
            torch.tensor(y,  dtype=torch.float32))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle)

    train_loader = make_loader(X1_tr, X2_tr, y_tr, BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(X1_vl, X2_vl, y_vl, BATCH_SIZE)
    test_loader  = make_loader(X1_te, X2_te, y_te, BATCH_SIZE)

    # ── Model ─────────────────────────────────────────────────────────
    model = DualHeadTransformer(
        input_dim_h1=X1_tr.shape[2], input_dim_h2=X2_tr.shape[2],
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dropout=DROPOUT
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([weight_up], dtype=torch.float32).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=8, factor=0.5)

    # ── Training ──────────────────────────────────────────────────────
    print(f"\n  Training for max {EPOCHS} epochs (patience={PATIENCE})...\n")
    best_val_loss    = float('inf')
    patience_ctr     = 0
    best_model_state = None
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        for X1b, X2b, yb in train_loader:
            X1b, X2b, yb = X1b.to(device), X2b.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(X1b, X2b).squeeze(-1)
            loss = criterion(logits, yb)
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
                logits = model(X1b, X2b).squeeze(-1)
                val_loss += criterion(logits, yb).item()
        avg_val = val_loss / len(val_loader)

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:03d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n    ⏹ Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
                break

    model.load_state_dict(best_model_state)
    epochs_trained = epoch

    # ── Threshold calibration on validation set ───────────────────────
    print("\n  Calibrating threshold on validation set...")
    model.eval()
    val_probs, val_true = [], []
    with torch.no_grad():
        for X1b, X2b, yb in val_loader:
            probs = torch.sigmoid(model(X1b.to(device), X2b.to(device))).cpu().numpy().flatten()
            val_probs.extend(probs)
            val_true.extend(yb.numpy().flatten())
    val_probs = np.array(val_probs)
    val_true  = np.array(val_true)

    best_t, best_val_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.90, 0.01):
        f1 = f1_score(val_true, (val_probs >= t).astype(int), zero_division=0)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_t = round(t, 2)

    print(f"  Optimal threshold: {best_t} (Val F1: {best_val_f1:.4f})")

    # ── Test evaluation ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TEST SET EVALUATION")
    print("=" * 70)
    test_probs, test_true = [], []
    with torch.no_grad():
        for X1b, X2b, yb in test_loader:
            probs = torch.sigmoid(model(X1b.to(device), X2b.to(device))).cpu().numpy().flatten()
            test_probs.extend(probs)
            test_true.extend(yb.numpy().flatten())
    test_probs = np.array(test_probs)
    test_true  = np.array(test_true)
    test_preds = (test_probs >= best_t).astype(int)

    acc = accuracy_score(test_true, test_preds)
    f1  = f1_score(test_true, test_preds, zero_division=0)

    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Threshold: {best_t}")
    print(f"\n  Classification Report:")
    print(classification_report(test_true, test_preds, target_names=['Down', 'Up']))
    print(f"  Confusion Matrix:")
    cm = confusion_matrix(test_true, test_preds)
    print(f"              Predicted")
    print(f"              Down   Up")
    print(f"  Actual Down  {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"  Actual Up    {cm[1][0]:4d}  {cm[1][1]:4d}")

    # ── Save everything ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SAVING PRODUCTION ARTIFACTS")
    print("=" * 70)

    torch.save(best_model_state, os.path.join(MODEL_DIR, 'model3_best.pt'))
    print(f"  ✅ Model weights → models/model3_best.pt")

    joblib.dump(scaler_h1,     os.path.join(MODEL_DIR, 'scaler_head1.pkl'))
    joblib.dump(scaler_h2_mm,  os.path.join(MODEL_DIR, 'scaler_head2_minmax.pkl'))
    joblib.dump(scaler_h2_std, os.path.join(MODEL_DIR, 'scaler_head2_std.pkl'))
    print(f"  ✅ Scalers → models/scaler_head1.pkl, scaler_head2_*.pkl")

    feature_config = {
        'onchain_lasso': HEAD1_FEATURES,
        'final_h1': HEAD1_FEATURES,
        'sentiment': HEAD2_FEATURES,
    }
    with open(os.path.join(MODEL_DIR, 'feature_sets.json'), 'w') as f:
        json.dump(feature_config, f, indent=2)
    print(f"  ✅ Features → models/feature_sets.json")

    metrics = {
        'accuracy': acc,
        'f1': f1,
        'best_threshold': best_t,
        'best_val_loss': best_val_loss,
        'val_f1': best_val_f1,
        'epochs_trained': epochs_trained,
        'hyperparameters': {
            'd_model': D_MODEL,
            'nhead': NHEAD,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
            'lr': LR,
            'batch_size': BATCH_SIZE,
            'seq_len': SEQ_LEN,
        },
        'grid_search_rank': 3,
        'features': HEAD1_FEATURES + HEAD2_FEATURES,
    }
    with open(os.path.join(MODEL_DIR, 'metrics_finetuned.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✅ Metrics → models/metrics_finetuned.json")

    # ── Training curves plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label='Train Loss', color='#2196F3', linewidth=2)
    ax.plot(val_losses,   label='Val Loss',   color='#FF5722', linewidth=2)
    ax.axvline(x=epochs_trained - PATIENCE, color='gray', linestyle='--', alpha=0.5, label='Best epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Rank 3 Production Model — Training Curves (d=32, nh=8, nl=1, do=0.1, seq=7)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'training_curves_finetuned.png'), dpi=150)
    plt.close(fig)
    print(f"  ✅ Plot → plots/training_curves_finetuned.png")

    # ── ROC Curve ─────────────────────────────────────────────────────
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(test_true, test_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, color='#2196F3', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Rank 3 Production Model — ROC Curve')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'roc_curve_finetuned.png'), dpi=150)
    plt.close(fig)
    print(f"  ✅ Plot → plots/roc_curve_finetuned.png")

    print("\n" + "=" * 70)
    print("  🎉 RANK 3 PRODUCTION MODEL SAVED SUCCESSFULLY!")
    print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | Threshold: {best_t}")
    print("=" * 70)


if __name__ == '__main__':
    main()
