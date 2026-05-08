import os, json, random, warnings, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, recall_score
import joblib

warnings.filterwarnings('ignore')

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR) # NeuralEdge/
DATA_DIR   = os.path.join(ROOT_DIR, '../Core_API_Service/data')
MODEL_DIR  = os.path.join(ROOT_DIR, 'models')
PLOT_DIR   = os.path.join(ROOT_DIR, 'plots')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── TARGET HYPERPARAMETERS (GRID SEARCH WINNER) ──────────────────────
D_MODEL    = 32
NHEAD      = 4
NUM_LAYERS = 2
DROPOUT    = 0.1
LR         = 0.002
BATCH_SIZE = 32
SEQ_LEN    = 7
EPOCHS     = 200
PATIENCE   = 30

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
    'AdrActCnt', 'AdrBalCnt', 'TxCnt', 'HashRate',
    'BlkCnt', 'SplyExNtv', 'FlowInExNtv', 'CapMVRVCur',
]
SIGNED_LOG = [
    'AdrActCnt_growth7d', 'AdrActCnt_growth30d',
    'CapMVRVCur_growth7d', 'CapMVRVCur_growth30d', 'gt_change7', 'gt_momentum',
]

# ══════════════════════════════════════════════════════════════════════
#  ARCHITECTURE
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
    def __init__(self, input_dim_h1, input_dim_h2, d_model=32, nhead=4, num_layers=2, dropout=0.1):
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

# ══════════════════════════════════════════════════════════════════════
#  DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════
def load_data():
    ohlcv = pd.read_csv(os.path.join(DATA_DIR, 'clean_ohlcv.csv'), parse_dates=['Date'])
    onchain = pd.read_csv(os.path.join(DATA_DIR, 'clean_onchain.csv'), parse_dates=['Date'])
    sentiment = pd.read_csv(os.path.join(DATA_DIR, 'clean_sentiment.csv'), parse_dates=['Date'])
    google = pd.read_csv(os.path.join(DATA_DIR, 'clean_google.csv'), parse_dates=['Date'])

    for frame in [ohlcv, onchain, sentiment, google]:
        frame['Date'] = frame['Date'].dt.normalize()

    df = ohlcv.merge(onchain, on='Date', how='left').merge(sentiment, on='Date', how='left').merge(google, on='Date', how='left')
    df = df.sort_values('Date').reset_index(drop=True)
    df = df[df['Date'] >= '2018-02-01'].reset_index(drop=True)

    df['momentum_7d'] = df['Close'].pct_change(7)
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.iloc[:-1].ffill().bfill().reset_index(drop=True)

    for col in LOG_POS:
        if col in df.columns: df[col] = np.log1p(df[col].clip(lower=0))
    for col in SIGNED_LOG:
        if col in df.columns: df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

    return df

def build_sequences(h1, h2, y, seq_len):
    X1, X2, yy = [], [], []
    for i in range(seq_len, len(h1)):
        X1.append(h1[i - seq_len:i])
        X2.append(h2[i - seq_len:i])
        yy.append(y[i])
    return np.array(X1), np.array(X2), np.array(yy)

def main():
    print(f"🚀 NeuralEdge Challenger Training Service")
    print(f"   d={D_MODEL}, nh={NHEAD}, nl={NUM_LAYERS}, lr={LR}\n")

    device = torch.device('cpu')
    df = load_data()
    n = len(df)
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    train, val, test = df.iloc[:n_train], df.iloc[n_train:n_train+n_val], df.iloc[n_train+n_val:]

    # Scaling
    s1 = StandardScaler()
    tr_h1 = s1.fit_transform(train[HEAD1_FEATURES])
    vl_h1 = s1.transform(val[HEAD1_FEATURES])
    te_h1 = s1.transform(test[HEAD1_FEATURES])

    sm = MinMaxScaler()
    ss = StandardScaler()
    tr_h2_b = sm.fit_transform(train[BOUNDED_SENTIMENT])
    vl_h2_b = sm.transform(val[BOUNDED_SENTIMENT])
    te_h2_b = sm.transform(test[BOUNDED_SENTIMENT])
    tr_h2_c = ss.fit_transform(train[CONTINUOUS_SENT])
    vl_h2_c = ss.transform(val[CONTINUOUS_SENT])
    te_h2_c = ss.transform(test[CONTINUOUS_SENT])

    tr_h2 = np.hstack([tr_h2_b, tr_h2_c, train[FLAG_COLS].values])
    vl_h2 = np.hstack([vl_h2_b, vl_h2_c, val[FLAG_COLS].values])
    te_h2 = np.hstack([te_h2_b, te_h2_c, test[FLAG_COLS].values])

    X1_tr, X2_tr, y_tr = build_sequences(tr_h1, tr_h2, train['target'].values, SEQ_LEN)
    X1_vl, X2_vl, y_vl = build_sequences(vl_h1, vl_h2, val['target'].values, SEQ_LEN)
    X1_te, X2_te, y_te = build_sequences(te_h1, te_h2, test['target'].values, SEQ_LEN)

    # Loader
    train_loader = DataLoader(TensorDataset(torch.tensor(X1_tr, dtype=torch.float32), torch.tensor(X2_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X1_vl, dtype=torch.float32), torch.tensor(X2_vl, dtype=torch.float32), torch.tensor(y_vl, dtype=torch.float32)), batch_size=BATCH_SIZE)

    model = DualHeadTransformer(X1_tr.shape[2], X2_tr.shape[2], D_MODEL, NHEAD, NUM_LAYERS, DROPOUT).to(device)
    pos_weight = torch.tensor([np.sqrt((1 - y_tr.mean()) / (y_tr.mean() + 1e-8))]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=8, factor=0.5)

    best_val_loss, patience_ctr, best_state = float('inf'), 0, None
    for epoch in range(1, EPOCHS+1):
        model.train()
        tloss = 0
        for x1b, x2b, yb in train_loader:
            optimizer.zero_grad(); loss = criterion(model(x1b, x2b).squeeze(-1), yb); loss.backward(); optimizer.step(); tloss += loss.item()
        
        model.eval(); vloss = 0
        with torch.no_grad():
            for x1b, x2b, yb in val_loader: vloss += criterion(model(x1b, x2b).squeeze(-1), yb).item()
        vloss /= len(val_loader)
        scheduler.step(vloss)

        if vloss < best_val_loss:
            best_val_loss, best_state, patience_ctr = vloss, model.state_dict(), 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE: break
        if epoch % 10 == 0: print(f"   Epoch {epoch:03} | Val Loss: {vloss:.4f}")

    model.load_state_dict(best_state)
    
    # Calibrate Threshold
    probs, trues = [], []
    with torch.no_grad():
        for x1b, x2b, yb in val_loader: probs.extend(torch.sigmoid(model(x1b, x2b)).numpy().flatten()); trues.extend(yb.numpy())
    probs, trues = np.array(probs), np.array(trues)
    
    best_t, best_score = 0.5, -1
    for t in np.arange(0.2, 0.8, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(trues, preds, zero_division=0)
        rd = recall_score(trues, preds, pos_label=0, zero_division=0)
        ru = recall_score(trues, preds, pos_label=1, zero_division=0)
        score = f1 - (0.5 * abs(rd - ru))
        if score > best_score: best_t, best_score = t, score

    # Test
    X1_te_t, X2_te_t, y_te_t = torch.tensor(X1_te, dtype=torch.float32), torch.tensor(X2_te, dtype=torch.float32), torch.tensor(y_te, dtype=torch.float32)
    with torch.no_grad(): t_probs = torch.sigmoid(model(X1_te_t, X2_te_t)).numpy().flatten()
    t_preds = (t_probs >= best_t).astype(int)
    acc, f1 = accuracy_score(y_te, t_preds), f1_score(y_te, t_preds)

    # Save
    torch.save(best_state, os.path.join(MODEL_DIR, 'model3_best.pt'))
    joblib.dump(s1, os.path.join(MODEL_DIR, 'scaler_head1.pkl'))
    joblib.dump(sm, os.path.join(MODEL_DIR, 'scaler_head2_minmax.pkl'))
    joblib.dump(ss, os.path.join(MODEL_DIR, 'scaler_head2_std.pkl'))
    
    from sklearn.metrics import roc_auc_score
    auc_score = roc_auc_score(y_te, t_probs)
    print(f"\n✅ Training Complete. Acc: {acc:.2%}, F1: {f1:.4f}, Score: {best_score:.4f}, T: {best_t:.2f}, AUC: {auc_score:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_te, t_preds))
    print("\nClassification Report:")
    print(classification_report(y_te, t_preds))

if __name__ == "__main__":
    main()
