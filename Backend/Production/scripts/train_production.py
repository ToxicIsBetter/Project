import os, json, random, warnings, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

warnings.filterwarnings('ignore')

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR) # NeuralEdge_Production/
DATA_SRC   = os.path.join(os.path.dirname(ROOT_DIR), 'NeuralEdge', 'data') # Use existing clean files
MODEL_DIR  = os.path.join(ROOT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── TARGET HYPERPARAMETERS (GRID SEARCH WINNER) ──────────────────────
D_MODEL    = 32
NHEAD      = 4
NUM_LAYERS = 2
DROPOUT    = 0.1
LR         = 0.002
BATCH_SIZE = 32
SEQ_LEN    = 7
EPOCHS     = 250  # Increased slightly for full coverage
THRESHOLD  = 0.65 # Institutional benchmark

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
    ohlcv = pd.read_csv(os.path.join(DATA_SRC, 'clean_ohlcv.csv'), parse_dates=['Date'])
    onchain = pd.read_csv(os.path.join(DATA_SRC, 'clean_onchain.csv'), parse_dates=['Date'])
    sentiment = pd.read_csv(os.path.join(DATA_SRC, 'clean_sentiment.csv'), parse_dates=['Date'])
    google = pd.read_csv(os.path.join(DATA_SRC, 'clean_google.csv'), parse_dates=['Date'])

    for frame in [ohlcv, onchain, sentiment, google]:
        frame['Date'] = frame['Date'].dt.normalize()

    df = ohlcv.merge(onchain, on='Date', how='left').merge(sentiment, on='Date', how='left').merge(google, on='Date', how='left')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # PRODUCTION FILTER: 2018 to Today
    df = df[df['Date'] >= '2018-01-01'].reset_index(drop=True)

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
    print(f"📡 Entering PRODUCTION Mode: Training on 100% of Available Data")
    print(f"📊 Range: 2018-01-01 to LATEST")
    
    device = torch.device('cpu')
    df = load_data()
    print(f"📈 Total training samples found: {len(df)}")

    # Scaling on 100% Data
    s1 = StandardScaler()
    full_h1 = s1.fit_transform(df[HEAD1_FEATURES])

    sm = MinMaxScaler()
    ss = StandardScaler()
    h2_b = sm.fit_transform(df[BOUNDED_SENTIMENT])
    h2_c = ss.fit_transform(df[CONTINUOUS_SENT])
    full_h2 = np.hstack([h2_b, h2_c, df[FLAG_COLS].values])

    X1, X2, y = build_sequences(full_h1, full_h2, df['target'].values, SEQ_LEN)

    # Loader (No Shuffle for sequence continuity if desired, but shuffle is fine for transformer)
    train_loader = DataLoader(TensorDataset(torch.tensor(X1, dtype=torch.float32), torch.tensor(X2, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True)

    model = DualHeadTransformer(X1.shape[2], X2.shape[2], D_MODEL, NHEAD, NUM_LAYERS, DROPOUT).to(device)
    pos_weight = torch.tensor([np.sqrt((1 - y.mean()) / (y.mean() + 1e-8))]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    print(f"🛠️ Starting full-data immersion for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS+1):
        model.train()
        epoch_loss = 0
        for x1b, x2b, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x1b, x2b).squeeze(-1), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 50 == 0 or epoch == 1:
            print(f"   Epoch {epoch:03} | Training Loss: {epoch_loss/len(train_loader):.4f}")

    # Generate Metadata
    final_metrics = {
        "architecture": f"d{D_MODEL}/h{NHEAD}/l{NUM_LAYERS}",
        "threshold": THRESHOLD,
        "data_utilization": "100%",
        "range": "2018-01-01 to Present",
        "timestamp": pd.Timestamp.now().isoformat()
    }

    # Save Production Artifacts
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'production_model.pt'))
    joblib.dump(s1, os.path.join(MODEL_DIR, 'scaler_head1.pkl'))
    joblib.dump(sm, os.path.join(MODEL_DIR, 'scaler_head2_minmax.pkl'))
    joblib.dump(ss, os.path.join(MODEL_DIR, 'scaler_head2_std.pkl'))
    
    with open(os.path.join(MODEL_DIR, 'production_metadata.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)

    print(f"\n✅ Production Model Finalized and Saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()
