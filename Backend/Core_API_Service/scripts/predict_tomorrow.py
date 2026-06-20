"""Quick prediction script for tomorrow using fresh data."""
import pandas as pd, numpy as np, torch, torch.nn as nn, joblib, json, os, warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(ROOT_DIR, '..', 'NeuralEdge', 'models')

# Features
H1 = ['AdrActCnt','AdrBalCnt','TxCnt','HashRate','BlkCnt','SplyExNtv','FlowInExNtv','CapMVRVCur',
      'AdrActCnt_growth7d','AdrActCnt_growth30d','CapMVRVCur_growth7d','CapMVRVCur_growth30d','momentum_7d']
H2 = ['google_trends','gt_ma7','gt_ma30','gt_change7','gt_momentum','fear_greed','fg_ma7','fg_ma14',
      'fg_change','fg_change7','fg_extreme_fear','fg_extreme_greed']
BOUNDED = ['fear_greed','fg_ma7','fg_ma14','google_trends','gt_ma7','gt_ma30']
FLAGS = ['fg_extreme_fear','fg_extreme_greed']
CONT = [c for c in H2 if c not in BOUNDED + FLAGS]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

class DualHeadTransformer(nn.Module):
    def __init__(self, d1, d2, dm=32, nh=4, nl=2, do=0.1):
        super().__init__()
        self.head1_proj = nn.Linear(d1, dm)
        self.head2_proj = nn.Linear(d2, dm)
        self.pos_encoder = PositionalEncoding(dm, dropout=do)
        el = nn.TransformerEncoderLayer(d_model=dm, nhead=nh, dropout=do, batch_first=True)
        self.transformer = nn.TransformerEncoder(el, num_layers=nl)
        self.head1 = nn.Sequential(nn.Linear(dm, 32), nn.ReLU(), nn.Dropout(do), nn.Linear(32, 1))
        self.head2 = nn.Sequential(nn.Linear(dm, 32), nn.ReLU(), nn.Dropout(do), nn.Linear(32, 1))
        self.dropout = nn.Dropout(do)
    def forward(self, x1, x2):
        x1, x2 = self.head1_proj(x1), self.head2_proj(x2)
        x = self.pos_encoder((x1 + x2) / 2)
        x = self.transformer(x)[:, -1, :]
        return self.dropout(self.head1(x)) + self.dropout(self.head2(x))

print("Loading data (last 100 rows only)...")
onch_cols = ['Date'] + [c for c in H1 if c != 'momentum_7d']
ohlcv = pd.read_csv(os.path.join(DATA_DIR, 'clean_ohlcv.csv'), parse_dates=['Date']).tail(100)
onch = pd.read_csv(os.path.join(DATA_DIR, 'clean_onchain.csv'), usecols=onch_cols, parse_dates=['Date']).tail(100)
sent = pd.read_csv(os.path.join(DATA_DIR, 'clean_sentiment.csv'), parse_dates=['Date']).tail(100)
goog = pd.read_csv(os.path.join(DATA_DIR, 'clean_google.csv'), parse_dates=['Date']).tail(100)

for f in [ohlcv, onch, sent, goog]:
    f['Date'] = f['Date'].dt.normalize()

df = ohlcv.merge(onch, on='Date', how='left').merge(sent, on='Date', how='left').merge(goog, on='Date', how='left')
df = df.sort_values('Date').ffill().bfill()
df['momentum_7d'] = df['Close'].pct_change(7)

print(f"Latest date:  {df['Date'].iloc[-1].date()}")
print(f"BTC Close:    ${df['Close'].iloc[-1]:,.2f}")

print("Loading model...")
model = DualHeadTransformer(13, 12)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'model3_best.pt'), weights_only=True))
model.eval()

s1 = joblib.load(os.path.join(MODEL_DIR, 'scaler_head1.pkl'))
smm = joblib.load(os.path.join(MODEL_DIR, 'scaler_head2_minmax.pkl'))
ss = joblib.load(os.path.join(MODEL_DIR, 'scaler_head2_std.pkl'))
with open(os.path.join(MODEL_DIR, 'metrics_finetuned.json')) as f:
    threshold = json.load(f).get('threshold', 0.65)

recent = df.tail(7).copy()
print(f"Window:       {recent['Date'].iloc[0].date()} -> {recent['Date'].iloc[-1].date()}")

X1 = torch.FloatTensor(s1.transform(recent[H1].values)).unsqueeze(0)
X2 = torch.FloatTensor(np.hstack([
    smm.transform(recent[BOUNDED].values),
    ss.transform(recent[CONT].values),
    recent[FLAGS].values
])).unsqueeze(0)

with torch.no_grad():
    prob = torch.sigmoid(model(X1, X2)).item()
pred = 1 if prob >= threshold else 0
dist = abs(prob - threshold)
conf = 'HIGH' if dist > 0.15 else 'MEDIUM' if dist > 0.07 else 'LOW'

print()
print('=' * 50)
print('  NeuralEdge Prediction for June 1, 2026')
print('=' * 50)
print(f'  Direction  : {"📈 UP" if pred == 1 else "📉 DOWN"}')
print(f'  Probability: {prob:.4f} ({prob:.2%})')
print(f'  Threshold  : {threshold}')
print(f'  Confidence : {conf}')
print(f'  Model      : Challenger (GS Run 82)')
print('=' * 50)
