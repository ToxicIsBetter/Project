"""
COMPARE TOP 3 — Train Grid Search top 3 configs on corrected data
==================================================================
Rank 1: d=32, nh=4, nl=1, do=0.1, lr=1e-3, bs=32, seq=7
Rank 2: d=64, nh=2, nl=3, do=0.1, lr=5e-4, bs=64, seq=7
Rank 3: d=32, nh=8, nl=1, do=0.1, lr=1e-3, bs=32, seq=7
"""
import os, json, random, warnings, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(False)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, 'data')
MODEL_DIR  = os.path.join(ROOT_DIR, 'models')
PLOT_DIR   = os.path.join(ROOT_DIR, 'plots')

# ── Top 3 Configs ─────────────────────────────────────────────────────
CONFIGS = [
    {'name': 'Rank 1', 'd_model': 32, 'nhead': 4,  'num_layers': 1, 'dropout': 0.1, 'lr': 1e-3, 'batch_size': 32, 'seq_len': 7},
    {'name': 'Rank 2', 'd_model': 64, 'nhead': 2,  'num_layers': 3, 'dropout': 0.1, 'lr': 5e-4, 'batch_size': 64, 'seq_len': 7},
    {'name': 'Rank 3', 'd_model': 32, 'nhead': 8,  'num_layers': 1, 'dropout': 0.1, 'lr': 1e-3, 'batch_size': 32, 'seq_len': 7},
]

HEAD1_FEATURES = [
    'AdrActCnt', 'AdrBalCnt', 'TxCnt', 'HashRate', 'BlkCnt',
    'SplyExNtv', 'FlowInExNtv', 'CapMVRVCur',
    'AdrActCnt_growth7d', 'AdrActCnt_growth30d',
    'CapMVRVCur_growth7d', 'CapMVRVCur_growth30d', 'momentum_7d',
]
HEAD2_FEATURES = [
    'google_trends', 'gt_ma7', 'gt_ma30', 'gt_change7', 'gt_momentum',
    'fear_greed', 'fg_ma7', 'fg_ma14', 'fg_change', 'fg_change7',
    'fg_extreme_fear', 'fg_extreme_greed',
]
BOUNDED_SENTIMENT = ['fear_greed', 'fg_ma7', 'fg_ma14', 'google_trends', 'gt_ma7', 'gt_ma30']
FLAG_COLS = ['fg_extreme_fear', 'fg_extreme_greed']
CONTINUOUS_SENT = [c for c in HEAD2_FEATURES if c not in BOUNDED_SENTIMENT + FLAG_COLS]
LOG_POS = ['AdrActCnt','AdrBalCnt','TxCnt','TxTfrCnt','HashRate','BlkCnt','SplyCur','SplyExNtv','SplyExpFut10yr','FeeTotNtv','FlowInExNtv','FlowOutExNtv','IssTotNtv','CapMVRVCur']
SIGNED_LOG = ['TxCnt_growth7d','TxCnt_growth30d','HashRate_growth7d','HashRate_growth30d','AdrActCnt_growth7d','AdrActCnt_growth30d','CapMVRVCur_growth7d','CapMVRVCur_growth30d','gt_change7','gt_momentum']

# ── Architecture ──────────────────────────────────────────────────────
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
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class DualHeadTransformer(nn.Module):
    def __init__(self, n_h1, n_h2, d_model=32, nhead=8, num_layers=1, dropout=0.1):
        super().__init__()
        self.head1_proj = nn.Linear(n_h1, d_model)
        self.head2_proj = nn.Linear(n_h2, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
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
        return self.dropout(self.head1(x)) + self.dropout(self.head2(x))

# ── Data Pipeline ─────────────────────────────────────────────────────
def load_data():
    ohlcv = pd.read_csv(os.path.join(DATA_DIR, 'clean_ohlcv.csv'), parse_dates=['Date'])
    onchain = pd.read_csv(os.path.join(DATA_DIR, 'clean_onchain.csv'), parse_dates=['Date'])
    sentiment = pd.read_csv(os.path.join(DATA_DIR, 'clean_sentiment.csv'), parse_dates=['Date'])
    google = pd.read_csv(os.path.join(DATA_DIR, 'clean_google.csv'), parse_dates=['Date'])
    for f in [ohlcv, onchain, sentiment, google]: f['Date'] = f['Date'].dt.normalize()
    df = ohlcv.merge(onchain, on='Date', how='left').merge(sentiment, on='Date', how='left').merge(google, on='Date', how='left')
    df = df.sort_values('Date').reset_index(drop=True)
    drop = [c for c in ['ROI1yr','CapMrktCurUSD','CapMrktEstUSD','ReferenceRate','SplyExUSD'] if c in df.columns]
    df.drop(columns=drop, inplace=True)
    df = df[df['Date'] >= '2018-02-01'].reset_index(drop=True)
    for col in ['fear_greed','fg_ma7','fg_ma14','fg_change','fg_change7','fg_extreme_fear','fg_extreme_greed']:
        if col in df.columns: df[col] = df[col].replace(50.0, np.nan).ffill(limit=3).bfill(limit=3).fillna(0.0)
    df['momentum_7d'] = df['Close'].pct_change(7)
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.iloc[:-1].reset_index(drop=True)
    df = df.ffill().bfill()
    for col in [c for c in LOG_POS if c in df.columns]: df[col] = np.log1p(df[col].clip(lower=0))
    for col in [c for c in SIGNED_LOG if c in df.columns]: df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
    return df

def split_scale(df):
    n = len(df); n_tr = int(n*0.70); n_vl = int(n*0.15)
    train, val, test = df.iloc[:n_tr], df.iloc[n_tr:n_tr+n_vl], df.iloc[n_tr+n_vl:]
    sc1 = StandardScaler(); sc2mm = MinMaxScaler(); sc2std = StandardScaler()
    tr_h1 = sc1.fit_transform(train[HEAD1_FEATURES].values)
    vl_h1 = sc1.transform(val[HEAD1_FEATURES].values)
    te_h1 = sc1.transform(test[HEAD1_FEATURES].values)
    b = [c for c in BOUNDED_SENTIMENT if c in HEAD2_FEATURES]
    f = [c for c in FLAG_COLS if c in HEAD2_FEATURES]
    co = [c for c in CONTINUOUS_SENT if c in HEAD2_FEATURES]
    tr_b = sc2mm.fit_transform(train[b].values); vl_b = sc2mm.transform(val[b].values); te_b = sc2mm.transform(test[b].values)
    if co:
        tr_c = sc2std.fit_transform(train[co].values); vl_c = sc2std.transform(val[co].values); te_c = sc2std.transform(test[co].values)
    else:
        tr_c = np.zeros((len(train),0)); vl_c = np.zeros((len(val),0)); te_c = np.zeros((len(test),0))
    tr_h2 = np.hstack([tr_b, tr_c, train[f].values]); vl_h2 = np.hstack([vl_b, vl_c, val[f].values]); te_h2 = np.hstack([te_b, te_c, test[f].values])
    return tr_h1, tr_h2, train['target'].values, vl_h1, vl_h2, val['target'].values, te_h1, te_h2, test['target'].values, sc1, sc2mm, sc2std

def build_seq(h1, h2, y, sl):
    X1, X2, yy = [], [], []
    for i in range(sl, len(h1)):
        X1.append(h1[i-sl:i]); X2.append(h2[i-sl:i]); yy.append(y[i])
    return np.array(X1), np.array(X2), np.array(yy)

def train_config(cfg, tr_h1, tr_h2, y_tr, vl_h1, vl_h2, y_vl, te_h1, te_h2, y_te):
    sl = cfg['seq_len']; bs = cfg['batch_size']
    X1t, X2t, yt = build_seq(tr_h1, tr_h2, y_tr, sl)
    X1v, X2v, yv = build_seq(vl_h1, vl_h2, y_vl, sl)
    X1e, X2e, ye = build_seq(te_h1, te_h2, y_te, sl)

    pw = np.sqrt((1 - yt.mean()) / (yt.mean() + 1e-8))
    def mk(X1, X2, y, bs, sh=False):
        return DataLoader(TensorDataset(torch.tensor(X1,dtype=torch.float32), torch.tensor(X2,dtype=torch.float32), torch.tensor(y,dtype=torch.float32)), batch_size=bs, shuffle=sh)

    tl, vl, el = mk(X1t,X2t,yt,bs,True), mk(X1v,X2v,yv,bs), mk(X1e,X2e,ye,bs)
    model = DualHeadTransformer(X1t.shape[2], X2t.shape[2], cfg['d_model'], cfg['nhead'], cfg['num_layers'], cfg['dropout'])
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw],dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)

    best_vl, pat, best_st = float('inf'), 0, None
    t_losses, v_losses = [], []
    for ep in range(1, 201):
        model.train(); el_ = 0
        for x1,x2,yb in tl:
            opt.zero_grad(); loss = crit(model(x1,x2).squeeze(-1), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); el_ += loss.item()
        t_losses.append(el_/len(tl))
        model.eval(); vl_ = 0
        with torch.no_grad():
            for x1,x2,yb in vl: vl_ += crit(model(x1,x2).squeeze(-1), yb).item()
        v_losses.append(vl_/len(vl)); sched.step(v_losses[-1])
        if v_losses[-1] < best_vl:
            best_vl = v_losses[-1]; best_st = {k:v.clone() for k,v in model.state_dict().items()}; pat = 0
        else:
            pat += 1
            if pat >= 25: break

    model.load_state_dict(best_st)
    model.eval()

    # Calibrate threshold
    probs_v, true_v = [], []
    with torch.no_grad():
        for x1,x2,yb in vl:
            probs_v.extend(torch.sigmoid(model(x1,x2)).numpy().flatten()); true_v.extend(yb.numpy().flatten())
    probs_v, true_v = np.array(probs_v), np.array(true_v)
    best_t, best_f1v = 0.5, 0
    for t in np.arange(0.10, 0.90, 0.01):
        f = f1_score(true_v, (probs_v>=t).astype(int), zero_division=0)
        if f > best_f1v: best_f1v = f; best_t = round(t,2)

    # Test
    probs_e, true_e = [], []
    with torch.no_grad():
        for x1,x2,yb in el:
            probs_e.extend(torch.sigmoid(model(x1,x2)).numpy().flatten()); true_e.extend(yb.numpy().flatten())
    probs_e, true_e = np.array(probs_e), np.array(true_e)
    preds_e = (probs_e >= best_t).astype(int)
    acc = accuracy_score(true_e, preds_e)
    f1 = f1_score(true_e, preds_e, zero_division=0)

    return {
        'acc': acc, 'f1': f1, 'val_f1': best_f1v, 'threshold': best_t,
        'epochs': ep, 'val_loss': best_vl,
        'model': model, 'state': best_st,
        'train_losses': t_losses, 'val_losses': v_losses,
        'report': classification_report(true_e, preds_e, target_names=['Down','Up']),
        'report_dict': classification_report(true_e, preds_e, target_names=['Down','Up'], output_dict=True),
        'scalers': None,  # filled in main
    }

def main():
    print("=" * 70)
    print("  COMPARING TOP 3 CONFIGS ON CORRECTED DATA")
    print("=" * 70)

    df = load_data()
    tr_h1, tr_h2, y_tr, vl_h1, vl_h2, y_vl, te_h1, te_h2, y_te, sc1, sc2mm, sc2std = split_scale(df)
    print(f"  Data: {len(df)} rows | Train: {len(y_tr)} | Val: {len(y_vl)} | Test: {len(y_te)}\n")

    results = []
    for cfg in CONFIGS:
        tag = f"{cfg['name']} (d={cfg['d_model']}, nh={cfg['nhead']}, nl={cfg['num_layers']})"
        print(f"  {'─'*60}")
        print(f"  Training {tag}...")
        r = train_config(cfg, tr_h1, tr_h2, y_tr, vl_h1, vl_h2, y_vl, te_h1, te_h2, y_te)
        r['config'] = cfg
        r['scalers'] = (sc1, sc2mm, sc2std)
        results.append(r)
        print(f"    Val F1: {r['val_f1']:.4f} | Test Acc: {r['acc']:.4f} | Test F1: {r['f1']:.4f} | t={r['threshold']} | Epochs: {r['epochs']}")

    # ── Comparison table ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Config':<35} {'Val F1':>8} {'Test Acc':>10} {'Test F1':>9} {'Thresh':>8} {'Epochs':>8}")
    print(f"  {'─'*35} {'─'*8} {'─'*10} {'─'*9} {'─'*8} {'─'*8}")

    best_idx, best_f1_score = -1, 0
    for i, r in enumerate(results):
        c = r['config']
        tag = f"{c['name']} d={c['d_model']} nh={c['nhead']} nl={c['num_layers']}"
        marker = ""
        
        # Balance guard check
        rep = r['report_dict']
        recall_down = rep['Down']['recall']
        recall_up = rep['Up']['recall']
        is_balanced = (recall_down >= 0.40) and (recall_up >= 0.40)
        
        if is_balanced and r['f1'] > best_f1_score:
            best_f1_score = r['f1']
            best_idx = i
            marker = " 🏆"
        elif not is_balanced:
            marker = " ⚠️ (Unbalanced)"

        print(f"  {tag:<35} {r['val_f1']:>8.4f} {r['acc']:>10.4f} {r['f1']:>9.4f} {r['threshold']:>8} {r['epochs']:>8} | R_Down: {recall_down:.2f} R_Up: {recall_up:.2f}{marker}")

    if best_idx == -1:
        print("\n  ⚠️ WARNING: No balanced models found. Falling back to highest raw F1 regardless of balance.")
        best_idx = int(np.argmax([r['f1'] for r in results]))

    best = results[best_idx]
    bc = best['config']
    print(f"\n  🏆 WINNER: {bc['name']} (d={bc['d_model']}, nh={bc['nhead']}, nl={bc['num_layers']})")
    print(f"     Test Accuracy: {best['acc']:.4f}")
    print(f"     Test F1:       {best['f1']:.4f}")
    print(f"\n  Classification Report:")
    print(best['report'])

    # ── Save winner as production model ───────────────────────────────
    torch.save(best['state'], os.path.join(MODEL_DIR, 'model3_best.pt'))
    sc1, sc2mm, sc2std = best['scalers']
    joblib.dump(sc1,    os.path.join(MODEL_DIR, 'scaler_head1.pkl'))
    joblib.dump(sc2mm,  os.path.join(MODEL_DIR, 'scaler_head2_minmax.pkl'))
    joblib.dump(sc2std, os.path.join(MODEL_DIR, 'scaler_head2_std.pkl'))

    metrics = {
        'accuracy': best['acc'], 'f1': best['f1'], 'best_threshold': best['threshold'],
        'best_val_loss': best['val_loss'], 'val_f1': best['val_f1'], 'epochs_trained': best['epochs'],
        'hyperparameters': {k: bc[k] for k in ['d_model','nhead','num_layers','dropout','lr','batch_size','seq_len']},
        'grid_search_rank': bc['name'], 'features': HEAD1_FEATURES + HEAD2_FEATURES,
    }
    with open(os.path.join(MODEL_DIR, 'metrics_finetuned.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(MODEL_DIR, 'feature_sets.json'), 'w') as f:
        json.dump({'onchain_lasso': HEAD1_FEATURES, 'final_h1': HEAD1_FEATURES, 'sentiment': HEAD2_FEATURES}, f, indent=2)

    # ── Comparison plot ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, r in enumerate(results):
        ax = axes[i]
        c = r['config']
        ax.plot(r['train_losses'], label='Train', color='#2196F3', linewidth=2)
        ax.plot(r['val_losses'], label='Val', color='#FF5722', linewidth=2)
        ax.set_title(f"{c['name']} (d={c['d_model']}, nh={c['nhead']}, nl={c['num_layers']})\n"
                     f"Acc={r['acc']:.3f} F1={r['f1']:.3f}", fontsize=10)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend(fontsize=8)
        if i == best_idx: ax.set_facecolor('#f0fff0')
    fig.suptitle('Top 3 Configurations — Training Curves (Corrected Data)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'top3_comparison.png'), dpi=150)
    plt.close(fig)

    print(f"\n  ✅ Winner saved to models/")
    print(f"  📊 Comparison plot saved to plots/top3_comparison.png")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
