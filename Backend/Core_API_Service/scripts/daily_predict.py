#!/usr/bin/env python3
"""
NeuralEdge Daily Predictor
==========================
One command. Fresh data. Tomorrow's prediction.

Usage:
    cd Project/Backend/Core_API_Service
    python scripts/daily_predict.py

Or from anywhere:
    /path/to/Project/.venv/bin/python3 /path/to/Core_API_Service/scripts/daily_predict.py
"""
import os, sys, json, warnings, requests
import numpy as np, pandas as pd, torch, torch.nn as nn, joblib
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, 'data')
MODEL_DIR  = os.path.join(ROOT_DIR, '..', 'NeuralEdge', 'models')
TODAY      = datetime.now().strftime('%Y-%m-%d')
TOMORROW   = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

# ── Feature config ────────────────────────────────────────────────────
H1_FEATURES = [
    'AdrActCnt','AdrBalCnt','TxCnt','HashRate','BlkCnt','SplyExNtv',
    'FlowInExNtv','CapMVRVCur','AdrActCnt_growth7d','AdrActCnt_growth30d',
    'CapMVRVCur_growth7d','CapMVRVCur_growth30d','momentum_7d',
]
H2_FEATURES = [
    'google_trends','gt_ma7','gt_ma30','gt_change7','gt_momentum',
    'fear_greed','fg_ma7','fg_ma14','fg_change','fg_change7',
    'fg_extreme_fear','fg_extreme_greed',
]
BOUNDED = ['fear_greed','fg_ma7','fg_ma14','google_trends','gt_ma7','gt_ma30']
FLAGS   = ['fg_extreme_fear','fg_extreme_greed']
CONT    = [c for c in H2_FEATURES if c not in BOUNDED + FLAGS]


def get_last_date(csv_name):
    df = pd.read_csv(os.path.join(DATA_DIR, csv_name))
    return pd.to_datetime(df['Date']).max()


# ══════════════════════════════════════════════════════════════════════
#  STEP 1 — DATA UPDATE
# ══════════════════════════════════════════════════════════════════════

def update_ohlcv():
    try:
        import yfinance as yf
    except ImportError:
        print("    ⚠️  yfinance missing — pip install yfinance"); return False

    last = get_last_date('clean_ohlcv.csv')
    start = (last + timedelta(days=1)).strftime('%Y-%m-%d')
    if start > TODAY:
        print("    ✅ OHLCV already current"); return True

    btc = yf.download('BTC-USD', start=start, end=TOMORROW, progress=False)
    if btc.empty:
        print("    ⚠️  No new OHLCV rows (market closed?)"); return True
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    btc = btc.reset_index()
    if 'index' in btc.columns:
        btc = btc.rename(columns={'index': 'Date'})
    cols = ['Date','Open','High','Low','Close','Volume']
    btc = btc[[c for c in cols if c in btc.columns]]
    btc['Date'] = pd.to_datetime(btc['Date']).dt.normalize().dt.strftime('%Y-%m-%d')

    existing = pd.read_csv(os.path.join(DATA_DIR, 'clean_ohlcv.csv'))
    combined = pd.concat([existing, btc], ignore_index=True)
    combined['Date'] = pd.to_datetime(combined['Date'])
    combined = combined.drop_duplicates('Date', keep='last').sort_values('Date').reset_index(drop=True)
    combined['Date'] = combined['Date'].dt.strftime('%Y-%m-%d')
    combined.to_csv(os.path.join(DATA_DIR, 'clean_ohlcv.csv'), index=False)
    print(f"    ✅ OHLCV +{len(btc)} rows → {combined['Date'].iloc[-1]}")
    return True


def update_onchain():
    last = get_last_date('clean_onchain.csv')
    start = (last + timedelta(days=1)).strftime('%Y-%m-%d')
    if start > TODAY:
        print("    ✅ On-chain already current"); return True

    raw_metrics = [
        'AdrActCnt','AdrBalCnt','BlkCnt','CapMVRVCur','CapMrktCurUSD',
        'CapMrktEstUSD','FeeTotNtv','FlowInExNtv','FlowInExUSD',
        'FlowOutExNtv','FlowOutExUSD','HashRate','IssTotNtv','IssTotUSD',
        'ROI1yr','ROI30d','ReferenceRate','ReferenceRateETH',
        'ReferenceRateEUR','ReferenceRateUSD','SplyCur','SplyExNtv',
        'SplyExUSD','SplyExpFut10yr','TxCnt','TxTfrCnt',
    ]
    url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    params = {'assets':'btc','metrics':','.join(raw_metrics),
              'start_time':start,'end_time':TODAY,'frequency':'1d','page_size':10000}
    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    ❌ CoinMetrics error: {e}"); return False

    if 'data' not in data or not data['data']:
        print("    ⚠️  No new on-chain rows"); return True

    new_df = pd.DataFrame(data['data']).rename(columns={'time':'Date'})
    new_df['Date'] = pd.to_datetime(new_df['Date']).dt.tz_localize(None).dt.normalize()
    if 'asset' in new_df.columns: new_df.drop(columns=['asset'], inplace=True)
    for col in new_df.columns:
        if col != 'Date': new_df[col] = pd.to_numeric(new_df[col], errors='coerce')

    # Volume metric
    for col in ['volume_reported_spot_usd_1d','AssetCompletionTime','AssetEODCompletionTime']:
        if col not in new_df.columns: new_df[col] = np.nan
    try:
        vr = requests.get(url, params={**params, 'metrics':'volume_reported_spot_usd_1d'}, timeout=30)
        if vr.ok:
            vd = vr.json()
            if 'data' in vd and vd['data']:
                vdf = pd.DataFrame(vd['data']).rename(columns={'time':'Date'})
                vdf['Date'] = pd.to_datetime(vdf['Date']).dt.tz_localize(None).dt.normalize()
                if 'asset' in vdf.columns: vdf.drop(columns=['asset'], inplace=True)
                vdf['volume_reported_spot_usd_1d'] = pd.to_numeric(vdf['volume_reported_spot_usd_1d'], errors='coerce')
                new_df = new_df.drop(columns=['volume_reported_spot_usd_1d'], errors='ignore')
                new_df = new_df.merge(vdf[['Date','volume_reported_spot_usd_1d']], on='Date', how='left')
    except Exception:
        pass

    # Compute technicals from full OHLCV
    ohlcv = pd.read_csv(os.path.join(DATA_DIR, 'clean_ohlcv.csv'), parse_dates=['Date'])
    ohlcv['Date'] = ohlcv['Date'].dt.normalize()
    full = ohlcv.sort_values('Date').reset_index(drop=True)
    c, h, l, v = full['Close'], full['High'], full['Low'], full['Volume']

    tech = pd.DataFrame({'Date': full['Date']})
    for w in [7,14,21,50,100,200]: tech[f'SMA_{w}'] = c.rolling(w).mean()
    for s in [12,26,50]: tech[f'EMA_{s}'] = c.ewm(span=s).mean()
    tech['MACD'] = tech['EMA_12'] - tech['EMA_26']
    tech['MACD_Signal'] = tech['MACD'].ewm(span=9).mean()
    tech['MACD_Hist'] = tech['MACD'] - tech['MACD_Signal']
    delta = c.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    for p in [14,21]:
        rs = gain.rolling(p).mean() / (loss.rolling(p).mean() + 1e-10)
        tech[f'RSI_{p}'] = 100 - (100 / (1 + rs))
    sma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
    tech['BB_upper'] = sma20 + 2*std20; tech['BB_lower'] = sma20 - 2*std20
    tech['BB_width'] = (tech['BB_upper'] - tech['BB_lower']) / (sma20 + 1e-10)
    tech['BB_pct'] = (c - tech['BB_lower']) / (tech['BB_upper'] - tech['BB_lower'] + 1e-10)
    for d in [1,3,5,7,10,14,21,30]: tech[f'return_{d}d'] = c.pct_change(d)
    for d in [7,14,21,30,60]: tech[f'vol_{d}d'] = c.pct_change().rolling(d).std()
    tech['price_sma50_ratio'] = c / (tech['SMA_50'] + 1e-10)
    tech['price_sma200_ratio'] = c / (tech['SMA_200'] + 1e-10)
    tech['price_sma21_ratio'] = c / (tech['SMA_21'] + 1e-10)
    tech['sma7_sma21_cross'] = (tech['SMA_7'] > tech['SMA_21']).astype(float)
    tech['sma21_sma50_cross'] = (tech['SMA_21'] > tech['SMA_50']).astype(float)
    tech['sma50_sma200_cross'] = (tech['SMA_50'] > tech['SMA_200']).astype(float)
    tech['vol_sma7'] = v.rolling(7).mean(); tech['vol_sma30'] = v.rolling(30).mean()
    tech['vol_ratio'] = tech['vol_sma7'] / (tech['vol_sma30'] + 1e-10)
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    tech['OBV'] = obv; tech['OBV_sma7'] = obv.rolling(7).mean()
    l14 = l.rolling(14).min(); h14 = h.rolling(14).max()
    tech['Stoch_K'] = 100 * (c - l14) / (h14 - l14 + 1e-10)
    tech['Stoch_D'] = tech['Stoch_K'].rolling(3).mean()
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    tech['TR'] = tr; tech['ATR'] = tr.rolling(14).mean()
    for p in [5,10,20]: tech[f'ROC_{p}'] = c.pct_change(p) * 100

    new_dates = new_df['Date'].unique()
    tech_new = tech[tech['Date'].isin(new_dates)].copy()
    final = tech_new.merge(
        new_df.drop(columns=[c for c in tech_new.columns if c != 'Date'], errors='ignore'),
        on='Date', how='inner')

    # Growth features
    existing = pd.read_csv(os.path.join(DATA_DIR, 'clean_onchain.csv'), parse_dates=['Date'])
    existing['Date'] = existing['Date'].dt.normalize()
    for metric in ['AdrActCnt','TxCnt','HashRate','CapMVRVCur']:
        for p in [7,30]:
            col = f'{metric}_growth{p}d'
            if metric in new_df.columns:
                hist = existing[['Date', metric]].copy()
                combo = pd.concat([hist, new_df[['Date', metric]]]).drop_duplicates('Date').sort_values('Date')
                combo[col] = combo[metric].pct_change(p)
                final = final.merge(combo[combo['Date'].isin(new_dates)][['Date', col]], on='Date', how='left')

    for col in existing.columns:
        if col not in final.columns: final[col] = np.nan
    final = final[existing.columns]
    final['Date'] = final['Date'].dt.strftime('%Y-%m-%d')
    existing['Date'] = existing['Date'].dt.strftime('%Y-%m-%d')
    combined = pd.concat([existing, final], ignore_index=True)
    combined['Date'] = pd.to_datetime(combined['Date'])
    combined = combined.drop_duplicates('Date', keep='last').sort_values('Date').reset_index(drop=True)
    combined['Date'] = combined['Date'].dt.strftime('%Y-%m-%d')
    combined.to_csv(os.path.join(DATA_DIR, 'clean_onchain.csv'), index=False)
    print(f"    ✅ On-chain +{len(final)} rows → {combined['Date'].iloc[-1]}")
    return True


def update_sentiment():
    last = get_last_date('clean_sentiment.csv')
    days_needed = (datetime.now() - last).days + 1
    if days_needed <= 0:
        print("    ✅ Sentiment already current"); return True

    try:
        resp = requests.get(f"https://api.alternative.me/fng/?limit={days_needed+30}&format=json", timeout=30)
        resp.raise_for_status(); data = resp.json()
    except Exception as e:
        print(f"    ❌ Fear & Greed error: {e}"); return False
    if 'data' not in data:
        print("    ❌ No sentiment data"); return False

    records = [{'Date': datetime.fromtimestamp(int(e['timestamp'])).strftime('%Y-%m-%d'),
                'fear_greed': int(e['value'])} for e in data['data']]
    new_df = pd.DataFrame(records)
    new_df['Date'] = pd.to_datetime(new_df['Date'])

    existing = pd.read_csv(os.path.join(DATA_DIR, 'clean_sentiment.csv'), parse_dates=['Date'])
    combined = pd.concat([existing[['Date','fear_greed']], new_df], ignore_index=True)
    combined = combined.drop_duplicates('Date', keep='last').sort_values('Date').reset_index(drop=True)
    combined['fg_ma7'] = combined['fear_greed'].rolling(7).mean()
    combined['fg_ma14'] = combined['fear_greed'].rolling(14).mean()
    combined['fg_change'] = combined['fear_greed'].diff()
    combined['fg_change7'] = combined['fear_greed'].diff(7)
    combined['fg_extreme_fear'] = (combined['fear_greed'] <= 20).astype(int)
    combined['fg_extreme_greed'] = (combined['fear_greed'] >= 80).astype(int)
    combined['Date'] = combined['Date'].dt.strftime('%Y-%m-%d')
    combined.to_csv(os.path.join(DATA_DIR, 'clean_sentiment.csv'), index=False)
    added = len(combined) - len(existing)
    print(f"    ✅ Sentiment +{added} rows → {combined['Date'].iloc[-1]}")
    return True


def update_google():
    last = get_last_date('clean_google.csv')
    if last.date() >= datetime.now().date():
        print("    ✅ Google Trends already current"); return True
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("    ⚠️  pytrends missing — pip install pytrends"); return False
    try:
        pt = TrendReq(hl='en-US', tz=0)
        start = (last - timedelta(days=35)).strftime('%Y-%m-%d')
        pt.build_payload(['Bitcoin'], cat=0, timeframe=f'{start} {TODAY}', geo='', gprop='')
        trends = pt.interest_over_time()
    except Exception as e:
        print(f"    ❌ Google Trends error: {e}"); return False
    if trends.empty:
        print("    ❌ No Google Trends data"); return False

    trends = trends.reset_index().rename(columns={'date':'Date','Bitcoin':'google_trends'})
    if 'isPartial' in trends.columns: trends.drop(columns=['isPartial'], inplace=True)
    trends['Date'] = pd.to_datetime(trends['Date']).dt.normalize()

    existing = pd.read_csv(os.path.join(DATA_DIR, 'clean_google.csv'), parse_dates=['Date'])
    existing['Date'] = existing['Date'].dt.normalize()
    combined = pd.concat([existing[['Date','google_trends']], trends[['Date','google_trends']]], ignore_index=True)
    combined = combined.drop_duplicates('Date', keep='last').sort_values('Date').reset_index(drop=True)
    combined['gt_ma7'] = combined['google_trends'].rolling(7).mean()
    combined['gt_ma30'] = combined['google_trends'].rolling(30).mean()
    combined['gt_change7'] = combined['google_trends'].pct_change(7)
    combined['gt_momentum'] = combined['google_trends'] - combined['gt_ma30']
    combined['Date'] = combined['Date'].dt.strftime('%Y-%m-%d')
    combined.to_csv(os.path.join(DATA_DIR, 'clean_google.csv'), index=False)
    added = len(combined) - len(existing)
    print(f"    ✅ Google Trends +{added} rows → {combined['Date'].iloc[-1]}")
    return True


# ══════════════════════════════════════════════════════════════════════
#  STEP 2 — PREDICTION
# ══════════════════════════════════════════════════════════════════════

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


def predict():
    print("\n  📊 Loading full historical data...")
    onch_cols = ['Date'] + [c for c in H1_FEATURES if c != 'momentum_7d']
    ohlcv = pd.read_csv(os.path.join(DATA_DIR, 'clean_ohlcv.csv'), parse_dates=['Date'])
    onch  = pd.read_csv(os.path.join(DATA_DIR, 'clean_onchain.csv'), usecols=onch_cols, parse_dates=['Date'])
    sent  = pd.read_csv(os.path.join(DATA_DIR, 'clean_sentiment.csv'), parse_dates=['Date'])
    goog  = pd.read_csv(os.path.join(DATA_DIR, 'clean_google.csv'), parse_dates=['Date'])

    for f in [ohlcv, onch, sent, goog]: f['Date'] = f['Date'].dt.normalize()
    df = ohlcv.merge(onch, on='Date', how='left').merge(sent, on='Date', how='left').merge(goog, on='Date', how='left')
    df = df.sort_values('Date').ffill().bfill()
    df['momentum_7d'] = df['Close'].pct_change(7)

    latest_date = df['Date'].iloc[-1].date()
    latest_close = df['Close'].iloc[-1]

    print(f"  📅 Latest data:  {latest_date}")
    print(f"  💰 BTC Close:    ${latest_close:,.2f}")

    print("  🧠 Loading Challenger model...")
    model = DualHeadTransformer(13, 12)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'model3_best.pt'), weights_only=True))
    model.eval()

    s1  = joblib.load(os.path.join(MODEL_DIR, 'scaler_head1.pkl'))
    smm = joblib.load(os.path.join(MODEL_DIR, 'scaler_head2_minmax.pkl'))
    ss  = joblib.load(os.path.join(MODEL_DIR, 'scaler_head2_std.pkl'))
    with open(os.path.join(MODEL_DIR, 'metrics_finetuned.json')) as f:
        threshold = json.load(f).get('threshold', 0.65)

    recent = df.tail(7).copy()
    window_start = recent['Date'].iloc[0].date()
    window_end   = recent['Date'].iloc[-1].date()

    X1 = torch.FloatTensor(s1.transform(recent[H1_FEATURES].values)).unsqueeze(0)
    X2 = torch.FloatTensor(np.hstack([
        smm.transform(recent[BOUNDED].values),
        ss.transform(recent[CONT].values),
        recent[FLAGS].values,
    ])).unsqueeze(0)

    with torch.no_grad():
        prob = torch.sigmoid(model(X1, X2)).item()
    pred = 1 if prob >= threshold else 0
    dist = abs(prob - threshold)
    conf = 'HIGH' if dist > 0.15 else 'MEDIUM' if dist > 0.07 else 'LOW'
    direction = '📈 UP' if pred == 1 else '📉 DOWN'

    return latest_date, latest_close, window_start, window_end, direction, prob, threshold, conf


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          NeuralEdge — Daily Prediction Engine           ║")
    print("║          Dual-Head Transformer (GS Run 82)              ║")
    print(f"║          {TODAY}                                   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Step 1: Update data
    print("\n  ⬇️  STEP 1: Updating data sources...")
    results = {}
    results['OHLCV']     = update_ohlcv()
    results['On-Chain']  = update_onchain()
    results['Sentiment'] = update_sentiment()
    results['Google']    = update_google()

    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"\n  ⚠️  Warning: {', '.join(failed)} failed to update.")
        print("  Prediction will use the most recent available data.\n")

    # Step 2: Predict
    print("\n  🔮 STEP 2: Running prediction...")
    latest_date, close, w_start, w_end, direction, prob, thresh, conf = predict()

    # Output
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print(f"║  PREDICTION FOR: {TOMORROW}                          ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Direction  :  {direction:<40s} ║")
    print(f"║  Probability:  {prob:.4f} ({prob:.2%}){' '*(30-len(f'{prob:.4f} ({prob:.2%})'))}║")
    print(f"║  Threshold  :  {thresh:<40.4f} ║")
    print(f"║  Confidence :  {conf:<40s} ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  BTC Close  :  ${close:>10,.2f}{' '*28}║")
    print(f"║  Data Date  :  {str(latest_date):<40s} ║")
    print(f"║  Window     :  {str(w_start)} → {str(w_end)}{' '*(25-len(str(w_end)))}║")
    print(f"║  Model      :  Challenger (GS Run 82){' '*18}║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


if __name__ == '__main__':
    main()
