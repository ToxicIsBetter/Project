"""
pipeline.py — BTC Data Pipeline
Loads price + on-chain + sentiment data, engineers features, creates temporal splits.
Output is tailored for the Dual-Head Transformer (Head 1: 25 On-chain/Market, Head 2: 3 Sentiment).
"""
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. LOAD PRICE DATA (yfinance)
# ============================================================
print("📈 Loading BTC price data from yfinance...")
btc = yf.download('BTC-USD', start='2010-01-01', end='2026-03-21', interval='1d')
if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = btc.columns.get_level_values(0)
btc.index = pd.to_datetime(btc.index)
btc.index.name = 'date'
print(f"   Price data: {len(btc)} rows")

# ============================================================
# 2. LOAD ON-CHAIN DATA (CoinMetrics btc.csv)
# ============================================================
print("🔗 Loading on-chain data...")
cm = pd.read_csv(DATA_DIR / 'btc.csv')
cm['date'] = pd.to_datetime(cm['time'])
cm.set_index('date', inplace=True)

# We need these specific columns to build our 25 proxy features
onchain_cols = [
    'TxCnt', 'TxTfrCnt', 'SplyCur', 'AdrBalCnt', # Stationarity proxies
    'ROI1yr', 'ROI30d',                          # Realized
    'CapMVRVCur', 'CapMrktCurUSD', 'IssTotUSD',  # Unrealized
    'AdrActCnt'                                  # Activity
]
onchain_avail = [c for c in onchain_cols if c in cm.columns]

# ============================================================
# 3. LOAD SENTIMENT DATA (Fear & Greed Index)
# ============================================================
print("😱 Fetching Fear & Greed Index...")
try:
    fng_url = "https://api.alternative.me/fng/?limit=0"
    fng_resp = requests.get(fng_url, timeout=15)
    fng_resp.raise_for_status()
    fng_data = fng_resp.json()['data']
    fng_df = pd.DataFrame(fng_data)
    fng_df['date'] = pd.to_datetime(fng_df['timestamp'].astype(int), unit='s')
    fng_df['fear_greed'] = fng_df['value'].astype(int)
    fng_df = fng_df[['date', 'fear_greed']].set_index('date').sort_index()
    fng_df.to_csv(DATA_DIR / 'fear_greed.csv')
    has_sentiment = True
except Exception as e:
    print(f"   ⚠️ Fear & Greed fetch failed: {e}")
    has_sentiment = False

# ============================================================
# 4. JOIN + FEATURE ENGINEERING (MATCHING DUBEY & ENKE 2025)
# ============================================================
print("🔧 Engineering 25 On-Chain + 3 Sentiment features...")

df = btc.join(cm[onchain_avail], how='left')
df[onchain_avail] = df[onchain_avail].ffill()

if has_sentiment:
    df = df.join(fng_df, how='left')
    df['fear_greed'] = df['fear_greed'].ffill().bfill()
else:
    # fallback
    vol = df['Close'].pct_change().rolling(30).std()
    df['fear_greed'] = (100 - (vol / vol.max() * 100)).clip(0, 100)

df['fg_ma7'] = df['fear_greed'].rolling(7).mean()
df['fg_change'] = df['fear_greed'].diff(1)

# Helper: compute Realized Cap from MVRV and Market Cap
df['CapRealUSD'] = df['CapMrktCurUSD'] / df['CapMVRVCur']

# --- CATEGORY A: Stationarity (5 Features) ---
df['TxCnt'] = df['TxCnt']
df['TxTfrCnt'] = df['TxTfrCnt']
df['CapRealUSD_growth'] = df['CapRealUSD'].pct_change(7)  # Proxy for HODL wave movement
df['SplyCur'] = df['SplyCur']
df['AdrBalCnt'] = df['AdrBalCnt'] # Proxy for holding active accounts

# --- CATEGORY B: Realized Value (12 Features) ---
df['PriceOpen'] = df['Open']
df['PriceHigh'] = df['High']
df['PriceLow'] = df['Low']
df['PriceClose'] = df['Close']
df['CapRealUSD'] = df['CapRealUSD']
df['ROI1yr'] = df['ROI1yr']
df['ROI30d'] = df['ROI30d']
df['volatility_30d'] = df['Close'].pct_change().rolling(30).std()
df['volatility_7d'] = df['Close'].pct_change().rolling(7).std()
df['price_momentum_14d'] = df['Close'].pct_change(14)
# SOPR proxy: Price / Realized Price
realized_price = df['CapRealUSD'] / df['SplyCur']
df['price_realized_ratio'] = df['Close'] / realized_price
df['volume_usd'] = df['Volume']

# --- CATEGORY C: Unrealized Value (7 Features) ---
df['CapMVRVCur'] = df['CapMVRVCur']
df['MVRV_Z_Score'] = (df['CapMVRVCur'] - df['CapMVRVCur'].rolling(30).mean()) / df['CapMVRVCur'].rolling(30).std()
df['CapMrktCurUSD'] = df['CapMrktCurUSD']
df['Market_Realized_Delta'] = df['CapMrktCurUSD'] - df['CapRealUSD']
df['Market_Realized_Delta_Pct'] = df['Market_Realized_Delta'] / df['CapMrktCurUSD']
df['IssTotUSD'] = df['IssTotUSD']
df['Unrealized_Value_Momentum'] = df['CapMVRVCur'].diff(7)

# --- CATEGORY D: Activity (1 Feature) ---
df['AdrActCnt'] = df['AdrActCnt']

head_1_features = [
    'TxCnt', 'TxTfrCnt', 'CapRealUSD_growth', 'SplyCur', 'AdrBalCnt', 
    'PriceOpen', 'PriceHigh', 'PriceLow', 'PriceClose', 'CapRealUSD', 
    'ROI1yr', 'ROI30d', 'volatility_30d', 'volatility_7d', 'price_momentum_14d', 
    'price_realized_ratio', 'volume_usd',
    'CapMVRVCur', 'MVRV_Z_Score', 'CapMrktCurUSD', 'Market_Realized_Delta', 
    'Market_Realized_Delta_Pct', 'IssTotUSD', 'Unrealized_Value_Momentum',
    'AdrActCnt'
]

head_2_features = ['fear_greed', 'fg_ma7', 'fg_change']

feature_cols = head_1_features + head_2_features

# ============================================================
# 5. TARGET: next-day direction (1 = up, 0 = down)
# ============================================================
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Drop NaN rows from feature engineering (rolling windows, shifts, etc.)
df.dropna(subset=feature_cols + ['target'], inplace=True)

# Ensure no inf values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=feature_cols + ['target'], inplace=True)

print(f"   Total features: {len(feature_cols)} ({len(head_1_features)} Head 1 + {len(head_2_features)} Head 2)")

# ============================================================
# 6. TEMPORAL SPLITS (no lookahead bias)
# ============================================================
train = df[df.index < '2024-07-01']
val = df[(df.index >= '2024-07-01') & (df.index < '2025-01-01')]
test = df[df.index >= '2025-01-01']

print(f"\n📊 TEMPORAL SPLITS:")
print(f"   Train: {len(train)} rows ({train.index[0].date()} to {train.index[-1].date()})")
print(f"   Val:   {len(val)} rows ({val.index[0].date()} to {val.index[-1].date()})")
print(f"   Test:  {len(test)} rows ({test.index[0].date()} to {test.index[-1].date()})")

# ============================================================
# 7. SAVE
# ============================================================
df.to_csv(DATA_DIR / 'btc_features.csv')
train.to_csv(DATA_DIR / 'btc_train.csv')
val.to_csv(DATA_DIR / 'btc_val.csv')
test.to_csv(DATA_DIR / 'btc_test.csv')

# Save feature split mapping
with open(DATA_DIR / 'feature_cols.txt', 'w') as f:
    f.write('\n'.join(feature_cols))

with open(DATA_DIR / 'head_1_cols.txt', 'w') as f:
    f.write('\n'.join(head_1_features))

with open(DATA_DIR / 'head_2_cols.txt', 'w') as f:
    f.write('\n'.join(head_2_features))

print(f"\n✅ Pipeline complete! Files saved to {DATA_DIR}/")
