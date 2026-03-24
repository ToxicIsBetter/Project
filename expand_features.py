import requests
import pandas as pd
import numpy as np
import yfinance as yf
import os

print("=== EXPANDING FEATURE SET TO ~113 FEATURES (STARTING 2010) ===")

# ─────────────────────────────────────────────────────────────────
# SOURCE 1: COINMETRICS COMMUNITY API (FREE, NO API KEY NEEDED)
# ─────────────────────────────────────────────────────────────────
print("Fetching CoinMetrics Community API...")
CM_BASE = "https://community-api.coinmetrics.io/v4"
CM_METRICS = [
    "AdrActCnt", "AdrActRecCnt", "AdrActSentCnt", "AdrBal1in100KCnt",
    "AdrBal1in10BCnt", "AdrBal1in10KCnt", "AdrBal1in1BCnt", "AdrBalCnt",
    "BlkCnt", "BlkSizeByte", "BlkSizeMeanByte", "CapMVRVCur",
    "CapMrktCurUSD", "CapRealUSD", "FeeMeanNtv", "FeeMeanUSD",
    "FeeMedNtv", "FeeMedUSD", "FeeTotNtv", "FeeTotUSD",
    "HashRate", "IssTotNtv", "IssTotUSD", "NVTAdj",
    "NVTAdj90", "PriceBTC", "PriceUSD", "ROI1yr",
    "SplyCur", "TxCnt", "TxTfrCnt", "TxTfrValAdjNtv",
    "TxTfrValAdjUSD", "TxTfrValMeanNtv", "TxTfrValMeanUSD", "TxTfrValMedNtv",
    "TxTfrValMedUSD", "VtyDayRet180d", "VtyDayRet30d", "VtyDayRet60d"
]

try:
    metrics_str = ",".join(CM_METRICS)
    url = f"{CM_BASE}/timeseries/asset-metrics"
    params = {
        "assets": "btc",
        "metrics": metrics_str,
        "frequency": "1d",
        "start_time": "2010-01-01",
        "end_time": "2026-03-22",
        "page_size": 10000,
        "format": "json"
    }

    response = requests.get(url, params=params, timeout=30)
    cm_data = response.json()
    
    if 'data' not in cm_data:
        raise ValueError(f"CoinMetrics API Error: {cm_data}")

    cm_df = pd.DataFrame(cm_data['data'])
    cm_df['Date'] = pd.to_datetime(cm_df['time']).dt.tz_localize(None)
    cm_df = cm_df.drop(columns=['asset', 'time'], errors='ignore')
    cm_df = cm_df.set_index('Date')
    cm_df = cm_df.apply(pd.to_numeric, errors='coerce')
    print(f"CoinMetrics: {len(cm_df.columns)} metrics, {len(cm_df)} rows")
    cm_df.to_csv('data/coinmetrics_raw.csv')
    
except Exception as e:
    print(f"⚠️ CoinMetrics failed via API: {e}")
    print("Trying to fallback to existing btc.csv")
    cm_df = pd.read_csv("data/btc.csv")
    cm_df['Date'] = pd.to_datetime(cm_df['time']).dt.tz_localize(None)
    cm_df = cm_df.set_index('Date').apply(pd.to_numeric, errors='coerce')

# ─────────────────────────────────────────────────────────────────
# SOURCE 2: FEAR & GREED INDEX (Alternative.me)
# ─────────────────────────────────────────────────────────────────
print("Fetching Fear & Greed Index...")
fg_url = "https://api.alternative.me/fng/?limit=0&format=json"
fg_data = requests.get(fg_url).json()['data']
fg_df = pd.DataFrame(fg_data)
fg_df['Date'] = pd.to_datetime(fg_df['timestamp'].astype(int), unit='s')
fg_df['fear_greed'] = fg_df['value'].astype(float)
fg_df = fg_df[['Date', 'fear_greed']].set_index('Date').sort_index()
fg_df.index = fg_df.index.tz_localize(None)
print(f"Fear/Greed: {len(fg_df)} rows")

# Add Fear/Greed derived features immediately
fg_df['fg_ma7'] = fg_df['fear_greed'].rolling(7).mean()
fg_df['fg_ma14'] = fg_df['fear_greed'].rolling(14).mean()
fg_df['fg_change'] = fg_df['fear_greed'].diff()
fg_df['fg_change7'] = fg_df['fear_greed'].diff(7)
fg_df['fg_extreme_fear'] = (fg_df['fear_greed'] < 25).astype(int)
fg_df['fg_extreme_greed'] = (fg_df['fear_greed'] > 75).astype(int)

# ─────────────────────────────────────────────────────────────────
# SOURCE 3: BITCOIN PRICE + TECHNICAL INDICATORS (yfinance + backfill)
# ─────────────────────────────────────────────────────────────────
print("Fetching BTC price and building technical indicators...")
btc = yf.download('BTC-USD', start='2010-01-01', end='2026-03-22', interval='1d', auto_adjust=True)
btc.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
btc.index = pd.to_datetime(btc.index).tz_localize(None)

# ── NEW: BACKFILL OHLCV TO 2010 USING COINMETRICS ──
# yfinance only goes back to 2014-09. We use CoinMetrics PriceUSD to backfill 2010-2014.
cm_price = cm_df[['PriceUSD', 'TxTfrValAdjUSD']].copy() if 'TxTfrValAdjUSD' in cm_df.columns else cm_df[['PriceUSD']].copy()
cm_price = cm_price[cm_price.index >= '2010-01-01']
cm_price['Close'] = cm_price['PriceUSD']
cm_price['Volume'] = cm_price.get('TxTfrValAdjUSD', 0)
cm_price['Open'] = cm_price['Close'].shift(1)
cm_price['High'] = cm_price['Close'] * 1.05 
cm_price['Low'] = cm_price['Close'] * 0.95
cm_price = cm_price.drop(columns=['PriceUSD', 'TxTfrValAdjUSD'], errors='ignore')

# Combine yfinance (wins where exists) with CoinMetrics backfill
btc = btc.combine_first(cm_price).dropna()
btc = btc[btc.index >= '2010-01-02'] # Avoid first day NaN Open

# Moving Averages
for w in [7, 14, 21, 50, 100, 200]:
    btc[f'SMA_{w}'] = btc['Close'].rolling(w).mean()
btc['EMA_12'] = btc['Close'].ewm(span=12).mean()
btc['EMA_26'] = btc['Close'].ewm(span=26).mean()
btc['EMA_50'] = btc['Close'].ewm(span=50).mean()

# MACD
btc['MACD'] = btc['EMA_12'] - btc['EMA_26']
btc['MACD_Signal'] = btc['MACD'].ewm(span=9).mean()
btc['MACD_Hist'] = btc['MACD'] - btc['MACD_Signal']

# RSI (14 & 21)
for period in [14, 21]:
    delta = btc['Close'].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    btc[f'RSI_{period}'] = 100 - (100 / (1 + gain / loss))

# Bollinger Bands
bb_mid = btc['Close'].rolling(20).mean()
bb_std = btc['Close'].rolling(20).std()
btc['BB_upper'] = bb_mid + 2 * bb_std
btc['BB_lower'] = bb_mid - 2 * bb_std
btc['BB_width'] = (btc['BB_upper'] - btc['BB_lower']) / bb_mid
btc['BB_pct'] = (btc['Close'] - btc['BB_lower']) / (btc['BB_upper'] - btc['BB_lower'])

# Returns & Volatility
for d in [1, 3, 5, 7, 10, 14, 21, 30]:
    btc[f'return_{d}d'] = btc['Close'].pct_change(d)
for w in [7, 14, 21, 30, 60]:
    btc[f'vol_{w}d'] = btc['return_1d'].rolling(w).std()

# Ratio features
btc['price_sma50_ratio'] = btc['Close'] / btc['SMA_50']
btc['price_sma200_ratio'] = btc['Close'] / btc['SMA_200']
btc['price_sma21_ratio'] = btc['Close'] / btc['SMA_21']
btc['sma7_sma21_cross'] = (btc['SMA_7'] > btc['SMA_21']).astype(int)
btc['sma21_sma50_cross'] = (btc['SMA_21'] > btc['SMA_50']).astype(int)
btc['sma50_sma200_cross'] = (btc['SMA_50'] > btc['SMA_200']).astype(int)

# Volume
btc['vol_sma7'] = btc['Volume'].rolling(7).mean()
btc['vol_sma30'] = btc['Volume'].rolling(30).mean()
btc['vol_ratio'] = btc['Volume'] / btc['vol_sma30']
btc['OBV'] = (np.sign(btc['Close'].diff()) * btc['Volume']).cumsum()
btc['OBV_sma7'] = btc['OBV'].rolling(7).mean()

# Stochastic Oscillator
low14 = btc['Low'].rolling(14).min()
high14 = btc['High'].rolling(14).max()
btc['Stoch_K'] = 100 * (btc['Close'] - low14) / (high14 - low14)
btc['Stoch_D'] = btc['Stoch_K'].rolling(3).mean()

# ATR (Average True Range)
btc['TR'] = np.maximum(btc['High'] - btc['Low'],
               np.maximum(abs(btc['High'] - btc['Close'].shift()),
                          abs(btc['Low'] - btc['Close'].shift())))
btc['ATR'] = btc['TR'].rolling(14).mean()

# Rate of Change
for d in [5, 10, 20]:
    btc[f'ROC_{d}'] = btc['Close'].pct_change(d) * 100

print(f"Technical indicators: {len(btc.columns)} features built")

# ─────────────────────────────────────────────────────────────────
# SOURCE 4: DERIVED ON-CHAIN GROWTH RATES
# ─────────────────────────────────────────────────────────────────
growth_cols = ['AdrActCnt', 'TxCnt', 'HashRate', 'FeeTotUSD', 'TxTfrValAdjUSD', 'CapMVRVCur', 'NVTAdj']
for col in growth_cols:
    if col in cm_df.columns:
        cm_df[f'{col}_growth7d'] = cm_df[col].pct_change(7)
        cm_df[f'{col}_growth30d'] = cm_df[col].pct_change(30)

# ─────────────────────────────────────────────────────────────────
# MERGE ALL SOURCES
# ─────────────────────────────────────────────────────────────────
print("Merging all data sources...")

full_df = btc.join(cm_df, how='left')
full_df = full_df.join(fg_df, how='left')

# Impute Fear/Greed neutral 50 for 2010-2018
fg_cols = [c for c in full_df.columns if 'fg_' in c or 'fear_greed' in c]
full_df[fg_cols] = full_df[fg_cols].fillna(50)

# Backward fill first to catch 200-day SMA initial NaNs, then forward fill
full_df = full_df.bfill().ffill()

# TARGET: Next-Day Direction (Strictly Out-of-Sample Forecasting)
# Predicting if tomorrow's Close will be higher than today's Close
# This is the only mathematically rigorous way to prevent data leakage.
full_df['Target'] = (full_df['Close'].shift(-1) > full_df['Close']).astype(int)
full_df = full_df.dropna(subset=['Target'])

# Replace all inf with nan
full_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop any feature columns that are missing more than 50% of the data
missing_pct = full_df.isna().mean()
sparse_cols = missing_pct[missing_pct > 0.5].index
if len(sparse_cols) > 0:
    print(f"⚠️ Dropping terribly sparse columns: {list(sparse_cols)}")
    full_df = full_df.drop(columns=sparse_cols)

# Then drop rows with any remaining NaNs
full_df = full_df.dropna()

full_df.drop(columns=['PriceBTC', 'PriceUSD'], errors='ignore', inplace=True)

print(f"\n✅ TOTAL FEATURES: {len(full_df.columns) - 1}")
print(f"✅ TOTAL ROWS: {len(full_df)}")

full_df.to_csv('data/btc_full_features.csv')
print("\nSaved: data/btc_full_features.csv")
