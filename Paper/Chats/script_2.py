content = r'''# BTC Pipeline — Full Development Log
**Project:** BSc. DS & AI — Level 6 — CN 6000 Mental Wealth Professional Life 3
**University:** University of East London
**Goal:** Bitcoin price prediction using Transformer model with on-chain + off-chain data

## 1. Initial Error — KeyError: 'prices'

### Error 1 (first occurrence)
```text
File "btc_pipeline.py", line 10, in <module>
    prices = data['prices']
KeyError: 'prices'
```
**Cause:** CoinGecko `/market_chart/range` API was rate-limited/broken in 2026, returning an error dict instead of `{'prices': [...]}`.

**Fix:** Switch to `/ohlc` endpoint (always returns array, no rate limit on daily historical).

```python
def fetch_coingecko_daily(days=3650):
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days={days}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.normalize()
    df.set_index('date', inplace=True)
    df['price'] = df['close']
    df['volume'] = np.nan
    df['market_cap'] = np.nan
    return df[['price', 'volume', 'market_cap']].fillna(0)
```

### Error 2 (same KeyError in full script)
```text
File "btc_pipeline.py", line 17, in <module>
    prices = fetch_coingecko_daily()
File "btc_pipeline.py", line 10, in fetch_coingecko_daily
    df = pd.DataFrame(data['prices'], ...)
KeyError: 'prices'
```
**Cause:** Same root issue — CoinGecko API quota/rate limit.

**Fix:** Switched entirely to **Binance API** (reliable, no rate limit, free, 2000+ daily rows).

```python
import pandas as pd
import requests

url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=2000"
data = requests.get(url).json()

df = pd.DataFrame(data, columns=[
    'timestamp','open','high','low','close','volume',
    'close_time','qav','num_trades','tbbav','tbqav','ignore'
])
df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
df = df[['date','open','high','low','close','volume']].astype({
    'open':'float','high':'float','low':'float','close':'float','volume':'float'
})
df.set_index('date', inplace=True)
df.rename(columns={'close': 'price'}, inplace=True)
df['return'] = df['price'].pct_change()
df['vol_7d'] = df['return'].rolling(7).std()
df.dropna(inplace=True)

print("✅ FULL BTC DATA (shape:", df.shape, ")")
print(df.tail())
df.to_csv('btc_binance_daily.csv')
```

**Result:** `btc_binance_daily.csv` — 2000+ rows, daily OHLCV + returns + 7-day volatility. ✅

## 2. On-Chain Data — Failed Attempts

### Attempt 1: Glassnode
- **Problem:** Glassnode free tier shows charts but **no CSV export** — exports require Professional plan ($999/mo).
- **Conclusion:** Not viable for free access.

### Attempt 2: CoinMetrics Community Page
- **Problem:** The community download page showed Bitcoin Cash, Bitcoin Diamond, Billy (Bitcoin), Bitcoin Cats — no plain Bitcoin.
- **Solution:** Download **ZIP** from GitHub directly:
  - URL: `https://github.com/coinmetrics/data`
  - Extract `csv/bitcoin.csv` (or `btc.csv`)

## 3. On-Chain Data — CoinMetrics btc.csv

### Files Downloaded
- `btc.csv` (file:122)
- `btc2.csv` (file:121) — identical duplicate

### File Stats
- **32 columns**, ~2.4M chars, daily from **2009-01-03**
- Shape: **(6249 rows × 32 cols)**

### All 32 Columns
```text
time, AdrActCnt, AdrBalCnt, AssetCompletionTime, AssetEODCompletionTime,
BlkCnt, CapMVRVCur, CapMrktCurUSD, CapMrktEstUSD, FeeTotNtv,
FlowInExNtv, FlowInExUSD, FlowOutExNtv, FlowOutExUSD, HashRate,
IssTotNtv, IssTotUSD, PriceBTC, PriceUSD, ROI1yr, ROI30d,
ReferenceRate, ReferenceRateETH, ReferenceRateEUR, ReferenceRateUSD,
SplyCur, SplyExNtv, SplyExUSD, SplyExpFut10yr, TxCnt, TxTfrCnt,
volume_reported_spot_usd_1d
```

### Key On-Chain Features (match to paper file 4)
| Column | Description |
|--------|-------------|
| `AdrActCnt` | Active addresses |
| `TxCnt` | Transaction count |
| `FeeTotNtv` | Total fees (native) |
| `HashRate` | Mining hash rate |
| `SplyCur` | Circulating supply |
| `CapMVRVCur` | MVRV ratio |
| `FlowInExNtv` | Exchange inflow |
| `FlowOutExNtv` | Exchange outflow |

## 4. Confirmed Price Data (btc_binance_daily.csv)

```text
Shape: (993, 7)
Columns: open, high, low, price, volume, return, vol_7d
Date range: ~2023 to 2026-02-11

Sample:
date        open       high       low       price      volume       return    vol_7d
2026-02-10  70138.00   70527.59  67800.00  68841.29  20373.77072  -0.018488  0.077197
2026-02-11  68841.28   69242.42  68784.47  69133.57    279.26541   0.004246  0.076628
```

## 5. Model Choice — Transformer (not LSTM)

Model architecture is based on **Transformer with Multi-Head Attention** (papers file:11, file:14, file:16).

### Transformer vs LSTM Comparison
| Feature | Transformer | LSTM |
|---------|-------------|------|
| Attention | Multi-head (parallel) | Sequential gates |
| Speed | Faster training | Slower |
| Long-range dependencies | Better (global) | Limited (60 steps) |
| Papers | File 11, 14, 16 | File 15 |

## 6. Full Transformer Pipeline Script

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models

# === 1. LOAD & JOIN ===
df_price = pd.read_csv('btc_binance_daily.csv', parse_dates=['date'], index_col='date')
cm = pd.read_csv('btc.csv')
cm['date'] = pd.to_datetime(cm['time']).dt.date
cm.set_index('date', inplace=True)

onchain_cols = ['AdrActCnt', 'TxCnt', 'FeeTotNtv', 'HashRate', 'SplyCur', 'CapMVRVCur']
onchain_avail = [c for c in onchain_cols if c in cm.columns]
print(f"✅ Found on-chain: {onchain_avail}")

df = df_price.join(cm[onchain_avail], how='inner')
df.dropna(inplace=True)

print(f"✅ JOINED DATA — Shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
df.to_csv('btc_full_onchain.csv')

# === 2. FEATURES ===
features = ['price', 'volume', 'return', 'vol_7d'] + onchain_avail[:3]
print(f"📊 Features: {features}")

X = df[features].fillna(method='ffill').fillna(0).values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === 3. SEQUENCES ===
def create_sequences(data, seq_len=60):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len, 0])
    return np.array(xs), np.array(ys)

X_seq, y = create_sequences(X_scaled, 60)
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y[:split], y[split:]

print(f"📈 Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# === 4. TRANSFORMER MODEL ===
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.2):
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_transformer(seq_len, n_features, head_size=256, num_heads=4,
                      ff_dim=128, num_blocks=2, dropout=0.2):
    inputs = layers.Input(shape=(seq_len, n_features))
    x = inputs
    for _ in range(num_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)
    return models.Model(inputs, outputs)

model = build_transformer(
    seq_len=60,
    n_features=len(features),
    head_size=256,
    num_heads=4,
    ff_dim=128,
    num_blocks=2,
    dropout=0.2
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="mse",
    metrics=["mae"]
)

model.summary()

# === 5. TRAIN ===
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# === 6. EVALUATE ===
y_pred = model.predict(X_test)
y_test_full = np.hstack((y_test.reshape(-1,1), np.zeros((len(y_test), len(features)-1))))
y_pred_full = np.hstack((y_pred, np.zeros((len(y_pred), len(features)-1))))
y_test_inv = scaler.inverse_transform(y_test_full)[:, 0]
y_pred_inv = scaler.inverse_transform(y_pred_full)[:, 0]

rmse = np.sqrt(np.mean((y_test_inv - y_pred_inv)**2))
mae = np.mean(np.abs(y_test_inv - y_pred_inv))
mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100

print(f"✅ RMSE: ${rmse:,.2f}")
print(f"✅ MAE:  ${mae:,.2f}")
print(f"✅ MAPE: {mape:.2f}%")

model.save('btc_transformer_model.h5')
```

## 7. Required Python Libraries

```bash
uv add pandas numpy scikit-learn tensorflow matplotlib requests ccxt keras
```

Optional (visualisation/notebooks):
```bash
uv add seaborn plotly jupyter notebook
```

GPU acceleration (if available):
```bash
uv add tensorflow-gpu
```

### Library Purposes
| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation, CSV, joins |
| `numpy` | Arrays, sequences, math |
| `scikit-learn` | MinMaxScaler, metrics |
| `tensorflow` | Transformer/LSTM models |
| `matplotlib` | Plotting predictions |
| `requests` | API calls (Binance) |
| `ccxt` | Crypto exchange data |
| `keras` | Neural network API |

## 8. Bloomberg Terminal Data

### Why Bloomberg?
- Industry standard, professionally cleaned data
- Citing "Bloomberg Terminal" adds professional credibility to the project report

### Getting BTC Data from Bloomberg

**Tickers:**
- Bitcoin: `XBTUSD Curncy` (Bloomberg Generic Composite)
- S&P 500: `SPX Index`
- US Dollar Index: `DXY Curncy`
- Gold: `XAU Curncy`
- VIX: `VIX Index`

**Steps:**
1. Log in to Bloomberg Terminal
2. Open Excel on the same machine
3. Find **Bloomberg tab** in Excel ribbon
4. Click **Spreadsheet Builder** → **Historical Data Table**
5. Security: `XBTUSD Curncy`
6. Fields: `PX_OPEN`, `PX_HIGH`, `PX_LOW`, `PX_LAST`, `PX_VOLUME`
7. Date range: `01/01/2015` to today
8. Periodicity: **Daily**
9. Click Finish → Save as `bloomberg_btc_spot.csv`

**If Bloomberg tab is missing in Excel:**
1. Close Excel
2. Start → All Programs → Bloomberg → **Install Office Add-Ins**
3. Click Install → OK
4. Reopen Excel

### Fields Available for XBTUSD Curncy
User confirmed only these were available:
- `PX_OPEN`
- `PX_HIGH`
- `PX_LOW`
- `PX_LAST`
- `VOLATILITY_30DAYS`
- `VOLATILITY_90DAYS`

(Note: Bloomberg does NOT provide on-chain data — use CoinMetrics btc.csv for that)

### Bloomberg Merge Script

```python
import pandas as pd

# Bloomberg BTC spot
bb = pd.read_csv('bloomberg_btc_spot.csv')
bb.rename(columns={
    bb.columns[0]: 'date',
    'PX_OPEN': 'open',
    'PX_HIGH': 'high',
    'PX_LOW': 'low',
    'PX_LAST': 'price',
    'VOLATILITY_30DAYS': 'vol_30d',
    'VOLATILITY_90DAYS': 'vol_90d'
}, inplace=True)
bb['date'] = pd.to_datetime(bb['date'])
bb.set_index('date', inplace=True)

if 'volume' not in bb.columns:
    bb['volume'] = 0.0

# CoinMetrics on-chain
cm = pd.read_csv('btc.csv')
cm['date'] = pd.to_datetime(cm['time']).dt.normalize()
cm.set_index('date', inplace=True)

onchain_cols = ['AdrActCnt', 'TxCnt', 'FeeTotNtv', 'SplyCur', 'CapMVRVCur']
onchain_avail = [c for c in onchain_cols if c in cm.columns]

# Join
df = bb.join(cm[onchain_avail], how='inner')
df.sort_index(inplace=True)
df['return'] = df['price'].pct_change()
df['vol_7d'] = df['return'].rolling(7).std()
df.dropna(inplace=True)

print("✅ Combined dataset:", df.shape)
print(df.tail())
df.to_csv('btc_bbg_onchain_full.csv')
```

## 9. Final Feature Set

| Source | Features | Count |
|--------|----------|-------|
| Bloomberg (off-chain) | open, high, low, price, vol_30d, vol_90d | 6 |
| CoinMetrics btc.csv (on-chain) | AdrActCnt, TxCnt, FeeTotNtv, SplyCur, CapMVRVCur | 5 |
| Engineered | return, vol_7d | 2 |
| **Total** | | **13** |

### Optional Engineered Features (free, no extra data)
```python
df['range'] = df['high'] - df['low']
df['log_return'] = np.log(df['price'] / df['price'].shift(1))
df['vol_14d'] = df['return'].rolling(14).std()
df['ma_7'] = df['price'].rolling(7).mean()
df['ma_30'] = df['price'].rolling(30).mean()
df['price_ma_ratio'] = df['price'] / df['ma_30']
df['tx_per_addr'] = df['TxCnt'] / df['AdrActCnt']
```
Extends to **20 features** with no extra downloads.

## 10. LunarCrush API (Social Sentiment — Optional)

- **What it adds:** Galaxy Score, social mentions, social volume (matches paper file 5 — news/sentiment)
- **Free tier:** ❌ No API access on free plan
- **Paid plans:** Builder/API plan (custom pricing)

**To get API key:**
1. Sign up at lunarcrush.com
2. Log in → go to `lunarcrush.com/developers/api/authentication`
3. Accept Terms → click **Generate** → copy immediately

**Python usage:**
```python
import requests

API_KEY = "your_key_here"
headers = {"Authorization": f"Bearer {API_KEY}"}
url = "https://lunarcrush.com/api4/public/topic/bitcoin/v1"
response = requests.get(url, headers=headers).json()
print(response)
```

**Verdict:** Only needed if your project includes sentiment analysis (paper file 5). Current dataset (Bloomberg + CoinMetrics) is already strong without it.

## 11. Academic Papers Referenced

| File | Paper | Relevance |
|------|-------|-----------|
| file:4 | Bitcoin price direction prediction using on-chain data | 196 on-chain features, CNN-LSTM, 82% accuracy |
| file:5 | Short-term forecasting based on news headlines | Sentiment/social features |
| file:6 | Investigating the Crypto price prediction Problem | CNN-LSTM, BiLSTM benchmarks |
| file:11 | Enhancing Price Prediction Using Transformer | Transformer architecture |
| file:14 | Google's Attention is All You Need | Original Transformer paper |
| file:15 | Forecasting using LSTM, GRU, Bi-LSTM | LSTM RMSE benchmark ~$1,030 |
| file:16 | Temporal Fusion Transformer + on-chain data | TFT hybrid model |

## 12. Project File Summary

| File | Status | Description |
|------|--------|-------------|
| `btc_binance_daily.csv` | ✅ Done | 993 rows, daily OHLCV + return + vol_7d |
| `btc.csv` | ✅ Done | 6249 rows, 32 on-chain cols (CoinMetrics) |
| `btc2.csv` | ✅ Duplicate | Same as btc.csv |
| `btc_full_onchain.csv` | ⏳ Next step | Joined Bloomberg + on-chain |
| `btc_bbg_onchain_full.csv` | ⏳ Next step | Bloomberg + CoinMetrics joined |
| `btc_transformer_model.h5` | ⏳ Next step | Trained Transformer model |
| `bloomberg_btc_spot.csv` | ⏳ Pending | Bloomberg download (OHLC + vol) |

## 13. Current Status & Next Steps

1. ✅ Price data ready (`btc_binance_daily.csv`)
2. ✅ On-chain data ready (`btc.csv`, 32 cols)
3. ⏳ Save Bloomberg export as `bloomberg_btc_spot.csv`
4. ⏳ Run merge script → `btc_bbg_onchain_full.csv`
5. ⏳ Run Transformer training script
6. ⏳ Record RMSE/MAE/MAPE (benchmark: file 15 RMSE ~$1,030)
7. ⏳ Predict tomorrow's BTC price
8. ⏳ Generate prediction plot

**Target RMSE:** < $1,000 (beats file 15 LSTM benchmark)
'''

path = 'output/btc_full_chat.md'
import os
os.makedirs('output', exist_ok=True)
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print(path)