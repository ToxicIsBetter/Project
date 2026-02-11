import pandas as pd
import requests

# Binance: Reliable daily Klines (OHLCV, no rate limit)
url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=2000"  # Last ~5.5 yrs
data = requests.get(url).json()

df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'tbbav', 'tbqav', 'ignore'])
df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
df = df[['date', 'open', 'high', 'low', 'close', 'volume']].astype({'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'})
df.set_index('date', inplace=True)
df.rename(columns={'close': 'price'}, inplace=True)

df['return'] = df['price'].pct_change()
df['vol_7d'] = df['return'].rolling(7).std()
df.dropna(inplace=True)

print("✅ FULL BTC DATA (shape:", df.shape, ")")
print(df.tail())
df.to_csv('btc_binance_daily.csv')
