import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta

# CoinGecko: Daily OHLCV (free, no key, 10+ yrs)
def fetch_coingecko_daily(days=3650):  # ~10 yrs BTC
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range?vs_currency=usd&days={days}"
    data = requests.get(url).json()
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.normalize()
    df.set_index('date', inplace=True)
    df['volume'] = [v[1] for v in data['total_volumes']]
    df['market_cap'] = [m[1] for m in data['market_caps']]
    return df[['price', 'volume', 'market_cap']]

prices = fetch_coingecko_daily()

# Glassnode: Manual CSV or API (get free key: studio.glassnode.com)
# Example metrics (download CSV from Glassnode, load here)
# glassnode_csv = pd.read_csv('glassnode_active_addresses.csv')  # e.g.
# glassnode_csv['date'] = pd.to_datetime(glassnode_csv['t']).dt.normalize()
# glassnode_csv.set_index('date', inplace=True)
# prices = prices.join(glassnode_csv['v'].rename('active_addresses'))

# CoinMetrics: Free CSV download (community.coinmetrics.io → btc.csv)
# cm_df = pd.read_csv('cm_btc.csv', parse_dates=['date']).set_index('date')

# Align + basic features (add Glassnode/CM CSVs)
btc_df = prices.copy()
btc_df['return'] = btc_df['price'].pct_change()
btc_df['vol_7d'] = btc_df['return'].rolling(7).std()
btc_df.dropna(inplace=True)

# Save model-ready DF (2017+, daily)
btc_df.to_csv('btc_daily_features.csv')
print(btc_df.tail(10))  # Sample output
print(f"Shape: {btc_df.shape} | Columns: {list(btc_df.columns)}")
