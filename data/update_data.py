"""
Update Bitcoin data to latest date from APIs
"""
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta
import time

print("="*60)
print("UPDATING DATA TO LATEST DATE")
print("="*60)

# ────────────────────────────────────────────────────────────────
# 1. UPDATE OHLCV DATA (Binance API)
# ────────────────────────────────────────────────────────────────
print("\n1. Fetching latest OHLCV data from Binance...")

def fetch_latest_ohlcv():
    """Fetch OHLCV data from Binance"""
    url = "https://api.binance.com/api/v3/klines"
    
    # Get daily candles for BTCUSDT
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1d',
        'limit': 100  # Last 100 days
    }
    
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'tbbav', 'tbqav', 'ignore'
    ])
    
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['Date', 'open', 'high', 'low', 'close', 'volume']]
    
    # Convert to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    df = df.set_index('Date').sort_index()
    
    print(f"   ✅ Fetched {len(df)} rows")
    print(f"   Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    
    return df

ohlcv_latest = fetch_latest_ohlcv()

# Load existing OHLCV data
print("\n   Loading existing OHLCV data...")
ohlcv_existing = pd.read_csv('ohlcv_2010_to_now.csv', parse_dates=['Date'], index_col='Date')
print(f"   Existing data: {len(ohlcv_existing)} rows, ends at {ohlcv_existing.index[-1].strftime('%Y-%m-%d')}")

# Combine: keep existing, append new
ohlcv_combined = pd.concat([ohlcv_existing, ohlcv_latest])
ohlcv_combined = ohlcv_combined[~ohlcv_combined.index.duplicated(keep='last')]
ohlcv_combined = ohlcv_combined.sort_index()

print(f"   Combined data: {len(ohlcv_combined)} rows")

# Save updated OHLCV
ohlcv_combined.reset_index().to_csv('ohlcv_2010_to_now.csv', index=False)
print(f"   ✅ Saved updated OHLCV data")

# ────────────────────────────────────────────────────────────────
# 2. UPDATE SENTIMENT DATA (Alternative.me Fear & Greed)
# ────────────────────────────────────────────────────────────────
print("\n2. Fetching latest Fear & Greed data...")

def fetch_fear_greed(days=120):
    """Fetch Fear & Greed Index from Alternative.me"""
    url = f"https://api.alternative.me/fng/?limit={days}"
    
    try:
        resp = requests.get(url, timeout=15).json()
        
        if "data" not in resp:
            print(f"   ⚠️ API error: {resp}")
            return None
        
        data = resp["data"]
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('Date').sort_index()
        
        # Calculate additional features
        df['fg_ma7'] = df['value'].astype(int).rolling(7, min_periods=1).mean()
        df['fg_ma14'] = df['value'].astype(int).rolling(14, min_periods=1).mean()
        df['fg_change'] = df['value'].astype(int).diff()
        df['fg_change7'] = df['value'].astype(int).diff(7)
        df['fg_extreme_fear'] = (df['value'].astype(int) < 25).astype(int)
        df['fg_extreme_greed'] = (df['value'].astype(int) > 75).astype(int)
        
        df = df[['value', 'fg_ma7', 'fg_ma14', 'fg_change', 'fg_change7', 'fg_extreme_fear', 'fg_extreme_greed']]
        df.columns = ['fear_greed', 'fg_ma7', 'fg_ma14', 'fg_change', 'fg_change7', 'fg_extreme_fear', 'fg_extreme_greed']
        
        print(f"   ✅ Fetched {len(df)} rows")
        print(f"   Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        
        return df
        
    except Exception as e:
        print(f"   ⚠️ Failed: {e}")
        return None

fg_latest = fetch_fear_greed()

if fg_latest is not None:
    # Load existing sentiment
    print("\n   Loading existing sentiment data...")
    sentiment_existing = pd.read_csv('sentiment_2010_to_now.csv', parse_dates=['Date'], index_col='Date')
    print(f"   Existing data: {len(sentiment_existing)} rows")
    
    # Combine
    sentiment_combined = pd.concat([sentiment_existing, fg_latest])
    sentiment_combined = sentiment_combined[~sentiment_combined.index.duplicated(keep='last')]
    sentiment_combined = sentiment_combined.sort_index()
    
    print(f"   Combined data: {len(sentiment_combined)} rows")
    
    # Save updated sentiment
    sentiment_combined.reset_index().to_csv('sentiment_2010_to_now.csv')
    print(f"   ✅ Saved updated sentiment data")

# ────────────────────────────────────────────────────────────────
# 3. UPDATE ON-CHAIN DATA (Merge OHLCV with existing on-chain)
# ────────────────────────────────────────────────────────────────
print("\n3. Updating on-chain data...")

# Load existing on-chain data
onchain_existing = pd.read_csv('onchain_and_technicals_2010_to_now.csv', parse_dates=['Date'], index_col='Date')
print(f"   Existing on-chain: {len(onchain_existing)} rows, ends at {onchain_existing.index[-1].strftime('%Y-%m-%d')}")

# Get the latest on-chain features (forward fill for slow metrics)
# We'll extend the on-chain data with the latest OHLCV and forward-fill on-chain metrics
onchain_extended = onchain_existing.copy()

# Get last known on-chain values
last_onchain = onchain_existing.iloc[-1]

# Create new rows for the OHLCV dates that are beyond existing on-chain
ohlcv_reset = ohlcv_combined.reset_index()
new_dates = ohlcv_reset[ohlcv_reset['Date'] > onchain_existing.index[-1]]

if len(new_dates) > 0:
    print(f"   Adding {len(new_dates)} new days to on-chain data")
    
    # For new dates, create rows with forward-filled on-chain data
    new_rows = []
    for idx, row in new_dates.iterrows():
        new_row = last_onchain.copy()
        new_row.name = row['Date']
        new_rows.append(new_row)
    
    new_rows_df = pd.DataFrame(new_rows)
    onchain_extended = pd.concat([onchain_existing, new_rows_df])
    onchain_extended = onchain_extended.sort_index()
    
    # Update OHLCV columns with actual values
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in onchain_extended.columns:
            onchain_extended.loc[new_dates.index, col] = ohlcv_combined.loc[new_dates.index, col].values
    
    print(f"   ✅ Extended on-chain data to {len(onchain_extended)} rows")
else:
    print(f"   ℹ️ No new on-chain data needed (already up to date)")
    onchain_extended = onchain_existing

# Save updated on-chain
onchain_extended.reset_index().to_csv('onchain_and_technicals_2010_to_now.csv', index=False)
print(f"   ✅ Saved updated on-chain data")

print("\n" + "="*60)
print("UPDATE COMPLETE")
print("="*60)
print(f"\nFinal dataset ends at: {onchain_extended.index[-1].strftime('%Y-%m-%d')}")
print(f"Total rows: {len(onchain_extended):,}")
