"""Prepare data files for Model3 training"""
import pandas as pd

print("Preparing data for Model3_OnChain training...\n")

# 1. OHLCV
print("1. OHLCV data...")
df = pd.read_csv('ohlcv_2010_to_now.csv')
if df.columns[0] != 'Date':
    df.columns = ['Date'] + list(df.columns[1:])
df['Date'] = pd.to_datetime(df['Date'])
df.to_csv('ohlcv_ready.csv', index=False)
print(f"   ✅ {len(df)} rows, ends {df['Date'].max().strftime('%Y-%m-%d')}")

# 2. On-chain
print("2. On-chain data...")
df = pd.read_csv('onchain_and_technicals_2010_to_now.csv')
if df.columns[0] != 'Date':
    df.columns = ['Date'] + list(df.columns[1:])
df['Date'] = pd.to_datetime(df['Date'])
df.to_csv('onchain_ready.csv', index=False)
print(f"   ✅ {len(df)} rows")

# 3. Sentiment
print("3. Sentiment data...")
df = pd.read_csv('sentiment_2010_to_now.csv')
if df.columns[0] != 'Date':
    df.columns = ['Date'] + list(df.columns[1:])
df['Date'] = pd.to_datetime(df['Date'])
df.to_csv('sentiment_ready.csv', index=False)
print(f"   ✅ {len(df)} rows")

# 4. Google Trends (already in data folder)
print("4. Google Trends...")
df = pd.read_csv('google_trends_bitcoin.csv')
if df.columns[0] != 'Date':
    df.columns = ['Date'] + list(df.columns[1:])
df['Date'] = pd.to_datetime(df['Date'])
df.to_csv('google_trends_ready.csv', index=False)
print(f"   ✅ {len(df)} rows")

print("\n✅ All data prepared in /Project/data/")
