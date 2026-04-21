"""Create clean training data from updated source files"""
import pandas as pd
import numpy as np

print("Creating clean training data from updated files...\n")

# 1. Load OHLCV - use original file, fix column names
print("1. Loading OHLCV...")
df = pd.read_csv('ohlcv_2010_to_now.csv')
# First column is Date, rest are OHLCV
cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'open', 'high', 'low', 'close', 'volume']
df.columns = cols[:len(df.columns)]
df['Date'] = pd.to_datetime(df['Date'])
# Keep only uppercase columns (original data)
df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
df.to_csv('train_ohlcv.csv', index=False)
print(f"   ✅ {len(df)} rows")

# 2. Load on-chain - already clean
print("2. Loading on-chain...")
df = pd.read_csv('onchain_and_technicals_2010_to_now.csv')
if df.columns[0] == 'index':
    df.columns = ['Date'] + list(df.columns[1:])
df['Date'] = pd.to_datetime(df['Date'])
df.to_csv('train_onchain.csv', index=False)
print(f"   ✅ {len(df)} rows")

# 3. Load sentiment - use original, fix Date
print("3. Loading sentiment...")
df = pd.read_csv('sentiment_2010_to_now.csv')
# First two columns are index and Date
if df.columns[0] in ['index', 'Unnamed: 0']:
    df = df.drop(df.columns[0], axis=1)
    df.columns = ['Date'] + list(df.columns[1:])
df['Date'] = pd.to_datetime(df['Date'])
df.to_csv('train_sentiment.csv', index=False)
print(f"   ✅ {len(df)} rows")

# 4. Load Google Trends - already in Model3 folder, copy to data
print("4. Loading Google Trends...")
gt = pd.read_csv('google_trends_bitcoin.csv')
if gt.columns[0] != 'Date':
    gt.columns = ['Date'] + list(gt.columns[1:])
gt['Date'] = pd.to_datetime(gt['Date'])
gt.to_csv('train_google.csv', index=False)
print(f"   ✅ {len(gt)} rows")

print("\n✅ All training data files created!")
print("   - train_ohlcv.csv")
print("   - train_onchain.csv") 
print("   - train_sentiment.csv")
print("   - train_google.csv")
