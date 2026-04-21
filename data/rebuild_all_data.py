"""Rebuild all data files with correct dates and no corruption"""
import pandas as pd
import numpy as np
import requests
from datetime import datetime

print("="*60)
print("REBUILDING ALL DATA FILES")
print("="*60)

# 1. Load original source files
print("\n1. Loading original source files...")
ohlcv_raw = pd.read_csv('ohlcv_2010_to_now.csv')
onchain_raw = pd.read_csv('onchain_and_technicals_2010_to_now.csv')
sentiment_raw = pd.read_csv('sentiment_2010_to_now.csv')

# Fix OHLCV
print("   Fixing OHLCV...")
if ohlcv_raw.columns[0] == 'Date' or ohlcv_raw.columns[0] == 'index':
    ohlcv_raw.columns = ['Date'] + list(ohlcv_raw.columns[1:])
else:
    ohlcv_raw.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
ohlcv_raw['Date'] = pd.to_datetime(ohlcv_raw['Date'])
ohlcv_clean = ohlcv_raw[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
ohlcv_clean.to_csv('clean_ohlcv.csv', index=False)
print(f"   ✅ OHLCV: {len(ohlcv_clean)} rows")

# Fix On-chain
print("   Fixing On-chain...")
if onchain_raw.columns[0] != 'Date':
    onchain_raw.columns = ['Date'] + list(onchain_raw.columns[1:])
onchain_raw['Date'] = pd.to_datetime(onchain_raw['Date'])
onchain_clean = onchain_raw.copy()
onchain_clean.to_csv('clean_onchain.csv', index=False)
print(f"   ✅ On-chain: {len(onchain_clean)} rows")

# Fetch fresh sentiment from API
print("   Fetching fresh sentiment from Alternative.me...")
try:
    url = "https://api.alternative.me/fng/?limit=600"
    resp = requests.get(url, timeout=15).json()
    if 'data' in resp:
        data = resp['data']
        sentiment_df = pd.DataFrame(data)
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['timestamp'].astype(int), unit='s')
        sentiment_df['fg_ma7'] = sentiment_df['value'].astype(int).rolling(7, min_periods=1).mean()
        sentiment_df['fg_ma14'] = sentiment_df['value'].astype(int).rolling(14, min_periods=1).mean()
        sentiment_df['fg_change'] = sentiment_df['value'].astype(int).diff()
        sentiment_df['fg_change7'] = sentiment_df['value'].astype(int).diff(7)
        sentiment_df['fg_extreme_fear'] = (sentiment_df['value'].astype(int) < 25).astype(int)
        sentiment_df['fg_extreme_greed'] = (sentiment_df['value'].astype(int) > 75).astype(int)
        sentiment_df = sentiment_df[['Date', 'value', 'fg_ma7', 'fg_ma14', 'fg_change', 'fg_change7', 'fg_extreme_fear', 'fg_extreme_greed']]
        sentiment_df = sentiment_df.rename(columns={'value': 'fear_greed'})
        sentiment_df = sentiment_df.sort_values('Date')
        sentiment_df.to_csv('clean_sentiment.csv', index=False)
        print(f"   ✅ Sentiment: {len(sentiment_df)} rows (fresh from API)")
    else:
        print("   ⚠️ API failed, using backup")
        # Use backup with fixed dates
        if sentiment_raw.columns[0] != 'Date':
            sentiment_raw.columns = ['Date'] + list(sentiment_raw.columns[1:])
        sentiment_raw['Date'] = pd.to_datetime(sentiment_raw['Date'])
        sentiment_raw.to_csv('clean_sentiment.csv', index=False)
        print(f"   ✅ Sentiment (backup): {len(sentiment_raw)} rows")
except Exception as e:
    print(f"   ⚠️ Error: {e}")
    # Fallback
    if sentiment_raw.columns[0] != 'Date':
        sentiment_raw.columns = ['Date'] + list(sentiment_raw.columns[1:])
    sentiment_raw['Date'] = pd.to_datetime(sentiment_raw['Date'])
    sentiment_raw.to_csv('clean_sentiment.csv', index=False)
    print(f"   ✅ Sentiment (fallback): {len(sentiment_raw)} rows")

# Load Google Trends
print("   Loading Google Trends...")
gt = pd.read_csv('google_trends_bitcoin.csv')
if gt.columns[0] != 'Date':
    gt.columns = ['Date'] + list(gt.columns[1:])
gt['Date'] = pd.to_datetime(gt['Date'])
gt.to_csv('clean_google.csv', index=False)
print(f"   ✅ Google Trends: {len(gt)} rows")

print("\n✅ All clean data files created!")
print("   - clean_ohlcv.csv")
print("   - clean_onchain.csv")
print("   - clean_sentiment.csv")
print("   - clean_google.csv")
