"""Rebuild sentiment from scratch with real data"""
import pandas as pd
import requests
from datetime import datetime

print("Rebuilding sentiment data from Alternative.me API...\n")

# Fetch from API
url = "https://api.alternative.me/fng/?limit=600"  # Get ~600 days
try:
    resp = requests.get(url, timeout=15).json()
    
    if "data" not in resp:
        print(f"API error: {resp}")
        exit(1)
    
    data = resp["data"]
    df = pd.DataFrame(data)
    
    # Convert timestamp
    df['Date'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
    
    # Calculate features
    df['fg_ma7'] = df['value'].astype(int).rolling(7, min_periods=1).mean()
    df['fg_ma14'] = df['value'].astype(int).rolling(14, min_periods=1).mean()
    df['fg_change'] = df['value'].astype(int).diff()
    df['fg_change7'] = df['value'].astype(int).diff(7)
    df['fg_extreme_fear'] = (df['value'].astype(int) < 25).astype(int)
    df['fg_extreme_greed'] = (df['value'].astype(int) > 75).astype(int)
    
    # Select columns
    df = df[['Date', 'value', 'fg_ma7', 'fg_ma14', 'fg_change', 'fg_change7', 'fg_extreme_fear', 'fg_extreme_greed']]
    df = df.rename(columns={'value': 'fear_greed'})
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"Fetched {len(df)} rows from API")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Latest FG: {df['fear_greed'].iloc[0]} ({df['Date'].iloc[0].strftime('%Y-%m-%d')})")
    
    # Save
    df.to_csv('sentiment_new.csv', index=False)
    print(f"\n✅ Saved sentiment_new.csv")
    
except Exception as e:
    print(f"Error: {e}")
    exit(1)
