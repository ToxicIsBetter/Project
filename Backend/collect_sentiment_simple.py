"""
Simple Sentiment Data Collection  
Collects Fear & Greed Index from Alternative.me
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

def fetch_fear_greed(days=365):
    """Fetch Fear & Greed Index from Alternative.me"""
    url = f"https://api.alternative.me/fng/?limit={days}"
    
    try:
        resp = requests.get(url, timeout=15).json()
        
        if "data" not in resp:
            print(f"⚠️ Alternative.me API error: {resp}")
            return pd.DataFrame()
        
        data = resp["data"]
        
        # Create DataFrame and convert timestamp properly
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
        df["fg_score"] = df["value"].astype(int)
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        
        print(f"✅ Alternative.me Fear & Greed: {len(df)} rows")
        print(f"   Latest: {df['fg_score'].iloc[0]} ({df['value_classification'].iloc[0]})")
        print(f"   Date range: {df.index[0]} → {df.index[-1]}")
        return df[["fg_score"]]
        
    except Exception as e:
        print(f"⚠️ Alternative.me failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

if __name__ == "__main__":
    days = 365
    import sys
    if len(sys.argv) > 1:
        days = int(sys.argv[1])
    
    print("\n" + "="*55)
    print("Collecting Sentiment Data")
    print("="*55 + "\n")
    
    # Fetch Fear & Greed
    fg = fetch_fear_greed(days=days)
    
    if fg.empty:
        print("\n❌ No sentiment data collected")
        exit(1)
    
    # Resample to 8H
    fg_8h = fg.resample("8h").first()
    fg_8h.ffill(inplace=True)
    
    print(f"\n✅ Resampled to 8H: {fg_8h.shape}")
    print(f"   Date range: {fg_8h.index[0]} → {fg_8h.index[-1]}")
    
    # Save
    output = "btc_8h_sentiment.csv"
    fg_8h.to_csv(output)
    print(f"\n✅ Saved: {output}")
    
    # Merge with price data if exists
    price_file = Path("btc_8h_full.csv")
    if price_file.exists():
        print(f"\n📊 Merging with price data...")
        price = pd.read_csv(price_file, index_col=0, parse_dates=True)
        
        # Align indices
        fg_8h_aligned = fg_8h.reindex(price.index, method="ffill")
        
        # Join
        final = price.join(fg_8h_aligned)
        final.ffill(inplace=True)
        final.bfill(inplace=True)
        final.dropna(inplace=True)
        
        final_output = "btc_8h_complete.csv"
        final.to_csv(final_output)
        print(f"✅ Final dataset: {final_output}")
        print(f"   Shape: {final.shape}")
    else:
        print(f"\n⚠️ Price data not found. Run collect_8h_data.py first.")
    
    print("\n🎉 DONE")
