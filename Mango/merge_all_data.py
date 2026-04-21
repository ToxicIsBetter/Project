"""
Merge All Data Sources
Combines:
1. Price data (Binance 8H)
2. On-chain data (Blockchain.com + CoinMetrics)
3. Sentiment data (Fear & Greed, CFGI, Google Trends, Reddit)

Output: btc_8h_complete.csv (~40 features)
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("MERGING ALL DATA SOURCES")
print("=" * 60)

# Paths
PRICE_FILE = "btc_8h_full.csv"
SENTIMENT_FILE = "btc_8h_sentiment.csv"
OUTPUT_FILE = "btc_8h_complete.csv"

# Load price data
if not Path(PRICE_FILE).exists():
    print(f"❌ Price data not found: {PRICE_FILE}")
    print("   Run: python collect_8h_data.py")
    exit(1)

print(f"\nLoading price data: {PRICE_FILE}")
price = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True)
print(f"   Shape: {price.shape}")

# Load sentiment data (if exists)
sentiment = None
if Path(SENTIMENT_FILE).exists():
    print(f"Loading sentiment data: {SENTIMENT_FILE}")
    sentiment = pd.read_csv(SENTIMENT_FILE, index_col=0, parse_dates=True)
    print(f"   Shape: {sentiment.shape}")
else:
    print("⚠️ Sentiment data not found. Run collect_sentiment_8h.py")

# Merge
df = price.copy()

if sentiment is not None:
    print("\nMerging sentiment data...")
    df = df.join(sentiment, how="left")
    df.ffill(inplace=True)
    df.bfill(inplace=True)

# Drop rows with any remaining NaN
print("\nCleaning data...")
initial_shape = df.shape
df = df.dropna()
print(f"   Dropped {initial_shape[0] - len(df)} rows with NaN")

# Final report
print("\n" + "=" * 60)
print("FINAL DATASET")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"Date range: {df.index[0]} → {df.index[-1]}")
print(f"\nFeatures by category:")

# Categorize columns
price_cols = [
    c
    for c in df.columns
    if any(
        x in c.lower()
        for x in [
            "open",
            "high",
            "low",
            "close",
            "price",
            "volume",
            "return",
            "ma_",
            "vol_",
        ]
    )
]
onchain_cols = [
    c
    for c in df.columns
    if any(
        x in c.lower()
        for x in [
            "tx_",
            "hash",
            "fee",
            "mempool",
            "miner",
            "difficult",
            "utxo",
            "adr",
            "supply",
        ]
    )
]
sentiment_cols = [
    c
    for c in df.columns
    if any(
        x in c.lower()
        for x in ["fear", "greed", "cfgi", "gt_", "google", "reddit", "sentiment"]
    )
]
other_cols = [
    c for c in df.columns if c not in price_cols + onchain_cols + sentiment_cols
]

print(f"  Price features:     {len(price_cols)}")
print(f"  On-chain features:  {len(onchain_cols)}")
print(f"  Sentiment features: {len(sentiment_cols)}")
if other_cols:
    print(f"  Other features:     {len(other_cols)}")

print(f"\n  Total: {len(df.columns)} features")

# Save
df.to_csv(OUTPUT_FILE)
print(f"\n✅ Saved: {OUTPUT_FILE}")

# Feature list for model training
feature_list = {
    "price": price_cols,
    "onchain": onchain_cols,
    "sentiment": sentiment_cols,
    "other": other_cols,
    "all": list(df.columns),
}

import json

with open("feature_categories.json", "w") as f:
    json.dump(feature_list, f, indent=2)

print(f"✅ Feature categories saved: feature_categories.json")
print("\n" + "=" * 60)
print("READY FOR TRAINING")
print("=" * 60)
print(f"\nUse this dataset for Model3_8H training:")
print(f"  cd Model3_8H")
print(f"  python train_model3_8h.py --data ../{OUTPUT_FILE}")
print("\n🎉 DONE")
