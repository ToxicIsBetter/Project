"""
btc_8h_pipeline.py
Collects 8-hourly BTC data from:
1. Binance API → OHLCV price (8H candles)
2. Blockchain.com → On-chain metrics (resampled to 8H)
3. CoinMetrics btc.csv → On-chain metrics (daily → interpolated to 8H)

Output: btc_8h_full.csv
"""

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
from pathlib import Path


# ─────────────────────────────────────────────────────────────────
# 1. BINANCE — 8H OHLCV price data
# ─────────────────────────────────────────────────────────────────
def fetch_binance_8h(start_date="2017-01-01"):
    """Fetch Binance 8H OHLCV data."""
    url = "https://api.binance.com/api/v3/klines"

    start_ms = int(
        datetime.strptime(start_date, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1000
    )
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    all_data = []
    current = start_ms

    print(f"⬇️ Fetching Binance 8H data from {start_date}...")

    while current < end_ms:
        params = {
            "symbol": "BTCUSDT",
            "interval": "8h",
            "startTime": current,
            "endTime": end_ms,
            "limit": 1000,
        }

        try:
            resp = requests.get(url, params=params, timeout=30).json()
            if not resp:
                break
            all_data.extend(resp)
            # Use close_time (index 6) to move forward
            current = int(resp[-1][6]) + 1
            time.sleep(0.2)
        except Exception as e:
            print(f"⚠️ Binance error: {e}")
            time.sleep(5)
            break

    if not all_data:
        raise RuntimeError("No Binance data fetched")

    df = pd.DataFrame(
        all_data,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "qav",
            "num_trades",
            "tbbav",
            "tbqav",
            "ignore",
        ],
    )

    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    df.rename(columns={"close": "price"}, inplace=True)
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    # Feature engineering
    df["return"] = df["price"].pct_change()
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))
    df["range"] = df["high"] - df["low"]
    df["vol_7p"] = df["return"].rolling(7).std()
    df["vol_24p"] = df["return"].rolling(24).std()
    df["ma_24"] = df["price"].rolling(24).mean()
    df["ma_168"] = df["price"].rolling(168).mean()
    df["price_ma_ratio"] = df["price"] / df["ma_24"]

    print(f" ✅ Binance 8H: {df.shape[0]:,} rows ({df.index[0]} → {df.index[-1]})")
    return df


# ─────────────────────────────────────────────────────────────────
# 2. BLOCKCHAIN.COM — On-chain metrics
# ─────────────────────────────────────────────────────────────────
def fetch_blockchain_chart(chart_name, col_name, timespan="5years"):
    """Fetch a single Blockchain.com chart metric."""
    url = (
        f"https://api.blockchain.info/charts/{chart_name}"
        f"?timespan={timespan}&format=json&sampled=false"
    )

    try:
        resp = requests.get(url, timeout=15).json()
        df = pd.DataFrame(resp["values"])
        df["date"] = pd.to_datetime(df["x"], unit="s", utc=True)
        df = df[["date", "y"]].rename(columns={"y": col_name})
        df.set_index("date", inplace=True)
        return df
    except Exception as e:
        print(f" ⚠️ {chart_name} failed: {e}")
        return pd.DataFrame()


def fetch_all_blockchain():
    """Fetch Blockchain.com on-chain metrics."""
    metrics = {
        "n-transactions": "tx_count",
        "hash-rate": "hashrate",
        "transaction-fees": "fees_btc",
        "mempool-size": "mempool_size",
        "miners-revenue": "miners_revenue",
        "difficulty": "difficulty",
    }

    frames = []
    print("⬇️ Fetching Blockchain.com on-chain metrics...")

    for chart, col in metrics.items():
        df = fetch_blockchain_chart(chart, col)
        if not df.empty:
            frames.append(df)
            print(f" ✅ {chart}: {len(df):,} rows")
        time.sleep(1)  # Rate limit

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1)
    combined.sort_index(inplace=True)

    # Resample to 8H
    combined = combined.resample("8h").mean()
    print(f" ✅ Blockchain.com 8H: {combined.shape}")
    return combined


# ─────────────────────────────────────────────────────────────────
# 3. COINMETRICS — Daily → 8H interpolation
# ─────────────────────────────────────────────────────────────────
def load_coinmetrics_8h(path="btc.csv"):
    """Load CoinMetrics daily data and resample to 8H."""
    print(f"⬇️ Loading CoinMetrics from {path}...")

    if not Path(path).exists():
        print(f" ⚠️ File not found: {path}")
        return pd.DataFrame()

    cm = pd.read_csv(path)

    if "time" not in cm.columns or "date" not in cm.columns:
        # Try to find date column
        date_cols = [
            c for c in cm.columns if "date" in c.lower() or "time" in c.lower()
        ]
        if date_cols:
            cm["date"] = pd.to_datetime(cm[date_cols[0]], utc=True)
            cm.set_index("date", inplace=True)
            cm.drop(columns=[date_cols[0]], inplace=True, errors="ignore")
        else:
            print(" ⚠️ No date column found")
            return pd.DataFrame()
    else:
        cm["date"] = pd.to_datetime(cm["time"], utc=True)
        cm.set_index("date", inplace=True)
        cm.drop(columns=["time"], inplace=True, errors="ignore")

    # Keep numeric columns only
    cm = cm.select_dtypes(include=[np.number])

    # Interpolate to 8H
    cm_8h = cm.resample("8H").interpolate(method="time")

    print(f" ✅ CoinMetrics 8H: {cm_8h.shape}")
    return cm_8h


# ─────────────────────────────────────────────────────────────────
# 4. MERGE ALL
# ─────────────────────────────────────────────────────────────────
def build_8h_dataset(coinmetrics_path="btc.csv"):
    """Build complete 8H dataset."""
    print("\n" + "=" * 55)
    print(" Building 8H Dataset")
    print("=" * 55 + "\n")

    # Step 1: Price (Binance)
    df_price = fetch_binance_8h()

    # Step 2: Blockchain.com
    df_chain = fetch_all_blockchain()

    # Step 3: CoinMetrics
    df_cm = load_coinmetrics_8h(coinmetrics_path)

    # Step 4: Join all
    df = df_price.copy()

    if not df_chain.empty:
        df = df.join(df_chain, how="left")

    if not df_cm.empty:
        df = df.join(df_cm, how="left", rsuffix="_cm")

    # Step 5: Forward fill and drop NaN
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    print(f"\n{'=' * 55}")
    print(f"✅ Final 8H dataset: {df.shape}")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Date range: {df.index[0]} → {df.index[-1]}")
    print(f"{'=' * 55}\n")

    return df


# ─────────────────────────────────────────────────────────────────
# 5. RUN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Default CoinMetrics path
    cm_path = "btc.csv"
    if len(sys.argv) > 1:
        cm_path = sys.argv[1]

    # Build dataset
    df_8h = build_8h_dataset(coinmetrics_path=cm_path)

    # Save
    output_file = "btc_8h_full.csv"
    df_8h.to_csv(output_file)
    print(f"✅ Saved: {output_file}")
    print("\n🎉 DONE")
