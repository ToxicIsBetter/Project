"""
DATA GAP FILLER — Fetch missing OHLCV + Google Trends
======================================================
Saves to SEPARATE files so you can manually review before merging.

Output files (in FINAL_STRAW/data/):
  - fresh_ohlcv_20260109_to_now.csv      (OHLCV from Jan 9 2026 → today)
  - fresh_google_20260302_to_now.csv      (Google Trends from Mar 2 2026 → today)

Usage:
    cd FINAL_STRAW
    source ../../.venv/bin/activate
    pip install yfinance pytrends   # if not already installed
    python scripts/collect_missing_data.py
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, 'data')

TODAY = datetime.now().strftime('%Y-%m-%d')


# ══════════════════════════════════════════════════════════════════════
#  1. FETCH MISSING OHLCV (Yahoo Finance)
# ══════════════════════════════════════════════════════════════════════
def fetch_ohlcv():
    """Fetch BTC-USD daily OHLCV from 2026-01-09 to today."""
    print("\n" + "=" * 60)
    print("  1. Fetching OHLCV from Yahoo Finance")
    print("=" * 60)

    try:
        import yfinance as yf
    except ImportError:
        print("  ❌ yfinance not installed. Run: pip install yfinance")
        return False

    start_date = '2026-01-09'
    print(f"  Fetching BTC-USD: {start_date} → {TODAY}")

    btc = yf.download('BTC-USD', start=start_date, end=TODAY, progress=False)

    if btc.empty:
        print("  ❌ No data returned from Yahoo Finance.")
        return False

    # Flatten multi-level columns if present
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)

    btc = btc.reset_index()
    btc = btc.rename(columns={'index': 'Date'})

    # Keep only the columns matching clean_ohlcv.csv format
    cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols:
        if col not in btc.columns:
            print(f"  ⚠️ Missing column: {col}")

    btc = btc[[c for c in cols if c in btc.columns]]
    btc['Date'] = pd.to_datetime(btc['Date']).dt.normalize()
    btc = btc.sort_values('Date').reset_index(drop=True)

    out_path = os.path.join(DATA_DIR, 'fresh_ohlcv_20260109_to_now.csv')
    btc.to_csv(out_path, index=False)

    print(f"  ✅ Saved {len(btc)} rows → {os.path.basename(out_path)}")
    print(f"     Date range: {btc['Date'].min().date()} → {btc['Date'].max().date()}")
    print(f"     Latest close: ${btc['Close'].iloc[-1]:,.2f}")
    return True


# ══════════════════════════════════════════════════════════════════════
#  2. FETCH MISSING GOOGLE TRENDS
# ══════════════════════════════════════════════════════════════════════
def fetch_google_trends():
    """Fetch Google Trends 'Bitcoin' interest from 2026-03-01 to today."""
    print("\n" + "=" * 60)
    print("  2. Fetching Google Trends")
    print("=" * 60)

    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("  ❌ pytrends not installed. Run: pip install pytrends")
        return False

    start_date = '2026-02-01'  # Fetch a bit earlier for MA calculations
    print(f"  Fetching 'Bitcoin' trends: {start_date} → {TODAY}")

    try:
        pytrends = TrendReq(hl='en-US', tz=0)
        timeframe = f'{start_date} {TODAY}'
        pytrends.build_payload(['Bitcoin'], cat=0, timeframe=timeframe, geo='', gprop='')
        trends = pytrends.interest_over_time()
    except Exception as e:
        print(f"  ❌ Google Trends API error: {e}")
        print("  Trying alternative method...")

        # Fallback: generate empty template for manual fill
        dates = pd.date_range(start='2026-03-02', end=TODAY, freq='D')
        trends = pd.DataFrame({
            'Date': dates,
            'google_trends': np.nan,
            'gt_ma7': np.nan,
            'gt_ma30': np.nan,
            'gt_change7': np.nan,
            'gt_momentum': np.nan,
        })
        out_path = os.path.join(DATA_DIR, 'fresh_google_TEMPLATE_20260302_to_now.csv')
        trends.to_csv(out_path, index=False)
        print(f"  📝 Saved BLANK TEMPLATE ({len(trends)} rows) → {os.path.basename(out_path)}")
        print(f"     Fill in the 'google_trends' column manually from https://trends.google.com")
        return True

    if trends.empty:
        print("  ❌ No trends data returned.")
        return False

    trends = trends.reset_index()
    trends = trends.rename(columns={'date': 'Date', 'Bitcoin': 'google_trends'})

    if 'isPartial' in trends.columns:
        trends = trends.drop(columns=['isPartial'])

    trends['Date'] = pd.to_datetime(trends['Date']).dt.normalize()

    # Engineer the same derived features as clean_google.csv
    trends['gt_ma7']      = trends['google_trends'].rolling(7).mean()
    trends['gt_ma30']     = trends['google_trends'].rolling(30).mean()
    trends['gt_change7']  = trends['google_trends'].pct_change(7)
    trends['gt_momentum'] = trends['google_trends'] - trends['gt_ma30']

    # Only keep rows from March 2 onwards (the gap start)
    trends = trends[trends['Date'] >= '2026-03-02'].reset_index(drop=True)

    cols = ['Date', 'google_trends', 'gt_ma7', 'gt_ma30', 'gt_change7', 'gt_momentum']
    trends = trends[[c for c in cols if c in trends.columns]]

    out_path = os.path.join(DATA_DIR, 'fresh_google_20260302_to_now.csv')
    trends.to_csv(out_path, index=False)

    print(f"  ✅ Saved {len(trends)} rows → {os.path.basename(out_path)}")
    print(f"     Date range: {trends['Date'].min().date()} → {trends['Date'].max().date()}")
    return True


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  DATA GAP FILLER — Collecting Missing Data")
    print(f"  Today: {TODAY}")
    print("=" * 60)

    print("\n  Current data gaps:")
    print("    OHLCV:         2026-01-09 → today (NaN)")
    print("    Google Trends: 2026-03-02 → today (missing)")
    print("    Sentiment:     ✅ Up to date")
    print("    On-Chain:      ⚠️ May have NaN after Jan 8")

    ohlcv_ok  = fetch_ohlcv()
    google_ok = fetch_google_trends()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  OHLCV:         {'✅ Fetched' if ohlcv_ok else '❌ Failed'}")
    print(f"  Google Trends: {'✅ Fetched' if google_ok else '❌ Failed'}")
    print(f"\n  Output saved to: {DATA_DIR}/")
    print(f"    fresh_ohlcv_20260109_to_now.csv")
    print(f"    fresh_google_20260302_to_now.csv")
    print(f"\n  Next steps:")
    print(f"    1. Review the fresh CSV files")
    print(f"    2. Manually merge into clean_ohlcv.csv and clean_google.csv")
    print(f"    3. Retrain the model with: python scripts/retrain_rank3.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
