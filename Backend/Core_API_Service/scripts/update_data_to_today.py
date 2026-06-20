"""
DATA UPDATER — Fetch all missing data from April 2026 → Today
===============================================================
Updates all four clean_*.csv files in-place by appending new rows.

Sources:
  - OHLCV:      Yahoo Finance (BTC-USD)
  - On-Chain:   CoinMetrics Community API
  - Sentiment:  Alternative.me Fear & Greed Index API
  - Google:     PyTrends (Google Trends for 'Bitcoin')

Usage:
    cd Core_API_Service
    source ../../.venv/bin/activate
    python scripts/update_data_to_today.py
"""
import os
import sys
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, 'data')

TODAY = datetime.now().strftime('%Y-%m-%d')
TODAY_DT = datetime.now()


def get_last_date(csv_file):
    """Get the last date in a clean CSV file."""
    df = pd.read_csv(os.path.join(DATA_DIR, csv_file))
    return pd.to_datetime(df['Date']).max()


# ══════════════════════════════════════════════════════════════════════
#  1. UPDATE OHLCV (Yahoo Finance)
# ══════════════════════════════════════════════════════════════════════
def update_ohlcv():
    print("\n" + "=" * 60)
    print("  1. Updating OHLCV from Yahoo Finance")
    print("=" * 60)

    try:
        import yfinance as yf
    except ImportError:
        print("  ❌ yfinance not installed. Run: pip install yfinance")
        return False

    last_date = get_last_date('clean_ohlcv.csv')
    start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"  Current data ends: {last_date.date()}")
    print(f"  Fetching BTC-USD: {start_date} → {TODAY}")

    if start_date >= TODAY:
        print("  ✅ Already up to date!")
        return True

    btc = yf.download('BTC-USD', start=start_date, end=TODAY, progress=False)

    if btc.empty:
        print("  ❌ No new data returned.")
        return False

    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)

    btc = btc.reset_index()
    if 'index' in btc.columns:
        btc = btc.rename(columns={'index': 'Date'})

    cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    btc = btc[[c for c in cols if c in btc.columns]]
    btc['Date'] = pd.to_datetime(btc['Date']).dt.normalize().dt.strftime('%Y-%m-%d')
    btc = btc.sort_values('Date').reset_index(drop=True)

    # Append to existing
    existing = pd.read_csv(os.path.join(DATA_DIR, 'clean_ohlcv.csv'))
    combined = pd.concat([existing, btc], ignore_index=True)
    combined['Date'] = pd.to_datetime(combined['Date'])
    combined = combined.drop_duplicates(subset='Date', keep='last').sort_values('Date').reset_index(drop=True)
    combined['Date'] = combined['Date'].dt.strftime('%Y-%m-%d')
    combined.to_csv(os.path.join(DATA_DIR, 'clean_ohlcv.csv'), index=False)

    print(f"  ✅ Added {len(btc)} new rows. Total: {len(combined)}")
    print(f"     New end date: {combined['Date'].iloc[-1]}")
    return True


# ══════════════════════════════════════════════════════════════════════
#  2. UPDATE ON-CHAIN (CoinMetrics Community API)
# ══════════════════════════════════════════════════════════════════════
def update_onchain():
    print("\n" + "=" * 60)
    print("  2. Updating On-Chain from CoinMetrics Community API")
    print("=" * 60)

    last_date = get_last_date('clean_onchain.csv')
    start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"  Current data ends: {last_date.date()}")
    print(f"  Fetching: {start_date} → {TODAY}")

    if start_date >= TODAY:
        print("  ✅ Already up to date!")
        return True

    # Read existing to get column list
    existing = pd.read_csv(os.path.join(DATA_DIR, 'clean_onchain.csv'))
    all_cols = [c for c in existing.columns if c != 'Date']

    # Raw CoinMetrics metrics (the base metrics before derived features)
    raw_metrics = [
        'AdrActCnt', 'AdrBalCnt', 'BlkCnt', 'CapMVRVCur', 'CapMrktCurUSD',
        'CapMrktEstUSD', 'FeeTotNtv', 'FlowInExNtv', 'FlowInExUSD',
        'FlowOutExNtv', 'FlowOutExUSD', 'HashRate', 'IssTotNtv', 'IssTotUSD',
        'ROI1yr', 'ROI30d', 'ReferenceRate', 'ReferenceRateETH',
        'ReferenceRateEUR', 'ReferenceRateUSD', 'SplyCur', 'SplyExNtv',
        'SplyExUSD', 'SplyExpFut10yr', 'TxCnt', 'TxTfrCnt'
    ]

    metrics_str = ','.join(raw_metrics)
    url = f"https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    params = {
        'assets': 'btc',
        'metrics': metrics_str,
        'start_time': start_date,
        'end_time': TODAY,
        'frequency': '1d',
        'page_size': 10000,
    }

    print(f"  Requesting {len(raw_metrics)} metrics from CoinMetrics...")
    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  ❌ CoinMetrics API error: {e}")
        return False

    if 'data' not in data or len(data['data']) == 0:
        print("  ❌ No data returned from CoinMetrics.")
        return False

    rows = data['data']
    new_df = pd.DataFrame(rows)
    new_df = new_df.rename(columns={'time': 'Date'})
    new_df['Date'] = pd.to_datetime(new_df['Date']).dt.tz_localize(None).dt.normalize()

    # Drop asset column if present
    if 'asset' in new_df.columns:
        new_df = new_df.drop(columns=['asset'])

    # Convert numeric columns
    for col in new_df.columns:
        if col != 'Date':
            new_df[col] = pd.to_numeric(new_df[col], errors='coerce')

    # Add volume_reported_spot_usd_1d if missing
    if 'volume_reported_spot_usd_1d' not in new_df.columns:
        new_df['volume_reported_spot_usd_1d'] = np.nan

    # Also fetch volume separately if needed
    try:
        vol_url = f"https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
        vol_params = {
            'assets': 'btc',
            'metrics': 'volume_reported_spot_usd_1d',
            'start_time': start_date,
            'end_time': TODAY,
            'frequency': '1d',
            'page_size': 10000,
        }
        vol_resp = requests.get(vol_url, params=vol_params, timeout=30)
        if vol_resp.status_code == 200:
            vol_data = vol_resp.json()
            if 'data' in vol_data and len(vol_data['data']) > 0:
                vol_df = pd.DataFrame(vol_data['data'])
                vol_df = vol_df.rename(columns={'time': 'Date'})
                vol_df['Date'] = pd.to_datetime(vol_df['Date']).dt.tz_localize(None).dt.normalize()
                if 'asset' in vol_df.columns:
                    vol_df = vol_df.drop(columns=['asset'])
                vol_df['volume_reported_spot_usd_1d'] = pd.to_numeric(vol_df['volume_reported_spot_usd_1d'], errors='coerce')
                new_df = new_df.drop(columns=['volume_reported_spot_usd_1d'], errors='ignore')
                new_df = new_df.merge(vol_df[['Date', 'volume_reported_spot_usd_1d']], on='Date', how='left')
    except:
        pass

    # AssetCompletionTime and AssetEODCompletionTime — set to NaN (not available from community API)
    if 'AssetCompletionTime' not in new_df.columns:
        new_df['AssetCompletionTime'] = np.nan
    if 'AssetEODCompletionTime' not in new_df.columns:
        new_df['AssetEODCompletionTime'] = np.nan

    # Now compute OHLCV-derived technical indicators using existing + new OHLCV
    ohlcv = pd.read_csv(os.path.join(DATA_DIR, 'clean_ohlcv.csv'), parse_dates=['Date'])
    ohlcv['Date'] = ohlcv['Date'].dt.normalize()

    # Merge new on-chain rows with OHLCV for technical calculation
    merged = ohlcv.merge(new_df, on='Date', how='inner')

    if len(merged) == 0:
        print("  ⚠️ No overlapping dates between new on-chain and OHLCV data.")
        print("     Make sure OHLCV is updated first.")
        return False

    # We need the FULL existing + new OHLCV to compute rolling indicators
    full_ohlcv = ohlcv.sort_values('Date').reset_index(drop=True)
    close = full_ohlcv['Close']
    high = full_ohlcv['High']
    low = full_ohlcv['Low']
    volume = full_ohlcv['Volume']

    # Compute technicals on full OHLCV
    tech = pd.DataFrame({'Date': full_ohlcv['Date']})
    tech['SMA_7'] = close.rolling(7).mean()
    tech['SMA_14'] = close.rolling(14).mean()
    tech['SMA_21'] = close.rolling(21).mean()
    tech['SMA_50'] = close.rolling(50).mean()
    tech['SMA_100'] = close.rolling(100).mean()
    tech['SMA_200'] = close.rolling(200).mean()

    # EMA
    tech['EMA_12'] = close.ewm(span=12).mean()
    tech['EMA_26'] = close.ewm(span=26).mean()
    tech['EMA_50'] = close.ewm(span=50).mean()

    # MACD
    tech['MACD'] = tech['EMA_12'] - tech['EMA_26']
    tech['MACD_Signal'] = tech['MACD'].ewm(span=9).mean()
    tech['MACD_Hist'] = tech['MACD'] - tech['MACD_Signal']

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain_14 = gain.rolling(14).mean()
    avg_loss_14 = loss.rolling(14).mean()
    rs_14 = avg_gain_14 / (avg_loss_14 + 1e-10)
    tech['RSI_14'] = 100 - (100 / (1 + rs_14))

    avg_gain_21 = gain.rolling(21).mean()
    avg_loss_21 = loss.rolling(21).mean()
    rs_21 = avg_gain_21 / (avg_loss_21 + 1e-10)
    tech['RSI_21'] = 100 - (100 / (1 + rs_21))

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    tech['BB_upper'] = sma20 + 2 * std20
    tech['BB_lower'] = sma20 - 2 * std20
    tech['BB_width'] = (tech['BB_upper'] - tech['BB_lower']) / (sma20 + 1e-10)
    tech['BB_pct'] = (close - tech['BB_lower']) / (tech['BB_upper'] - tech['BB_lower'] + 1e-10)

    # Returns
    for d in [1, 3, 5, 7, 10, 14, 21, 30]:
        tech[f'return_{d}d'] = close.pct_change(d)

    # Volatility
    for d in [7, 14, 21, 30, 60]:
        tech[f'vol_{d}d'] = close.pct_change().rolling(d).std()

    # Price ratios
    tech['price_sma50_ratio'] = close / (tech['SMA_50'] + 1e-10)
    tech['price_sma200_ratio'] = close / (tech['SMA_200'] + 1e-10)
    tech['price_sma21_ratio'] = close / (tech['SMA_21'] + 1e-10)

    # Cross signals
    tech['sma7_sma21_cross'] = (tech['SMA_7'] > tech['SMA_21']).astype(float)
    tech['sma21_sma50_cross'] = (tech['SMA_21'] > tech['SMA_50']).astype(float)
    tech['sma50_sma200_cross'] = (tech['SMA_50'] > tech['SMA_200']).astype(float)

    # Volume features
    tech['vol_sma7'] = volume.rolling(7).mean()
    tech['vol_sma30'] = volume.rolling(30).mean()
    tech['vol_ratio'] = tech['vol_sma7'] / (tech['vol_sma30'] + 1e-10)

    # OBV
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    tech['OBV'] = obv
    tech['OBV_sma7'] = obv.rolling(7).mean()

    # Stochastics
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    tech['Stoch_K'] = 100 * (close - low14) / (high14 - low14 + 1e-10)
    tech['Stoch_D'] = tech['Stoch_K'].rolling(3).mean()

    # True Range / ATR
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tech['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tech['ATR'] = tech['TR'].rolling(14).mean()

    # ROC
    tech['ROC_5'] = close.pct_change(5) * 100
    tech['ROC_10'] = close.pct_change(10) * 100
    tech['ROC_20'] = close.pct_change(20) * 100

    # Filter tech to only new dates
    new_dates = new_df['Date'].unique()
    tech_new = tech[tech['Date'].isin(new_dates)].copy()

    # Merge technicals with on-chain
    final_new = tech_new.merge(new_df.drop(columns=[c for c in tech_new.columns if c != 'Date'], errors='ignore'), on='Date', how='inner')

    # Compute growth features on the combined dataset
    # We need historical context, so load existing and append
    existing_onchain = pd.read_csv(os.path.join(DATA_DIR, 'clean_onchain.csv'), parse_dates=['Date'])
    existing_onchain['Date'] = existing_onchain['Date'].dt.normalize()

    # For growth features, we need to compute on the full series
    # First, let's build the raw on-chain part of final_new
    for metric, periods in [('AdrActCnt', [7, 30]), ('TxCnt', [7, 30]), ('HashRate', [7, 30]), ('CapMVRVCur', [7, 30])]:
        for p in periods:
            col_name = f'{metric}_growth{p}d'
            if metric in new_df.columns:
                # Get historical values from existing
                hist_vals = existing_onchain[['Date', metric]].copy()
                new_vals = new_df[['Date', metric]].copy()
                combined_metric = pd.concat([hist_vals, new_vals]).drop_duplicates('Date').sort_values('Date')
                combined_metric[col_name] = combined_metric[metric].pct_change(p)
                # Extract only new dates
                growth_vals = combined_metric[combined_metric['Date'].isin(new_dates)][['Date', col_name]]
                final_new = final_new.merge(growth_vals, on='Date', how='left')

    # Ensure all columns from existing are present
    for col in existing_onchain.columns:
        if col not in final_new.columns:
            final_new[col] = np.nan

    # Reorder columns to match existing
    final_new = final_new[existing_onchain.columns]
    final_new['Date'] = final_new['Date'].dt.strftime('%Y-%m-%d')

    # Append
    existing_onchain['Date'] = existing_onchain['Date'].dt.strftime('%Y-%m-%d')
    combined = pd.concat([existing_onchain, final_new], ignore_index=True)
    combined['Date'] = pd.to_datetime(combined['Date'])
    combined = combined.drop_duplicates(subset='Date', keep='last').sort_values('Date').reset_index(drop=True)
    combined['Date'] = combined['Date'].dt.strftime('%Y-%m-%d')
    combined.to_csv(os.path.join(DATA_DIR, 'clean_onchain.csv'), index=False)

    print(f"  ✅ Added {len(final_new)} new rows. Total: {len(combined)}")
    print(f"     New end date: {combined['Date'].iloc[-1]}")
    return True


# ══════════════════════════════════════════════════════════════════════
#  3. UPDATE SENTIMENT (Alternative.me Fear & Greed Index)
# ══════════════════════════════════════════════════════════════════════
def update_sentiment():
    print("\n" + "=" * 60)
    print("  3. Updating Sentiment from Alternative.me")
    print("=" * 60)

    last_date = get_last_date('clean_sentiment.csv')
    start_date = last_date + timedelta(days=1)
    days_needed = (TODAY_DT - start_date).days + 1
    print(f"  Current data ends: {last_date.date()}")
    print(f"  Fetching last {days_needed} days of Fear & Greed data")

    if days_needed <= 0:
        print("  ✅ Already up to date!")
        return True

    url = f"https://api.alternative.me/fng/?limit={days_needed + 30}&format=json"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  ❌ API error: {e}")
        return False

    if 'data' not in data:
        print("  ❌ No data in response.")
        return False

    records = []
    for entry in data['data']:
        dt = datetime.fromtimestamp(int(entry['timestamp']))
        records.append({
            'Date': dt.strftime('%Y-%m-%d'),
            'fear_greed': int(entry['value']),
        })

    new_df = pd.DataFrame(records)
    new_df['Date'] = pd.to_datetime(new_df['Date'])
    new_df = new_df.sort_values('Date').reset_index(drop=True)

    # Load existing and combine for rolling calculations
    existing = pd.read_csv(os.path.join(DATA_DIR, 'clean_sentiment.csv'), parse_dates=['Date'])
    combined = pd.concat([existing[['Date', 'fear_greed']], new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset='Date', keep='last').sort_values('Date').reset_index(drop=True)

    # Compute derived features on full series
    combined['fg_ma7'] = combined['fear_greed'].rolling(7).mean()
    combined['fg_ma14'] = combined['fear_greed'].rolling(14).mean()
    combined['fg_change'] = combined['fear_greed'].diff()
    combined['fg_change7'] = combined['fear_greed'].diff(7)
    combined['fg_extreme_fear'] = (combined['fear_greed'] <= 20).astype(int)
    combined['fg_extreme_greed'] = (combined['fear_greed'] >= 80).astype(int)

    combined['Date'] = combined['Date'].dt.strftime('%Y-%m-%d')
    combined.to_csv(os.path.join(DATA_DIR, 'clean_sentiment.csv'), index=False)

    new_count = len(combined) - len(existing)
    print(f"  ✅ Added {new_count} new rows. Total: {len(combined)}")
    print(f"     New end date: {combined['Date'].iloc[-1]}")
    return True


# ══════════════════════════════════════════════════════════════════════
#  4. UPDATE GOOGLE TRENDS
# ══════════════════════════════════════════════════════════════════════
def update_google_trends():
    print("\n" + "=" * 60)
    print("  4. Updating Google Trends")
    print("=" * 60)

    last_date = get_last_date('clean_google.csv')
    start_date = last_date - timedelta(days=35)  # Extra days for MA calculation
    print(f"  Current data ends: {last_date.date()}")
    print(f"  Fetching Google Trends: {start_date.date()} → {TODAY}")

    if last_date.date() >= TODAY_DT.date():
        print("  ✅ Already up to date!")
        return True

    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("  ❌ pytrends not installed. Run: pip install pytrends")
        return False

    try:
        pytrends = TrendReq(hl='en-US', tz=0)
        timeframe = f'{start_date.strftime("%Y-%m-%d")} {TODAY}'
        pytrends.build_payload(['Bitcoin'], cat=0, timeframe=timeframe, geo='', gprop='')
        trends = pytrends.interest_over_time()
    except Exception as e:
        print(f"  ❌ Google Trends API error: {e}")
        print("  ⚠️ Google Trends may rate-limit. Try again later.")
        return False

    if trends.empty:
        print("  ❌ No trends data returned.")
        return False

    trends = trends.reset_index()
    trends = trends.rename(columns={'date': 'Date', 'Bitcoin': 'google_trends'})
    if 'isPartial' in trends.columns:
        trends = trends.drop(columns=['isPartial'])
    trends['Date'] = pd.to_datetime(trends['Date']).dt.normalize()

    # Load existing and combine for rolling calculations
    existing = pd.read_csv(os.path.join(DATA_DIR, 'clean_google.csv'), parse_dates=['Date'])
    existing['Date'] = existing['Date'].dt.normalize()

    # Combine raw google_trends values
    combined = pd.concat([existing[['Date', 'google_trends']], trends[['Date', 'google_trends']]], ignore_index=True)
    combined = combined.drop_duplicates(subset='Date', keep='last').sort_values('Date').reset_index(drop=True)

    # Compute derived features
    combined['gt_ma7'] = combined['google_trends'].rolling(7).mean()
    combined['gt_ma30'] = combined['google_trends'].rolling(30).mean()
    combined['gt_change7'] = combined['google_trends'].pct_change(7)
    combined['gt_momentum'] = combined['google_trends'] - combined['gt_ma30']

    combined['Date'] = combined['Date'].dt.strftime('%Y-%m-%d')
    combined.to_csv(os.path.join(DATA_DIR, 'clean_google.csv'), index=False)

    new_count = len(combined) - len(existing)
    print(f"  ✅ Added {new_count} new rows. Total: {len(combined)}")
    print(f"     New end date: {combined['Date'].iloc[-1]}")
    return True


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  NeuralEdge DATA UPDATER")
    print(f"  Updating all data to: {TODAY}")
    print("=" * 60)

    results = {}
    results['OHLCV'] = update_ohlcv()
    results['On-Chain'] = update_onchain()
    results['Sentiment'] = update_sentiment()
    results['Google Trends'] = update_google_trends()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for source, ok in results.items():
        status = '✅ Updated' if ok else '❌ Failed'
        print(f"  {source:20s} {status}")

    print(f"\n  Data directory: {DATA_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
