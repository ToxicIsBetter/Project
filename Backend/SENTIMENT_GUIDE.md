# NeuralEdge Sentiment Data Collection Guide

## Overview

This pipeline collects **free sentiment data** from 4 sources and merges it with price/on-chain data for a complete ~40 feature dataset.

## Data Sources

| Source | Frequency | History | API Key | Features |
|--------|-----------|---------|---------|----------|
| **Alternative.me** | Daily | ~2 years | ❌ No | `fg_score` (0-100) |
| **CFGI.io** | 15-min | Recent | ❌ No | `cfgi_score` |
| **Google Trends** | Daily | Variable | ❌ No | `gt_bitcoin` (0-100) |
| **Reddit** (optional) | Real-time | Recent | ⚠️ Free app | `reddit_sentiment`, `reddit_score_sum` |

## Installation

```bash
# Required dependencies
uv add requests pandas numpy pytrends

# Optional (for Reddit sentiment)
uv add praw textblob
```

## Quick Start (3 Steps)

### Step 1: Collect Price + On-Chain Data (~10 min)

```bash
cd /home/shyam/UbuntuCode/CN\ 6000\ Mental\ Wealth\ Professional\ Life\ 3\ \(Project\)/Project

uv run python Mango/collect_8h_data.py
```

**Output:** `btc_8h_full.csv` (~70,000 rows × 20 features)

---

### Step 2: Collect Sentiment Data (~2 min)

```bash
uv run python Mango/collect_sentiment_8h.py 365
```

**Output:** `btc_8h_sentiment.csv` (~2,000 rows × 8 features)

**Optional Reddit Setup:**
1. Go to `reddit.com/prefs/apps`
2. Create app (type: **Script**)
3. Copy `client_id` and `client_secret`
4. Edit `collect_sentiment_8h.py` lines 165-166 with your credentials
5. Re-run with `reddit_enabled=True`

---

### Step 3: Merge All Data (~30 sec)

```bash
uv run python Mango/merge_all_data.py
```

**Output:** `btc_8h_complete.csv` (~70,000 rows × 40 features)

---

## Feature Breakdown

### Price Features (13)
- `open`, `high`, `low`, `price`, `volume`
- `return`, `log_return`, `range`
- `vol_7p`, `vol_24p` (rolling volatility)
- `ma_24`, `ma_168` (moving averages)
- `price_ma_ratio`

### On-Chain Features (~20)
**From Blockchain.com:**
- `tx_count` - Transaction count
- `hashrate` - Network hash rate
- `fees_btc` - Transaction fees (BTC)
- `mempool_size` - Pending transactions
- `miners_revenue` - Daily miner revenue
- `difficulty` - Mining difficulty

**From CoinMetrics:**
- `AdrActCnt` - Active addresses
- `TxCnt` - Transaction count
- `SplyCur` - Current supply
- `CapMVRVCur` - MVRV ratio
- `HashRate`, `BlkCnt`, etc.

### Sentiment Features (~8)
- `fg_score` - Fear & Greed (Alternative.me)
- `cfgi_score` - CFGI sentiment
- `gt_bitcoin` - Google Trends interest
- `reddit_sentiment` - Reddit post sentiment (optional)
- `reddit_score_sum` - Total Reddit score
- `reddit_comments_sum` - Total Reddit comments

---

## Complete Pipeline (All 3 Steps)

```bash
cd /home/shyam/UbuntuCode/CN\ 6000\ Mental\ Wealth\ Professional\ Life\ 3\ \(Project\)/Project

# Step 1: Price + On-chain
uv run python Mango/collect_8h_data.py

# Step 2: Sentiment
uv run python Mango/collect_sentiment_8h.py

# Step 3: Merge
uv run python Mango/merge_all_data.py

# Step 4: Train model
cd Mango/Model3_8H
uv run python train_model3_8h.py --data ../../btc_8h_complete.csv
```

---

## Expected Output

```
============================================================
FINAL DATASET
============================================================
Shape: (70000, 41)
Date range: 2017-01-01 00:00:00 → 2026-04-18 16:00:00

Features by category:
  Price features:     13
  On-chain features:  20
  Sentiment features: 8

  Total: 41 features

✅ Saved: btc_8h_complete.csv
```

---

## Troubleshooting

### "No module named 'pytrends'"
```bash
uv add pytrends
```

### "Google Trends rate limited"
- Wait 5 minutes and re-run
- Or reduce days parameter: `collect_sentiment_8h.py 90`

### "Reddit credentials not set"
- Reddit sentiment is optional
- Skip it or set up free app at `reddit.com/prefs/apps`

### "Not enough data points"
- Ensure you have at least 1 year of data
- Increase `days` parameter in collection scripts

---

## Data Quality Notes

| Source | Reliability | Latency | Notes |
|--------|-------------|---------|-------|
| Binance | ⭐⭐⭐⭐⭐ | Real-time | Excellent |
| Blockchain.com | ⭐⭐⭐⭐ | ~1 hour | Good |
| CoinMetrics | ⭐⭐⭐⭐⭐ | Daily | Excellent |
| Alternative.me | ⭐⭐⭐⭐ | Daily | Good |
| CFGI.io | ⭐⭐⭐ | 15-min | Recent only |
| Google Trends | ⭐⭐⭐ | Daily | Gaps possible |
| Reddit | ⭐⭐ | Real-time | Noisy, optional |

**Recommendation:** Use all sources, but weight price/on-chain higher than sentiment.

---

## Next Steps

1. ✅ Collect all data (3 scripts above)
2. ✅ Train 8H model (`Model3_8H/train_model3_8h.py`)
3. ✅ Compare with Daily model (`compare_daily_vs_8h.py`)
4. ✅ Deploy winner to production API

---

## Files Created

| File | Description |
|------|-------------|
| `btc_8h_full.csv` | Price + on-chain (8H) |
| `btc_8h_sentiment.csv` | All sentiment sources (8H) |
| `btc_8h_complete.csv` | **Final merged dataset** |
| `feature_categories.json` | Feature grouping for model |

---

## Questions?

- **Why 8H frequency?** → Balances noise vs opportunity frequency
- **Why not hourly?** → Too noisy, sentiment updates slowly
- **Reddit worth it?** → Optional, adds noise but captures retail sentiment spikes
