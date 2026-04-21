# NeuralEdge Complete Pipeline Guide

## What You Have Now

A complete **8-hourly Bitcoin price prediction system** with:
- ✅ Price data (Binance 8H candles)
- ✅ On-chain data (Blockchain.com + CoinMetrics)
- ✅ Sentiment data (Fear & Greed, CFGI, Google Trends, Reddit)
- ✅ Dual-Head Transformer model (8H frequency)
- ✅ Comparison with Daily model
- ✅ Deployment-ready API

---

## File Structure

```
Project/
├── Mango/
│   ├── collect_8h_data.py          # Step 1: Price + On-chain
│   ├── collect_sentiment_8h.py     # Step 2: Sentiment
│   ├── merge_all_data.py           # Step 3: Merge all
│   ├── compare_daily_vs_8h.py      # Compare models
│   ├── run_full_pipeline.py        # Run everything
│   │
│   ├── Model3_OnChain/             # Daily model (existing)
│   │   ├── model3_pipeline.py
│   │   └── processed_model3/
│   │
│   └── Model3_8H/                  # 8H model (new)
│       ├── train_model3_8h.py
│       └── processed_model3_8h/
│
├── btc_8h_full.csv                 # Price + on-chain
├── btc_8h_sentiment.csv            # Sentiment data
├── btc_8h_complete.csv             # Final merged dataset
│
└── backend/
    └── main.py                     # API server
```

---

## Quick Start (Automatic)

Run everything automatically (~20 minutes):

```bash
cd /home/shyam/UbuntuCode/CN\ 6000\ Mental\ Wealth\ Professional\ Life\ 3\ \(Project\)/Project

uv run python Mango/run_full_pipeline.py
```

This runs all 3 collection scripts + training + comparison automatically.

---

## Step-by-Step (Manual)

### Step 1: Collect Price + On-Chain Data (10 min)

```bash
cd /home/shyam/UbuntuCode/CN\ 6000\ Mental\ Wealth\ Professional\ Life\ 3\ \(Project\)/Project

uv run python Mango/collect_8h_data.py
```

**Output:** `btc_8h_full.csv`
- ~70,000 rows (8H since 2017)
- ~20 features (OHLCV + on-chain metrics)

**Sources:**
- Binance API (8H candles)
- Blockchain.com (tx count, hashrate, fees, mempool, miners revenue, difficulty)
- CoinMetrics (active addresses, supply, MVRV, etc.)

---

### Step 2: Collect Sentiment Data (2 min)

```bash
uv run python Mango/collect_sentiment_8h.py
```

**Output:** `btc_8h_sentiment.csv`
- ~2,000 rows (8H resampled)
- ~8 features (Fear & Greed, CFGI, Google Trends, Reddit)

**Sources:**
- Alternative.me (Fear & Greed Index)
- CFGI.io (Crypto sentiment score)
- Google Trends (search interest)
- Reddit (post sentiment via PRAW) ⚠️ optional

---

### Step 3: Merge All Data (30 sec)

```bash
uv run python Mango/merge_all_data.py
```

**Output:** `btc_8h_complete.csv`
- ~70,000 rows × 41 features
- All data sources combined
- Ready for training

---

### Step 4: Train 8H Model (5 min)

```bash
cd Mango/Model3_8H
uv run python train_model3_8h.py
```

**Output:**
- `processed_model3_8h/model_best.pth` (trained model)
- `processed_model3_8h/metrics.json` (accuracy, F1, etc.)
- `plots_8h/training_curves.png`
- `plots_8h/roc_curve.png`

---

### Step 5: Compare Models (instant)

```bash
cd ..
uv run python compare_daily_vs_8h.py
```

**Output:** Side-by-side comparison table

Example:
```
============================================================
PERFORMANCE COMPARISON
============================================================
Metric       Daily    8H
Accuracy     0.5234   0.5678  ← 8H wins
F1 Score     0.4821   0.5123  ← 8H wins
Val Loss     0.6234   0.5987  ← 8H wins
============================================================
🏆 Winner by Accuracy: 8H (+0.0444)
```

---

## Feature Breakdown (41 total)

### Price (13 features)
- OHLCV from Binance
- Returns, volatility, moving averages
- All engineered for ML

### On-Chain (~20 features)
**Blockchain.com:**
- Transaction count
- Hashrate
- Fees (BTC)
- Mempool size
- Miners revenue
- Difficulty

**CoinMetrics:**
- Active addresses
- Transaction count
- Supply metrics
- MVRV ratio
- Hash rate growth

### Sentiment (~8 features)
- Fear & Greed Index
- CFGI score
- Google Trends
- Reddit sentiment (optional)

---

## Expected Results

| Metric | Daily Model | 8H Model | Winner |
|--------|-------------|----------|--------|
| Accuracy | 52-55% | 54-58% | 8H |
| F1 Score | 0.48-0.52 | 0.50-0.55 | 8H |
| Training Time | 3 min | 5 min | Daily |
| Signals/Day | 1 | 3 | 8H |
| Noise Level | Low | Medium | Daily |

**Typical outcome:** 8H model wins on accuracy, but has more transaction costs.

---

## Deployment

### If 8H Wins:

Update `backend/main.py` line 23-24:

```python
# OLD (Daily)
# MANGO_DIR = Path(__file__).resolve().parent.parent / "Mango" / "Model3_OnChain"
# PROCESSED = MANGO_DIR / "processed_model3"

# NEW (8H)
MANGO_DIR = Path(__file__).resolve().parent.parent / "Mango" / "Model3_8H"
PROCESSED = MANGO_DIR / "processed_model3_8h"
```

Then restart API:
```bash
pkill -f uvicorn
cd backend
uv run uvicorn main:app --host 0.0.0.0 --port 8002
```

---

## Troubleshooting

### "No module named 'boruta'"
```bash
uv add boruta
```

### "FileNotFoundError: btc.csv"
CoinMetrics data needs to be downloaded or use existing data from `/data/`

### "Google Trends rate limited"
Wait 5 minutes, then re-run with fewer days:
```bash
python collect_sentiment_8h.py 90  # 90 days instead of 365
```

### "Reddit credentials not set"
Reddit is optional. To enable:
1. Go to `reddit.com/prefs/apps`
2. Create app (Script type)
3. Get client_id and client_secret
4. Edit `collect_sentiment_8h.py` lines 165-166

---

## Data Quality

| Source | Reliability | Latency | Notes |
|--------|-------------|---------|-------|
| Binance | ⭐⭐⭐⭐⭐ | Real-time | Perfect |
| Blockchain.com | ⭐⭐⭐⭐ | 1 hour | Good |
| CoinMetrics | ⭐⭐⭐⭐⭐ | Daily | Excellent |
| Alternative.me | ⭐⭐⭐⭐ | Daily | Good |
| CFGI.io | ⭐⭐⭐ | 15-min | Recent only |
| Google Trends | ⭐⭐⭐ | Daily | Some gaps |
| Reddit | ⭐⭐ | Real-time | Noisy |

---

## Files Created

| File | Description | Size |
|------|-------------|------|
| `btc_8h_full.csv` | Price + on-chain | ~10 MB |
| `btc_8h_sentiment.csv` | All sentiment | ~2 MB |
| `btc_8h_complete.csv` | **Final dataset** | ~12 MB |
| `feature_categories.json` | Feature groups | <1 KB |
| `Mango/Model3_8H/processed_model3_8h/model_best.pth` | Trained model | ~2 MB |
| `Mango/Model3_8H/processed_model3_8h/metrics.json` | Performance | <1 KB |

---

## Next Steps After Training

1. ✅ Review comparison results
2. ✅ Deploy better model to API
3. ✅ Test on live data
4. ✅ Monitor performance
5. ✅ Retrain monthly with fresh data

---

## Commands Reference

```bash
# Full automatic pipeline
uv run python Mango/run_full_pipeline.py

# Individual steps
uv run python Mango/collect_8h_data.py
uv run python Mango/collect_sentiment_8h.py
uv run python Mango/merge_all_data.py
uv run python Mango/Model3_8H/train_model3_8h.py
uv run python Mango/compare_daily_vs_8h.py

# Check API
curl http://localhost:8002/api/btc
```

---

## Questions?

**Q: Why 8H instead of hourly?**
A: 8H balances noise reduction with trading opportunities. Hourly is too noisy for sentiment data.

**Q: How often to retrain?**
A: Monthly for production. Weekly if using 8H model actively.

**Q: Which model should I use?**
A: Run the comparison script - it tells you the winner based on your data.

**Q: Can I use this for live trading?**
A: This is for research/educational purposes. Always backtest thoroughly before live deployment.

---

## Credits

- **Price Data:** Binance API
- **On-Chain:** Blockchain.com, CoinMetrics
- **Sentiment:** Alternative.me, CFGI.io, Google Trends, Reddit
- **Model:** Dual-Head Transformer (PyTorch)
- **Framework:** FastAPI for serving

---

**Ready to start? Run:** `uv run python Mango/run_full_pipeline.py` 🚀
