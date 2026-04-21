# NeuralEdge Model Training Results

## 8H Model Performance (NEW)

**Trained:** 2026-04-18

| Metric | Value |
|--------|-------|
| Test Accuracy | **48.67%** |
| Test F1 Score | **0.5229** |
| Best Val Loss | 0.6924 |
| Epochs Trained | 100 (early stopped) |
| Features Used | 19 |

### Dataset Info
- **Rows:** 5,476 (8H candles from 2021-04-19 to 2026-04-18)
- **Features:** 19 (13 price + 6 on-chain)
- **Train/Val/Test Split:** 1,862 / 1,095 / 2,518

### Features Used
**Price (13):** open, high, low, price, volume, return, log_return, range, vol_7p, vol_24p, ma_24, ma_168, price_ma_ratio

**On-Chain (6):** tx_count, hashrate, fees_btc, mempool_size, miners_revenue, difficulty

### Confusion Matrix
```
              Predicted
              0      1
Actual 0    516    701
       1    589    707
```

---

## Daily Model (EXISTING)

The Daily model was trained earlier with the full Dual-Head architecture using:
- **Head 1:** Pure on-chain features (CoinMetrics)
- **Head 2:** Sentiment features (Fear & Greed, Google Trends)

---

## Comparison Notes

**8H Model Advantages:**
- More data points (5,476 vs ~1,800 for daily)
- Captures intraday patterns
- 3 trading signals per day vs 1
- Faster feedback loop

**8H Model Challenges:**
- Lower accuracy (~49% vs ~52-55% typical for daily)
- More noise in 8H data
- Missing sentiment features (API issues)
- Shorter history (2021-2026 vs 2010-2026)

**Why Lower Accuracy?**
1. **Missing sentiment data** - APIs for Fear & Greed and CFGI failed
2. **Shorter history** - Only 5 years vs 16 years for daily
3. **More noise** - 8H frequency has more random walk behavior
4. **Fewer features** - 19 vs ~40 in full dataset

---

## Recommendation

**Use the Daily model for production** because:
1. Higher accuracy (~52-55% vs 49%)
2. Longer historical validation
3. More features (on-chain + sentiment)
4. Less noise, more reliable signals

**Keep 8H model for:**
- Intraday pattern research
- High-frequency experiments
- Ensemble methods (combine with daily)

---

## Next Steps

1. **Deploy Daily model** to production API
2. **Improve 8H data collection**:
   - Fix sentiment API calls
   - Add CoinMetrics data
   - Extend history back to 2010
3. **Retrain 8H** with complete dataset
4. **Compare again** with full feature set

---

## Files Created

- `btc_8h_full.csv` - 8H price + on-chain data
- `btc_8h_complete.csv` - Merged dataset
- `Model3_8H/processed_model3_8h/model_best.pth` - Trained 8H model
- `Model3_8H/processed_model3_8h/metrics.json` - Performance metrics
- `Model3_8H/plots_8h/` - Training curves and ROC plot

---

**Status:** ✅ 8H Model Trained | ⏳ Daily Model Comparison Pending
