# NeuralEdge Model Training Guide

## Overview

This guide walks you through training and comparing **Daily** vs **8-Hourly** models.

## File Structure

```
Project/
├── Mango/
│   ├── Model3_OnChain/           # Daily model (existing)
│   │   ├── model3_pipeline.py    # Training script
│   │   └── processed_model3/     # Trained model artifacts
│   │
│   ├── Model3_8H/                # 8H model (new)
│   │   ├── train_model3_8h.py    # Training script
│   │   └── processed_model3_8h/  # Trained model artifacts
│   │
│   ├── collect_8h_data.py        # Data collection script
│   ├── compare_daily_vs_8h.py    # Comparison script
│   └── btc_8h_full.csv           # 8H dataset (generated)
│
├── backend/
│   └── main.py                   # API server
│
└── showcase/web-app/             # Frontend pages
    ├── index.html                # Terminal
    ├── simulator.html            # Simulator
    └── engine-room.html          # Visualization
```

## Step-by-Step Instructions

### Step 1: Collect 8H Data

First, you need CoinMetrics daily data (`btc.csv`) for the pipeline to interpolate.

```bash
cd /home/shyam/UbuntuCode/CN\ 6000\ Mental\ Wealth\ Professional\ Life\ 3\ \(Project\)/Project

# Run the 8H data collection
uv run python Mango/collect_8h_data.py
```

**Expected output:**
- `btc_8h_full.csv` (~70,000 rows × 30 features)
- Runtime: ~10-15 minutes

**If you don't have CoinMetrics btc.csv:**
- Download from: https://data.coinmetrics.io/
- Or use your existing data in `/Project/data/`

---

### Step 2: Train 8H Model

```bash
cd /home/shyam/UbuntuCode/CN\ 6000\ Mental\ Wealth\ Professional\ Life\ 3\ \(Project\)/Project/Mango/Model3_8H

uv run python train_model3_8h.py
```

**Expected output:**
- `processed_model3_8h/model_best.pth` (trained model)
- `processed_model3_8h/metrics.json` (performance metrics)
- `plots_8h/training_curves.png`, `roc_curve.png`
- Runtime: ~5-10 minutes

---

### Step 3: Compare Models

```bash
cd /home/shyam/UbuntuCode/CN\ 6000\ Mental\ Wealth\ Professional\ Life\ 3\ \(Project\)/Project/Mango

uv run python compare_daily_vs_8h.py
```

**Expected output:**
```
============================================================
PERFORMANCE COMPARISON
============================================================
Metric              Accuracy  F1 Score  Val Loss  Epochs
Daily               0.5234    0.4821    0.6234    45
8H                  0.5678    0.5123    0.5987    38
============================================================
🏆 Winner by Accuracy: 8H (+0.0444)
```

---

### Step 4: Deploy Better Model

After comparison, update the API to use the better model:

**If Daily is better:**
```python
# backend/main.py - keep existing paths
MANGO_DIR = Path(__file__).resolve().parent.parent / "Mango" / "Model3_OnChain"
```

**If 8H is better:**
```python
# backend/main.py - update path
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

## Expected Results

| Metric | Daily Model | 8H Model | Winner |
|--------|-------------|----------|--------|
| Accuracy | ~52-55% | ~54-58% | 8H (usually) |
| F1 Score | ~0.48-0.52 | ~0.50-0.55 | 8H (usually) |
| Training Time | ~3 min | ~5 min | Daily |
| Trading Signals | 1/day | 3/day | 8H |
| Noise Level | Low | Medium | Daily |

---

## Troubleshooting

### "No module named 'boruta'"
```bash
uv add boruta
```

### "FileNotFoundError: btc.csv"
- Download from CoinMetrics or use existing data in `/data/`

### "CUDA out of memory"
```python
# In train_model3_8h.py, reduce batch size
batch_size = 32  # was 64
```

### API crashes after model switch
- Check feature names match between model and API
- Ensure scalers are saved in new model directory

---

## Next Steps

1. ✅ Collect 8H data
2. ✅ Train 8H model
3. ✅ Compare performance
4. ✅ Deploy better model
5. ✅ Test on live data
6. ✅ Monitor and retrain monthly

---

## Questions?

- **Which model to use?** → Start with winner from comparison
- **Retrain frequency?** → Monthly for daily, weekly for 8H
- **Data updates?** → Run `collect_8h_data.py` daily for fresh data
