# FINAL_STRAW Folder Contents

## 📁 Complete File List

### Models & Weights (2.7 MB)
- `model3_best.pt` - Fine-tuned Dual-Head Transformer weights
- `scaler_head1.pkl` - Head 1 feature scaler
- `scaler_head2_minmax.pkl` - Head 2 MinMax scaler
- `scaler_head2_std.pkl` - Head 2 Standard scaler
- `feature_sets.json` - Feature configuration
- `metrics_finetuned.json` - Performance metrics
- `momentum_model.pkl` - Bonus: Momentum RF model
- `multiday_model.pkl` - Bonus: Multi-day model
- `lr_model.pkl` - Bonus: Logistic Regression model
- `scaler_baseline.pkl` - Baseline scaler

### Data Files (9.0 MB)
- `clean_ohlcv.csv` - OHLCV price data (2010-2026)
- `clean_onchain.csv` - On-chain metrics (87 features)
- `clean_sentiment.csv` - Fear & Greed Index
- `clean_google.csv` - Google Trends data

### Scripts
- `scripts/predict.py` - Main prediction script
- `verify_installation.py` - Installation verification

### Documentation
- `README.md` - User guide and quick start
- `MODEL_SUMMARY.md` - Comprehensive model documentation
- `CONTENTS.md` - This file
- `requirements.txt` - Python dependencies

### Plots
- `plots/training_curves_finetuned.png` - Training visualization
- `plots/roc_curve_finetuned.png` - ROC curve

### Utilities
- `run_prediction.sh` - Quick start script

## 🎯 Model Performance Summary

| Metric | Value | Status |
|--------|-------|--------|
| Accuracy | 57.46% | ✅ Above 50% |
| F1 Score | 0.5398 | ✅ Above 0.50 |
| Recall (Up) | 65% | ✅ Balanced |
| Recall (Down) | 53% | ✅ Balanced |
| Threshold | 0.27 | ✅ Calibrated |

## 🚀 Quick Start Guide

### 1. Verify Installation
```bash
cd FINAL_STRAW
python verify_installation.py
```

### 2. Run Prediction
```bash
bash run_prediction.sh
```

### 3. Python API
```python
from scripts.predict import BitcoinPredictor

predictor = BitcoinPredictor(model_dir='models')
predictor.load_model()

import pandas as pd
df = pd.read_csv('data/clean_ohlcv.csv', parse_dates=['Date'])
pred, prob = predictor.predict(df, return_probs=True)

print(f"Direction: {'UP' if pred == 1 else 'DOWN'}")
print(f"Probability: {prob:.2%}")
```

## 📊 What's Included

### Core Model Files
- ✅ Trained model weights (2.3 MB)
- ✅ Feature scalers (3 files)
- ✅ Feature configuration
- ✅ Performance metrics

### Supporting Files
- ✅ Clean data files (4 CSV files, 9 MB total)
- ✅ Prediction script with examples
- ✅ Training visualizations
- ✅ Comprehensive documentation

### Bonus Models
- ✅ Momentum-based Random Forest
- ✅ Multi-day prediction model
- ✅ Logistic Regression baseline

## 📝 Model Details

### Architecture
- **Type**: Dual-Head Transformer
- **Sequence Length**: 5 days
- **Head 1 Features**: 13 (On-chain + Momentum)
- **Head 2 Features**: 12 (Sentiment)
- **Total Parameters**: ~568K

### Training
- **Train/Val/Test**: 70% / 15% / 15%
- **Epochs**: 69 (early stopped)
- **Best Val Loss**: 0.6416
- **Validation F1**: 0.7113

### Fine-Tuning Techniques
1. Soft class weighting (sqrt scaling)
2. Threshold calibration (0.27)
3. Momentum features (5 added)
4. Enhanced dropout (0.30)
5. LASSO feature selection

## ✅ Production Ready

All required files are included for:
- ✅ Model deployment
- ✅ Inference on new data
- ✅ Performance monitoring
- ✅ Reproducibility
- ✅ Documentation

## 📞 Support

Refer to:
- `README.md` for usage instructions
- `MODEL_SUMMARY.md` for technical details
- `verify_installation.py` for troubleshooting

---

**Status**: Production Ready ✅  
**Version**: Fine-Tuned v1.0  
**Date**: 2026-04-19
