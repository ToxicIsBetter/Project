"""
Compare Daily vs 8H Model Performance
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
DAILY_METRICS = "Model3_OnChain/processed_model3/metrics.json"
DAILY_MODEL_DIR = "Model3_OnChain/processed_model3"

_8H_METRICS = "Model3_8H/processed_model3_8h/metrics.json"
_8H_MODEL_DIR = "Model3_8H/processed_model3_8h"

print("=" * 60)
print("DAILY vs 8H MODEL COMPARISON")
print("=" * 60)

# Load metrics
daily_metrics = None
if Path(DAILY_METRICS).exists():
    with open(DAILY_METRICS) as f:
        daily_metrics = json.load(f)
    print(f"\n✅ Daily model metrics loaded")
else:
    print(f"\n⚠️ Daily metrics not found: {DAILY_METRICS}")

h8_metrics = None
if Path(_8H_METRICS).exists():
    with open(_8H_METRICS) as f:
        h8_metrics = json.load(f)
    print(f"✅ 8H model metrics loaded")
else:
    print(f"⚠️ 8H metrics not found: {_8H_METRICS}")

if not daily_metrics or not h8_metrics:
    print("\n❌ Cannot compare - missing metrics files")
    exit(1)

# Comparison table
print("\n" + "=" * 60)
print("PERFORMANCE COMPARISON")
print("=" * 60)

comparison = pd.DataFrame(
    {
        "Metric": ["Accuracy", "F1 Score", "Best Val Loss", "Epochs Trained"],
        "Daily": [
            f"{daily_metrics.get('accuracy', 'N/A'):.4f}"
            if isinstance(daily_metrics.get("accuracy"), float)
            else "N/A",
            f"{daily_metrics.get('f1', 'N/A'):.4f}"
            if isinstance(daily_metrics.get("f1"), float)
            else "N/A",
            f"{daily_metrics.get('best_val_loss', 'N/A'):.4f}"
            if isinstance(daily_metrics.get("best_val_loss"), float)
            else "N/A",
            str(daily_metrics.get("epochs_trained", "N/A")),
        ],
        "8H": [
            f"{h8_metrics.get('accuracy', 'N/A'):.4f}"
            if isinstance(h8_metrics.get("accuracy"), float)
            else "N/A",
            f"{h8_metrics.get('f1', 'N/A'):.4f}"
            if isinstance(h8_metrics.get("f1"), float)
            else "N/A",
            f"{h8_metrics.get('best_val_loss', 'N/A'):.4f}"
            if isinstance(h8_metrics.get("best_val_loss"), float)
            else "N/A",
            str(h8_metrics.get("epochs_trained", "N/A")),
        ],
    }
)

print("\n" + comparison.to_string(index=False))
print("\n" + "=" * 60)

# Feature comparison
daily_features = len(daily_metrics.get("feature_cols", []))
h8_features = len(h8_metrics.get("feature_cols", []))

print(f"\nFEATURE COMPARISON")
print(f"  Daily model features: {daily_features}")
print(f"  8H model features:    {h8_features}")

# Determine winner
print("\n" + "=" * 60)
print("WINNER ANALYSIS")
print("=" * 60)

try:
    daily_acc = daily_metrics.get("accuracy", 0)
    h8_acc = h8_metrics.get("accuracy", 0)

    if daily_acc > h8_acc:
        winner = "DAILY"
        diff = daily_acc - h8_acc
    else:
        winner = "8H"
        diff = h8_acc - daily_acc

    print(f"\n🏆 Winner by Accuracy: {winner} (+{diff:.4f})")
    print(f"   Daily: {daily_acc:.4f} | 8H: {h8_acc:.4f}")

except Exception as e:
    print(f"Could not determine winner: {e}")

print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)

try:
    if h8_acc > daily_acc:
        print("\n✅ Use 8H model for production")
        print("   - Higher frequency = more trading opportunities")
        print("   - Better at capturing intraday patterns")
        print("   - More responsive to market changes")
    else:
        print("\n✅ Use Daily model for production")
        print("   - More stable predictions")
        print("   - Less noise from intraday volatility")
        print("   - Lower transaction costs")
except:
    print("\n⚠️ Could not make recommendation")

print("\n" + "=" * 60)
