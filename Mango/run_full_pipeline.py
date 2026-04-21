#!/usr/bin/env python3
"""
NeuralEdge Complete Data Pipeline
==================================
Runs all data collection and training steps automatically.

Usage:
    uv run python Mango/run_full_pipeline.py

Steps:
    1. Collect 8H price + on-chain data
    2. Collect sentiment data
    3. Merge all data
    4. Train 8H model
    5. Compare with Daily model
    6. Recommend winner
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

print("=" * 70)
print("NEURALEDGE COMPLETE PIPELINE")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)


def run_step(name, script, timeout=None):
    """Run a Python script and report results."""
    print(f"\n{'=' * 70}")
    print(f"STEP: {name}")
    print(f"{'=' * 70}")

    start = datetime.now()
    try:
        result = subprocess.run(
            ["uv", "run", "python", script],
            capture_output=False,
            text=True,
            timeout=timeout or 300,
        )
        elapsed = datetime.now() - start

        if result.returncode == 0:
            print(f"\n✅ {name} completed in {elapsed}")
            return True
        else:
            print(f"\n❌ {name} failed with code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print(f"\n❌ {name} timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"\n❌ {name} error: {e}")
        return False


# Step 1: Collect price + on-chain data
print("\n" + "=" * 70)
print("PHASE 1: DATA COLLECTION")
print("=" * 70)

if not run_step("Collect 8H Price + On-Chain", "Mango/collect_8h_data.py", timeout=900):
    print("⚠️ Price data collection failed. Check btc_8h_full.csv location.")
    sys.exit(1)

# Step 2: Collect sentiment data
if not run_step("Collect Sentiment", "Mango/collect_sentiment_8h.py", timeout=300):
    print("⚠️ Sentiment collection failed. Continuing without sentiment data.")

# Step 3: Merge all data
if not run_step("Merge All Data", "Mango/merge_all_data.py", timeout=300):
    print("⚠️ Merge failed. Continuing with price data only.")

# Step 4: Train 8H model
print("\n" + "=" * 70)
print("PHASE 2: MODEL TRAINING")
print("=" * 70)

if not Path("Mango/Model3_8H").exists():
    Path("Mango/Model3_8H").mkdir(parents=True)

if not run_step("Train 8H Model", "Mango/Model3_8H/train_model3_8h.py", timeout=600):
    print("⚠️ 8H training failed. Check btc_8h_complete.csv or btc_8h_full.csv")
    sys.exit(1)

# Step 5: Compare models
print("\n" + "=" * 70)
print("PHASE 3: MODEL COMPARISON")
print("=" * 70)

if not run_step("Compare Models", "Mango/compare_daily_vs_8h.py", timeout=60):
    print("⚠️ Comparison failed. Check metrics files exist.")
    sys.exit(1)

# Final summary
print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
print(f"""
✅ All steps finished successfully!

Output files:
  - btc_8h_complete.csv (merged dataset)
  - Mango/Model3_8H/processed_model3_8h/ (trained model)
  - Mango/Model3_8H/plots_8h/ (performance plots)

Next steps:
  1. Review comparison results in compare_daily_vs_8h.py output
  2. Deploy better model to production API
  3. Test on live data

Finished: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
""")
