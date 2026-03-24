"""
baselines.py — Baseline Models for BTC Direction Prediction
Establishes performance floor that deep learning models must beat.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')

# ============================================================
# 1. LOAD DATA
# ============================================================
print("📊 Loading data...")
train = pd.read_csv(DATA_DIR / 'btc_train.csv', parse_dates=['date'], index_col='date')
val = pd.read_csv(DATA_DIR / 'btc_val.csv', parse_dates=['date'], index_col='date')
test = pd.read_csv(DATA_DIR / 'btc_test.csv', parse_dates=['date'], index_col='date')

with open(DATA_DIR / 'feature_cols.txt') as f:
    feature_cols = f.read().strip().split('\n')

X_train, y_train = train[feature_cols].values, train['target'].values
X_val, y_val = val[feature_cols].values, val['target'].values
X_test, y_test = test[feature_cols].values, test['target'].values

# Replace inf with nan, then fill
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"   Features: {len(feature_cols)}")

# ============================================================
# 2. BASELINES
# ============================================================
def evaluate(name, y_true, y_pred):
    """Print and return metrics for a model."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    return {'name': name, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

results = []

# --- Baseline 1: Random Guess ---
np.random.seed(42)
y_random = np.random.randint(0, 2, size=len(y_test))
results.append(evaluate("Random Guess", y_test, y_random))

# --- Baseline 2: Always Predict Up ---
y_up = np.ones(len(y_test), dtype=int)
results.append(evaluate("Always Up", y_test, y_up))

# --- Baseline 3: Logistic Regression ---
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
y_lr = lr.predict(X_test_s)
results.append(evaluate("Logistic Regression", y_test, y_lr))

# --- Baseline 4: Random Forest ---
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)
y_rf = rf.predict(X_test_s)
results.append(evaluate("Random Forest", y_test, y_rf))

# --- Baseline 5: Gradient Boosting ---
gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
gb.fit(X_train_s, y_train)
y_gb = gb.predict(X_test_s)
results.append(evaluate("Gradient Boosting", y_test, y_gb))

# ============================================================
# 3. SUMMARY TABLE
# ============================================================
print(f"\n{'='*60}")
print(f"  BASELINE COMPARISON (Test Set)")
print(f"{'='*60}")
print(f"{'Model':<25} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
print(f"{'-'*60}")
for r in results:
    print(f"{r['name']:<25} {r['accuracy']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f} {r['f1']:>8.4f}")

# Save results
with open(DATA_DIR / 'baseline_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Baseline results saved to {DATA_DIR / 'baseline_results.json'}")
