import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
import os

# Set paths
M1_DIR = 'GoogleTrends/processed_model1'
M2_DIR = 'CleanSentiment/processed'
PLOTS_DIR = 'comparison_plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load Model 1 Predictions (Baseline 2018+)
m1_probs = np.load(f'{M1_DIR}/model1_probs_test.npy')
m1_preds = np.load(f'{M1_DIR}/model1_preds_test.npy')
m1_ytrue = np.load(f'{M1_DIR}/model1_ytrue_test.npy')

# Load Model 2 Predictions (Full Enriched 2013+)
m2_probs = np.load(f'{M2_DIR}/model2_probs_test.npy')
m2_preds = np.load(f'{M2_DIR}/model2_preds_test.npy')
m2_ytrue = np.load(f'{M2_DIR}/model2_ytrue_test.npy')

# Verify target arrays are identical
assert np.array_equal(m1_ytrue, m2_ytrue), "Test sets do not match! Cannot compare."
y_true = m1_ytrue

# 1. Classification Metrics
def get_metrics(y_true, preds, probs):
    return {
        'Accuracy': accuracy_score(y_true, preds),
        'F1 Score': f1_score(y_true, preds),
        'Precision': precision_score(y_true, preds),
        'Recall': recall_score(y_true, preds),
        'AUC-ROC': roc_auc_score(y_true, probs)
    }

m1_metrics = get_metrics(y_true, m1_preds, m1_probs)
m2_metrics = get_metrics(y_true, m2_preds, m2_probs)

print("="*50)
print("CLASSIFICATION METRICS (Jan 2024 - Mar 2026)")
print("="*50)
print(f"{'Metric':<12} | {'Model 1 (Baseline)':<18} | {'Model 2 (Enriched)':<18}")
print("-" * 55)
for k in m1_metrics.keys():
    print(f"{k:<12} | {m1_metrics[k]:<18.4f} | {m2_metrics[k]:<18.4f}")

# 2. Confusion Matrices
m1_cm = confusion_matrix(y_true, m1_preds)
m2_cm = confusion_matrix(y_true, m2_preds)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(m1_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title('Model 1 (Baseline) Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_xticklabels(['Down', 'Up'])
axes[0].set_yticklabels(['Down', 'Up'])

sns.heatmap(m2_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1], cbar=False)
axes[1].set_title('Model 2 (Enriched) Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_xticklabels(['Down', 'Up'])
axes[1].set_yticklabels(['Down', 'Up'])

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/confusion_matrices.png', dpi=150)
plt.close()

# 3. ROC Curves
fpr1, tpr1, _ = roc_curve(y_true, m1_probs)
fpr2, tpr2, _ = roc_curve(y_true, m2_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, label=f'Model 1 (AUC = {m1_metrics["AUC-ROC"]:.3f})', color='blue')
plt.plot(fpr2, tpr2, label=f'Model 2 (AUC = {m2_metrics["AUC-ROC"]:.3f})', color='darkorange')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison (Test Set)')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/roc_curves.png', dpi=150)
plt.close()

# 4. Trading Simulation
# Load the actual price data to calculate daily returns
ohlcv = pd.read_csv('../BTC_Price_in_2026/ohlcv_2010_to_now.csv', parse_dates=['Date'])
# The test set is the last N rows
# Because sequence length is 5, the test split shifted by 5. 
# We need exactly the dates corresponding to the test labels.
# In model pipeline: val_end='2023-12-31'. test = df[date > val_end].
# Then y_true corresponds to test indices [SEQ_LEN:]
test_df = ohlcv[ohlcv['Date'] > '2023-12-31'].copy().reset_index(drop=True)
test_df = test_df.iloc[5:5+len(y_true)].reset_index(drop=True)

# Calculate daily returns: (Close[t+1] - Close[t]) / Close[t]
# Since our targets predict if Close[t+1] > Close[t], buying at Close[t] yields this return at Close[t+1].
test_df['daily_return'] = test_df['Close'].shift(-1) / test_df['Close'] - 1
test_df.loc[test_df.index[-1], 'daily_return'] = 0 # No next day for last row

# Baseline Buy and Hold
test_df['Buy_Hold_Return'] = (1 + test_df['daily_return']).cumprod()

# Strategy: if pred==1, buy/hold (1x return). If pred==0, cash (0x return).
test_df['M1_Strategy_Return'] = (1 + test_df['daily_return'] * m1_preds).cumprod()
test_df['M2_Strategy_Return'] = (1 + test_df['daily_return'] * m2_preds).cumprod()

plt.figure(figsize=(10, 6))
plt.plot(test_df['Date'], test_df['Buy_Hold_Return'], label='Buy & Hold Baseline', color='black', alpha=0.6)
plt.plot(test_df['Date'], test_df['M1_Strategy_Return'], label='Model 1 Strategy', color='blue')
plt.plot(test_df['Date'], test_df['M2_Strategy_Return'], label='Model 2 Strategy', color='darkorange')
plt.title('Simulated Cumulative Returns ($1 Invested Jan 2024)')
plt.ylabel('Cumulative Return Multiplier')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/trading_simulation.png', dpi=150)
plt.close()

print("\n" + "="*50)
print("TRADING SIMULATION ($10,000 Invested)")
print("="*50)
bh_final = test_df['Buy_Hold_Return'].iloc[-1] * 10000
m1_final = test_df['M1_Strategy_Return'].iloc[-1] * 10000
m2_final = test_df['M2_Strategy_Return'].iloc[-1] * 10000

print(f"Buy & Hold:             ${bh_final:,.2f}")
print(f"Model 1 (Baseline):     ${m1_final:,.2f}")
print(f"Model 2 (Enriched):     ${m2_final:,.2f}")
print("\nPlots saved to Mango/comparison_plots/")
