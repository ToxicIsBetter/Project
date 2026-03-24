"""
evaluate.py — Model Comparison Dashboard
Generates comparison table, confusion matrices, ROC curves, and trading simulation.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

DATA_DIR = Path('data')
PLOT_DIR = Path('plots')
PLOT_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. LOAD RESULTS
# ============================================================
print("📊 Loading model results...")

with open(DATA_DIR / 'baseline_results.json') as f:
    baselines = json.load(f)

models = {}
for res_file, name in [
    ('lstm_results.json', 'LSTM'),
    ('transformer_results.json', 'Transformer'),
    ('cnnlstm_results.json', 'CNN-LSTM Hybrid')
]:
    fpath = DATA_DIR / res_file
    if fpath.exists():
        with open(fpath) as f:
            models[name] = json.load(f)
        print(f"   ✅ Loaded {name}")
    else:
        print(f"   ⚠️ {name} results not found, skipping")

# Load Dual-Head
dp_path = DATA_DIR / 'dual_head_predictions.csv'
if dp_path.exists():
    dhd = pd.read_csv(dp_path)
    y_true, y_pred, y_prob = dhd['y_true'].values, dhd['y_pred'].values, dhd['y_prob'].values
    models['Dual-Head Transformer'] = {
        'name': 'Dual-Head Transformer',
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred)),
        'predictions': y_pred.tolist(),
        'probabilities': y_prob.tolist(),
        'labels': y_true.tolist()
    }
    print("   ✅ Loaded Dual-Head Transformer")

# ============================================================
# 2. COMPARISON TABLE
# ============================================================
all_results = baselines + [v for v in models.values()]

print(f"\n{'='*70}")
print(f"  MODEL COMPARISON — Test Set")
print(f"{'='*70}")
print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print(f"{'-'*70}")
for r in all_results:
    print(f"{r['name']:<25} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f}")

# Best model
best = max(all_results, key=lambda x: x['accuracy'])
print(f"\n🏆 Best model by accuracy: {best['name']} ({best['accuracy']:.4f})")

# ============================================================
# 3. COMPARISON BAR CHART
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
names = [r['name'] for r in all_results]
metrics = ['accuracy', 'precision', 'recall', 'f1']
x = np.arange(len(names))
width = 0.2

colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
for i, metric in enumerate(metrics):
    values = [r[metric] for r in all_results]
    bars = ax.bar(x + i * width, values, width, label=metric.capitalize(), color=colors[i], alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Bitcoin Direction Prediction — Model Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(names, rotation=15, ha='right')
ax.legend(loc='upper left')
ax.set_ylim(0, 1.05)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random chance')
plt.tight_layout()
plt.savefig(PLOT_DIR / 'model_comparison.png', dpi=150)
print(f"   📊 Saved: {PLOT_DIR / 'model_comparison.png'}")
plt.close()

# ============================================================
# 4. ROC CURVES (DL models only — need probabilities)
# ============================================================
if models:
    fig, ax = plt.subplots(figsize=(8, 8))
    colors_roc = ['darkorange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'blue', 'black', 'cyan', 'magenta', 'yellow', 'blue', 'black']
    
    for i, (name, data) in enumerate(models.items()):
        if 'probabilities' in data and 'labels' in data:
            fpr, tpr, _ = roc_curve(data['labels'], data['probabilities'])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors_roc[i], lw=2,
                    label=f'{name} (AUC = {roc_auc:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — Deep Learning Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'roc_curves.png', dpi=150)
    print(f"   📊 Saved: {PLOT_DIR / 'roc_curves.png'}")
    plt.close()

# ============================================================
# 5. CONFUSION MATRICES (DL models)
# ============================================================
if models:
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]
    
    for ax, (name, data) in zip(axes, models.items()):
        cm = confusion_matrix(data['labels'], data['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        ax.set_title(f'{name}', fontsize=13, fontweight='bold')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
    
    plt.suptitle('Confusion Matrices — Deep Learning Models', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
    print(f"   📊 Saved: {PLOT_DIR / 'confusion_matrices.png'}")
    plt.close()

# ============================================================
# 6. TRAINING LOSS CURVES
# ============================================================
if models:
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5))
    if len(models) == 1:
        axes = [axes]
    
    for ax, (name, data) in zip(axes, models.items()):
        if 'train_losses' in data:
            epochs = range(1, len(data['train_losses']) + 1)
            ax.plot(epochs, data['train_losses'], label='Train Loss', color='#2196F3', lw=2)
            ax.plot(epochs, data['val_losses'], label='Val Loss', color='#E91E63', lw=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'{name} — Training Curves', fontweight='bold')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'training_curves.png', dpi=150)
    print(f"   📊 Saved: {PLOT_DIR / 'training_curves.png'}")
    plt.close()

# ============================================================
# 7. TRADING SIMULATION
# ============================================================
test_df = pd.read_csv(DATA_DIR / 'btc_test.csv', parse_dates=['date'], index_col='date')
if models:
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Buy & Hold
    try:
        full_df = pd.read_csv(DATA_DIR / 'btc_full_features.csv', parse_dates=['Date'], index_col='Date')
        returns = full_df.loc[test_df.index, 'Close'].pct_change().fillna(0)
    except Exception:
        # Fallback if Close is miraculously retained or we only have return_1d
        if 'Close' in test_df.columns:
            returns = test_df['Close'].pct_change().fillna(0)
        elif 'return_1d' in test_df.columns:
            returns = test_df['return_1d'].fillna(0)
        else:
            returns = pd.Series(np.zeros(len(test_df)), index=test_df.index)
            
    buy_hold = (1 + returns).cumprod()
    ax.plot(test_df.index, buy_hold, label='Buy & Hold', color='gray', lw=2, alpha=0.7)
    
    for name, data in models.items():
        if 'predictions' in data:
            preds = np.array(data['predictions'])
            # Align predictions with test data (account for sequence length offset)
            aligned_returns = returns.values[-len(preds):]
            # Strategy: if model predicts up, be long; if down, be flat
            strategy_returns = aligned_returns * preds
            strategy_equity = (1 + strategy_returns).cumprod()
            dates = test_df.index[-len(preds):]
            ax.plot(dates, strategy_equity, label=f'{name} Strategy', lw=2)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Equity (starting $1)', fontsize=12)
    ax.set_title('Trading Simulation — Model Strategies vs Buy & Hold', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'trading_simulation.png', dpi=150)
    print(f"   📊 Saved: {PLOT_DIR / 'trading_simulation.png'}")
    plt.close()

# ============================================================
# 8. FEATURE IMPORTANCE (from baselines)
# ============================================================
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    train = pd.read_csv(DATA_DIR / 'btc_train.csv', parse_dates=['date'], index_col='date')
    with open(DATA_DIR / 'feature_cols.txt') as f:
        feature_cols = f.read().strip().split('\n')
    
    X = np.nan_to_num(train[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y = train['target'].values
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    importances.plot(kind='barh', ax=ax, color='#2196F3', alpha=0.85)
    ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'feature_importance.png', dpi=150)
    print(f"   📊 Saved: {PLOT_DIR / 'feature_importance.png'}")
    plt.close()
except Exception as e:
    print(f"   ⚠️ Feature importance plot failed: {e}")

# ============================================================
# 9. SAVE FINAL SUMMARY
# ============================================================
summary = {
    'all_results': all_results,
    'best_model': best['name'],
    'best_accuracy': best['accuracy'],
}
with open(DATA_DIR / 'final_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n🎉 EVALUATION COMPLETE!")
print(f"   All plots saved to {PLOT_DIR}/")
print(f"   Summary saved to {DATA_DIR / 'final_summary.json'}")
