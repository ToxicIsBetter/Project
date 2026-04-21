"""Momentum-based prediction - uses price trends which ARE predictable"""
import numpy as np, pandas as pd, json, os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib

print("="*60)
print("MOMENTUM MODEL - Price Trends")
print("="*60)

# Load
ohlcv = pd.read_csv("../../data/train_ohlcv.csv", parse_dates=["Date"])
onchain = pd.read_csv("../../data/train_onchain.csv", parse_dates=["Date"])

df = ohlcv.merge(onchain, on="Date", how="left")
df = df[df['Date'] >= '2018-02-01'].reset_index(drop=True)

# Create momentum features
df['return_1d'] = df['Close'].pct_change()
df['return_3d'] = df['Close'].pct_change(3)
df['return_7d'] = df['Close'].pct_change(7)
df['return_14d'] = df['Close'].pct_change(14)
df['return_30d'] = df['Close'].pct_change(30)

# Volume change
df['vol_change'] = df['Volume'].pct_change()

# On-chain momentum
df['addr_change'] = df['AdrActCnt'].pct_change(7)
df['tx_change'] = df['TxCnt'].pct_change(7)

# Target: 3-day forward return (simpler than 5-day)
HORIZON = 3
df['target'] = (df['Close'].shift(-HORIZON) > df['Close']).astype(int)
df = df.iloc[:-HORIZON].reset_index(drop=True)
df = df.dropna()

print(f"Data: {len(df)} rows")
print(f"Target: Price higher in {HORIZON} days")
print(f"Class balance: Up={df['target'].mean():.2f}, Down={1-df['target'].mean():.2f}")

# Features - MOMENTUM ONLY (no raw price)
features = ['return_1d', 'return_3d', 'return_7d', 'return_14d', 'return_30d', 
            'vol_change', 'addr_change', 'tx_change']
features = [f for f in features if f in df.columns and df[f].isna().sum() < len(df)*0.5]

print(f"Features: {features}")

X = df[features].fillna(0).values
y = df['target'].values

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
n = len(X_scaled)
train_end, val_end = int(n*0.70), int(n*0.85)
X_train, X_val, X_test = X_scaled[:train_end], X_scaled[train_end:val_end], X_scaled[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Random Forest (better for momentum patterns)
print("\nTraining Random Forest...")
best_acc, best_m, best_n = 0, None, 0
for n_est in [50, 100, 200]:
    m = RandomForestClassifier(n_estimators=n_est, max_depth=5, random_state=42, 
                               class_weight='balanced', n_jobs=-1)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_val, m.predict(X_val))
    print(f"  n_estimators={n_est:3d} | Val Acc: {acc:.4f}")
    if acc > best_acc:
        best_acc, best_m, best_n = acc, m, n_est

print(f"\nBest: n_estimators={best_n}, Val Acc: {best_acc:.4f}")

# Test
pred = best_m.predict(X_test)
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred)

print(f"\n{'='*40}")
print("TEST RESULTS")
print(f"{'='*40}")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, pred)
print(cm)
print(f"\nClassification Report:")
print(classification_report(y_test, pred))

# Feature importance
fi = pd.DataFrame({'feature': features, 'importance': best_m.feature_importances_})
fi = fi.sort_values('importance', ascending=False)
print(f"\nFeature Importance:")
print(fi)

# Save
joblib.dump(best_m, 'processed_model3/momentum_model.pkl')
joblib.dump(scaler, 'processed_model3/momentum_scaler.pkl')
with open('processed_model3/momentum_metrics.json', 'w') as f:
    json.dump({'accuracy': float(acc), 'f1': float(f1), 'horizon': HORIZON}, f, indent=2)

print("\n✅ MOMENTUM MODEL TRAINED!")
