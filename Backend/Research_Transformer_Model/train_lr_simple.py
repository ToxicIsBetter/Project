"""Simple Logistic Regression for 50%+ accuracy"""
import numpy as np, pandas as pd, json, os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib

print("="*60)
print("Logistic Regression Baseline")
print("="*60)

# Load
ohlcv = pd.read_csv("../../data/train_ohlcv.csv", parse_dates=["Date"])
onchain = pd.read_csv("../../data/train_onchain.csv", parse_dates=["Date"])

df = ohlcv.merge(onchain, on="Date", how="left")
df = df[df['Date'] >= '2018-02-01'].reset_index(drop=True)
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.iloc[:-1].reset_index(drop=True)

# Features
features = ['Close', 'Volume', 'AdrActCnt', 'TxCnt', 'HashRate', 'BlkCnt', 'CapMVRVCur']
features = [f for f in features if f in df.columns]
df = df[features + ['target']].dropna()

print(f"\nData: {len(df)} rows")
print(f"Up: {(df['target']==1).sum()} ({100*(df['target']==1).mean():.1f}%)")
print(f"Down: {(df['target']==0).sum()} ({100*(df['target']==0).mean():.1f}%)")

# Prepare
X = df[features].values
y = df['target'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
n = len(X_scaled)
train_end, val_end = int(n*0.70), int(n*0.85)
X_train, X_val, X_test = X_scaled[:train_end], X_scaled[train_end:val_end], X_scaled[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Train with different C
print("\nTraining...")
best_acc, best_c, best_m = 0, 0, None
for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    m = LogisticRegression(C=C, max_iter=2000, random_state=42)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_val, m.predict(X_val))
    print(f"  C={C:4.0f} | Val Acc: {acc:.4f}")
    if acc > best_acc:
        best_acc, best_c, best_m = acc, C, m

print(f"\nBest C: {best_c}, Val Acc: {best_acc:.4f}")

# Test
pred = best_m.predict(X_test)
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred)

print(f"\nTest Accuracy: {acc:.4f}")
print(f"Test F1: {f1:.4f}")
print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, pred)
print(cm)
print(f"\nClassification Report:")
print(classification_report(y_test, pred))

# Save
joblib.dump(best_m, 'processed_model3/lr_model.pkl')
joblib.dump(scaler, 'processed_model3/lr_scaler.pkl')
with open('processed_model3/lr_metrics.json', 'w') as f:
    json.dump({'accuracy': float(acc), 'f1': float(f1), 'C': best_c, 'val_acc': float(best_acc)}, f, indent=2)

print("\n✅ Done! Model saved.")
