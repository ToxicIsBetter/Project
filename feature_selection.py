"""
feature_selection.py — Academic Feature Selection
Matches Dubey & Enke (2025) methodology precisely:
Applies Boruta feature selection (Random Forest wrapper)
to extract the confirmed predictive features from the 113+ feature pool.
"""
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from boruta import BorutaPy
from pathlib import Path

# Fix deprecation warning from numpy in boruta
np.bool = bool
np.int = int
np.float = float

warnings.filterwarnings('ignore')

DATA_DIR = Path('data')

print("=== RUNNING BORUTA FEATURE SELECTION ===")

# 1. Load the massive dataset
df = pd.read_csv(DATA_DIR / 'btc_full_features.csv', index_col='Date', parse_dates=True)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Separation
feature_cols = [c for c in df.columns if c != 'Target']
X = df[feature_cols].values
y = df['Target'].values

print(f"Data shape for selection: {X.shape}")

# Scale features (Boruta doesn't strictly need scaling for RF, but good practice)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Boruta Importance
print("Initializing Random Forest for Boruta...")
# Boruta requires a random forest classifier
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=7)

print("Running BorutaPy algorithm (this may take a minute)...")
# Initialize Boruta
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, max_iter=100)

# Fit Boruta
feat_selector.fit(X_scaled, y)

# 3. Extract Selected Features
confirmed_features = np.array(feature_cols)[feat_selector.support_]
tentative_features = np.array(feature_cols)[feat_selector.support_weak_]

print(f"\n✅ Boruta finished! Confirmed features: {len(confirmed_features)}")
print(f"   Tentative features: {len(tentative_features)}")

# Combine confirmed and tentative if confirmed is too low, otherwise just take confirmed
if len(confirmed_features) < 15:
    selected_features = list(confirmed_features) + list(tentative_features)
else:
    selected_features = list(confirmed_features)

# If it's still massive, we trim to top 30 based on pure RF importance to prevent overfitting
if len(selected_features) > 30:
    print(f"Boruta selected {len(selected_features)} features. Trimming to Top 30 via RF importance.")
    rf.fit(X_scaled, y)
    importances = rf.feature_importances_
    # Create dict
    imp_dict = {f: imp for f, imp in zip(feature_cols, importances)}
    selected_features = sorted(selected_features, key=lambda x: imp_dict[x], reverse=True)[:30]

print("\n🏆 FINAL BORUTA SELECTED FEATURES:")
for f in selected_features:
    print(f"   - {f}")

# 5. Split chosen features for the Dual-Head Transformer
sentiment_keywords = ['fear', 'greed', 'fg_']
head_2_cols = [f for f in selected_features if any(k in f.lower() for k in sentiment_keywords)]
head_1_cols = [f for f in selected_features if f not in head_2_cols]

print(f"\n🧠 Head 1 (Market/On-Chain): {len(head_1_cols)} features")
print(f"🧠 Head 2 (Sentiment): {len(head_2_cols)} features")

# If sentiment wasn't selected by Boruta, force the top one in so architecture doesn't break
if len(head_2_cols) == 0:
    print("⚠️ WARNING: No sentiment features confirmed by Boruta. Forcing 'fg_change7' into Head 2.")
    head_2_cols.append('fg_change7')
    if 'fg_change7' in head_1_cols:
         head_1_cols.remove('fg_change7')

# Save configs
with open(DATA_DIR / 'feature_cols.txt', 'w') as f:
    f.write('\n'.join(head_1_cols + head_2_cols))
    
with open(DATA_DIR / 'head_1_cols.txt', 'w') as f:
    f.write('\n'.join(head_1_cols))
    
with open(DATA_DIR / 'head_2_cols.txt', 'w') as f:
    f.write('\n'.join(head_2_cols))

# 6. Save the final curated dataset splits
df_selected = df[head_1_cols + head_2_cols + ['Target']]
df_selected = df_selected.rename(columns={'Target': 'target'})
df_selected.index.name = 'date'
df_selected.to_csv(DATA_DIR / 'btc_features.csv')

train = df_selected[df_selected.index < '2024-07-01']
val = df_selected[(df_selected.index >= '2024-07-01') & (df_selected.index < '2025-01-01')]
test = df_selected[df_selected.index >= '2025-01-01']

train.to_csv(DATA_DIR / 'btc_train.csv')
val.to_csv(DATA_DIR / 'btc_val.csv')
test.to_csv(DATA_DIR / 'btc_test.csv')

print("\n✅ Curated Boruta splits saved to data/ (btc_features.csv, btc_train.csv, etc.)")
