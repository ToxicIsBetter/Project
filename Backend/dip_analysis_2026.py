import numpy as np
import pandas as pd
import os

# Set paths
M1_DIR = '/home/shyam/UbuntuCode/CN 6000 Mental Wealth Professional Life 3 (Project)/Project/Mango/GoogleTrends/processed_model1'
M2_DIR = '/home/shyam/UbuntuCode/CN 6000 Mental Wealth Professional Life 3 (Project)/Project/Mango/CleanSentiment/processed'

# Load predictions
m1_preds = np.load(f'{M1_DIR}/model1_preds_test.npy')
m2_preds = np.load(f'{M2_DIR}/model2_preds_test.npy')
y_true = np.load(f'{M1_DIR}/model1_ytrue_test.npy')

# Load actual price data
ohlcv = pd.read_csv('/home/shyam/UbuntuCode/CN 6000 Mental Wealth Professional Life 3 (Project)/Project/BTC_Price_in_2026/ohlcv_2010_to_now.csv', parse_dates=['Date'])

# Align to test set (Jan 2024 to Mar 2026)
test_df = ohlcv[ohlcv['Date'] > '2023-12-31'].copy().reset_index(drop=True)
test_df = test_df.iloc[5:5+len(y_true)].reset_index(drop=True)

test_df['daily_return'] = test_df['Close'].shift(-1) / test_df['Close'] - 1
test_df.loc[test_df.index[-1], 'daily_return'] = 0

test_df['M1_Pred'] = m1_preds
test_df['M2_Pred'] = m2_preds
test_df['Actual'] = y_true

# Isolate 2026 Dip
dip_2026 = test_df[test_df['Date'] >= '2026-01-01'].copy().reset_index(drop=True)

# Calculate cumulative returns from start of 2026
dip_2026['Buy_Hold_Return'] = (1 + dip_2026['daily_return']).cumprod()
dip_2026['M1_Return'] = (1 + dip_2026['daily_return'] * dip_2026['M1_Pred']).cumprod()
dip_2026['M2_Return'] = (1 + dip_2026['daily_return'] * dip_2026['M2_Pred']).cumprod()

print("="*60)
print(f"2026 DIP ANALYSIS ({dip_2026['Date'].min().date()} to {dip_2026['Date'].max().date()})")
print("="*60)
print(f"Total Trading Days in 2026: {len(dip_2026)}")
print(f"Starting BTC Price (Jan 1): ${dip_2026['Close'].iloc[0]:,.2f}")
print(f"Ending BTC Price (Mar 20):  ${dip_2026['Close'].iloc[-1]:,.2f}\n")

print("--- Cumulative Return Multiplier (Jan 1 = 1.0) ---")
print(f"Buy & Hold:         {dip_2026['Buy_Hold_Return'].iloc[-1]:.4f} ({(dip_2026['Buy_Hold_Return'].iloc[-1] - 1)*100:.2f}%)")
print(f"Model 1 (Baseline): {dip_2026['M1_Return'].iloc[-1]:.4f} ({(dip_2026['M1_Return'].iloc[-1] - 1)*100:.2f}%)")
print(f"Model 2 (Enriched): {dip_2026['M2_Return'].iloc[-1]:.4f} ({(dip_2026['M2_Return'].iloc[-1] - 1)*100:.2f}%)\n")

print("--- Accuracy During 2026 Crash ---")
print(f"Model 1 Accuracy: {(dip_2026['M1_Pred'] == dip_2026['Actual']).mean() * 100:.2f}%")
print(f"Model 2 Accuracy: {(dip_2026['M2_Pred'] == dip_2026['Actual']).mean() * 100:.2f}%\n")

print("--- Behavior (Days Invested) ---")
print(f"Actual 'Up' Days: {dip_2026['Actual'].sum()} / {len(dip_2026)}")
print(f"Model 1 Bought:     {dip_2026['M1_Pred'].sum()} / {len(dip_2026)} days")
print(f"Model 2 Bought:     {dip_2026['M2_Pred'].sum()} / {len(dip_2026)} days")
