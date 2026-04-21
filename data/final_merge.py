"""Final clean merge with all data through 2026-04-18"""
import pandas as pd

print("Creating final clean merged dataset...\n")

# Load clean files
ohlcv = pd.read_csv('train_ohlcv.csv', parse_dates=['Date'])
onchain = pd.read_csv('train_onchain.csv', parse_dates=['Date'])
sentiment = pd.read_csv('train_sentiment_fixed.csv', parse_dates=['Date'])
google = pd.read_csv('train_google.csv', parse_dates=['Date'])

print(f"OHLCV: {len(ohlcv)} rows")
print(f"On-chain: {len(onchain)} rows")
print(f"Sentiment: {len(sentiment)} rows")
print(f"Google: {len(google)} rows")

# Merge
merged = ohlcv.merge(onchain, on='Date', how='left')
print(f"\nAfter OHLCV+Onchain: {len(merged)} rows")

merged2 = merged.merge(sentiment, on='Date', how='left')
print(f"After +Sentiment: {len(merged2)} rows")

merged3 = merged2.merge(google, on='Date', how='left')
print(f"After +Google: {len(merged3)} rows")

# Check for NaN in critical columns
print(f"\nNaN check:")
print(f"  Close: {merged3['Close'].isna().sum()}")
print(f"  fear_greed: {merged3['fear_greed'].isna().sum()}")
print(f"  fg_ma7: {merged3['fg_ma7'].isna().sum()}")

# Save final merged file
merged3.to_csv('train_final.csv', index=False)
print(f"\n✅ Saved train_final.csv ({len(merged3)} rows)")

# Verify
df_check = pd.read_csv('train_final.csv', parse_dates=['Date'])
print(f"\nVerification:")
print(f"  Shape: {df_check.shape}")
print(f"  Date range: {df_check['Date'].min()} to {df_check['Date'].max()}")
print(f"  Total NaN: {df_check.isna().sum().sum()}")
