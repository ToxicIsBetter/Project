import pandas as pd
df = pd.read_csv('data/clean_ohlcv.csv')
n = len(df)
n_train = int(n * 0.70)
n_val   = int(n * 0.15)
test_start = df.iloc[n_train + n_val]['Date']
print("Test Start Date:", test_start)
