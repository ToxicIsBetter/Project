import pandas as pd
import numpy as np

df = pd.read_csv("btc.csv")

df_price = pd.read_csv('btc_binance_daily.csv', parse_dates=['date'], index_col='date')
print("Price data:", df_price.shape)
print(df_price.tail(2))

cm = pd.read_csv('btc.csv')
print("\nOn-Chain cols(32):", list(cm.columns))
print("\nOn-Chain shape:", cm.shape)
