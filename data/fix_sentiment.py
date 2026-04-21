import pandas as pd

# Load original sentiment file
df = pd.read_csv('sentiment_2010_to_now.csv')
print("Original columns:", df.columns.tolist()[:10])

# The file has: index, Date, fear_greed, fg_ma7, fg_ma14, fg_change, fg_change7, fg_extreme_fear, fg_extreme_greed
# First column is unnamed index, second is Date
if df.columns[0] == 'Date' and df.columns[1] == 'Date':
    # Two Date columns - drop first
    df = df.drop('Date', axis=1)
    df.columns = ['Date'] + list(df.columns[1:])
elif df.columns[0] == 'Date':
    # Only one Date column - good
    pass
else:
    # First column is index
    df = df.drop(df.columns[0], axis=1)
    df.columns = ['Date'] + list(df.columns[1:])

df['Date'] = pd.to_datetime(df['Date'])
df = df[['Date', 'fear_greed', 'fg_ma7', 'fg_ma14', 'fg_change', 'fg_change7', 'fg_extreme_fear', 'fg_extreme_greed']]
df.to_csv('train_sentiment_fixed.csv', index=False)
print(f"✅ Fixed sentiment: {len(df)} rows")
print(f"Columns: {list(df.columns)}")
