import re

# Read the file
with open('model3_pipeline.py', 'r') as f:
    content = f.read()

# Replace the three read_csv calls
old_lines = [
    "ohlcv = pd.read_csv('../../data/ohlcv_2010_to_now.csv', parse_dates=['Date'])",
    "onchain = pd.read_csv('../../data/onchain_and_technicals_2010_to_now.csv', parse_dates=['Date'])",
    "sentiment = pd.read_csv('../../data/sentiment_2010_to_now.csv', parse_dates=['Date'])"
]

new_lines = [
    "ohlcv = pd.read_csv('../../data/ohlcv_2010_to_now.csv', parse_dates=['Date'], date_format='%Y-%m-%d')",
    "onchain = pd.read_csv('../../data/onchain_and_technicals_2010_to_now.csv', parse_dates=['Date'], date_format='%Y-%m-%d')",
    "sentiment = pd.read_csv('../../data/sentiment_2010_to_now.csv', parse_dates=['Date'], date_format='%Y-%m-%d')"
]

for old, new in zip(old_lines, new_lines):
    content = content.replace(old, new)

# Write back
with open('model3_pipeline.py', 'w') as f:
    f.write(content)

print("✅ Fixed date parsing in model3_pipeline.py")
