import pandas as pd

# Load and fix onchain data
print("Fixing onchain_and_technicals_2010_to_now.csv...")
df = pd.read_csv('onchain_and_technicals_2010_to_now.csv')
df.columns = ['Date'] + list(df.columns[1:])
df.to_csv('onchain_and_technicals_2010_to_now.csv', index=False)
print(f"✅ Fixed - first column is now 'Date'")
print(f"   Columns: {df.columns[:5].tolist()}")

# Load and fix sentiment data  
print("\nFixing sentiment_2010_to_now.csv...")
df = pd.read_csv('sentiment_2010_to_now.csv')
if df.columns[0] == 'Unnamed: 0' or df.columns[0] == 'index':
    df.columns = ['Date'] + list(df.columns[1:])
    df.to_csv('sentiment_2010_to_now.csv', index=False)
    print(f"✅ Fixed - first column is now 'Date'")
else:
    print(f"ℹ️ Already has correct column names")

# Load and fix ohlcv data
print("\nFixing ohlcv_2010_to_now.csv...")
df = pd.read_csv('ohlcv_2010_to_now.csv')
if df.columns[0] == 'Unnamed: 0' or df.columns[0] == 'index':
    df.columns = ['Date'] + list(df.columns[1:])
    df.to_csv('ohlcv_2010_to_now.csv', index=False)
    print(f"✅ Fixed - first column is now 'Date'")
else:
    print(f"ℹ️ Already has correct column names")

print("\n✅ All data files fixed!")
