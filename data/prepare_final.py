"""Final data preparation with proper datetime"""
import pandas as pd

files = {
    'ohlcv_ready.csv': 'ohlcv_2010_to_now.csv',
    'onchain_ready.csv': 'onchain_and_technicals_2010_to_now.csv', 
    'sentiment_ready.csv': 'sentiment_2010_to_now.csv',
    'google_trends_ready.csv': 'google_trends_bitcoin.csv'
}

for out_name, in_name in files.items():
    print(f"Processing {in_name}...")
    df = pd.read_csv(in_name)
    
    # Fix first column name and type
    if df.columns[0] != 'Date':
        df.columns = ['Date'] + list(df.columns[1:])
    
    # Convert to datetime and save
    df['Date'] = pd.to_datetime(df['Date'])
    df.to_csv(out_name, index=False)
    print(f"  ✅ {out_name}: {len(df)} rows, Date type: {df['Date'].dtype}")

print("\n✅ All files prepared with datetime Date column")
