import requests
import pandas as pd
import time
from functools import reduce

SANTIMENT_API_KEY = 'buwn6m4bhghssgtb_pvnxgzgr5luh55r4'
GRAPHQL_URL = 'https://api.santiment.net/graphql'
HEADERS = {'Authorization': f'Apikey {SANTIMENT_API_KEY}'}

def fetch_santiment_metric(metric_name, slug, from_date, to_date):
    query = f"""
    {{
      getMetric(metric: "{metric_name}") {{
        timeseriesData(
          slug: "{slug}"
          from: "{from_date}T00:00:00Z"
          to: "{to_date}T00:00:00Z"
          interval: "1d"
        ) {{
          datetime
          value
        }}
      }}
    }}
    """
    try:
        response = requests.post(GRAPHQL_URL, json={'query': query}, headers=HEADERS)
        data = response.json()
        if 'errors' in data:
            print(f"❌ Error on {metric_name}: {data['errors']}")
            return pd.DataFrame()
        rows = data['data']['getMetric']['timeseriesData']
        if not rows:
            print(f"⚠️  No data returned for {metric_name}")
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df['Date'] = pd.to_datetime(df['datetime']).dt.tz_localize(None).dt.normalize()
        df = df.rename(columns={'value': metric_name}).drop(columns='datetime')
        print(f"✅ {metric_name}: {len(df)} rows | {df['Date'].min().date()} to {df['Date'].max().date()}")
        return df
    except Exception as e:
        print(f"❌ Exception on {metric_name}: {e}")
        return pd.DataFrame()

# Step 1 — check exactly what metrics your API key has access to
check_query = """
{
  currentUser {
    apikeysExpiration
    subscriptions {
      plan { name }
    }
  }
}
"""
resp = requests.post(GRAPHQL_URL, json={'query': check_query}, headers=HEADERS)
print("Account info:", resp.json())
time.sleep(1)

# Step 2 — fetch with corrected metric names (use _total suffix variants)
metrics = [
    'social_volume_total',
    'sentiment_balance_total',
    'social_dominance_total',
    'sentiment_volume_consumed_twitter',
    'sentiment_volume_consumed_reddit',
]

FROM_YEAR = 2016
TO_YEAR   = 2026
SLUG      = 'bitcoin'

all_frames = {m: [] for m in metrics}

for year in range(FROM_YEAR, TO_YEAR + 1):
    from_date = f'{year}-01-01'
    to_date   = f'{year}-12-31' if year < TO_YEAR else '2026-03-21'
    print(f"\n--- Fetching year {year} ---")
    for metric in metrics:
        df_chunk = fetch_santiment_metric(metric, SLUG, from_date, to_date)
        if not df_chunk.empty:
            all_frames[metric].append(df_chunk)
        time.sleep(1.5)

# Combine each metric across years
combined = []
for metric, frames in all_frames.items():
    if frames:
        full = pd.concat(frames).drop_duplicates('Date').sort_values('Date')
        combined.append(full)
        print(f"{metric}: {len(full)} total rows")

if combined:
    san_df = reduce(lambda a, b: a.merge(b, on='Date', how='outer'), combined)
    san_df = san_df.sort_values('Date').reset_index(drop=True)

    # Derived features
    if 'social_volume_total' in san_df.columns:
        san_df['social_vol_ma7']     = san_df['social_volume_total'].rolling(7).mean()
        san_df['social_vol_ma30']    = san_df['social_volume_total'].rolling(30).mean()
        san_df['social_vol_change7'] = san_df['social_volume_total'].pct_change(7)
    if 'sentiment_balance_total' in san_df.columns:
        san_df['sentiment_bal_ma7']  = san_df['sentiment_balance_total'].rolling(7).mean()

    san_df.to_csv('santiment_btc.csv', index=False)
    print(f"\n✅ Saved santiment_btc.csv")
    print(f"   {len(san_df)} rows | {san_df['Date'].min().date()} to {san_df['Date'].max().date()}")
else:
    print("❌ No data fetched — check account info printed above for subscription details")
