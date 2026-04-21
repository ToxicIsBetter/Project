"""
Sentiment Data Collection Pipeline
Collects free sentiment data from:
1. Alternative.me → Fear & Greed Index (daily → 8H)
2. CFGI.io → Crypto Fear & Greed (15-min → 8H)
3. Google Trends → Search interest (daily → 8H)
4. Reddit (PRAW) → Post sentiment (optional)

Output: btc_8h_sentiment.csv
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
import time


# ─────────────────────────────────────────────────────────────────
# 1. ALTERNATIVE.ME — Fear & Greed Index
# ─────────────────────────────────────────────────────────────────
def fetch_alternative_me_fear_greed(days=365):
    """Fetch Fear & Greed Index from Alternative.me"""
    url = f"https://api.alternative.me/fng/?limit={days}"

    try:
        resp = requests.get(url, timeout=15).json()

        # API returns data directly, not wrapped in status
        if "data" not in resp:
            raise RuntimeError(f"Alternative.me API error: {resp}")

        data = resp["data"]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.rename(
            columns={
                "value": "fg_alternative",
                "value_classification": "fg_classification",
            }
        )
        df["fg_score"] = df["fg_alternative"].astype(int)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        print(f"✅ Alternative.me Fear & Greed: {len(df)} rows")
        return df[["fg_score", "fg_alternative"]]

    except Exception as e:
        print(f"⚠️ Alternative.me failed: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────
# 2. CFGI.IO — Crypto Fear & Greed Index
# ─────────────────────────────────────────────────────────────────
def fetch_cfgi_sentiment():
    """
    Fetch CFGI.io sentiment score (updates every 15 min)
    More granular than Alternative.me
    """
    url = "https://api.cfgi.io/sentiment"

    try:
        resp = requests.get(url, timeout=15).json()

        # CFGI returns current score + history
        # If no history, create single row
        if "data" in resp:
            data = resp["data"]
        else:
            data = [resp]

        records = []
        for item in data:
            ts = item.get("timestamp") or item.get("time")
            if ts:
                if isinstance(ts, int):
                    ts = datetime.fromtimestamp(ts, tz=timezone.utc)
                else:
                    ts = pd.to_datetime(ts)

                records.append(
                    {
                        "timestamp": ts,
                        "cfgi_score": item.get("score") or item.get("value"),
                        "cfgi_classification": item.get("classification", ""),
                    }
                )

        if not records:
            # Fallback: current only
            records = [
                {
                    "timestamp": datetime.now(timezone.utc),
                    "cfgi_score": resp.get("score", 50),
                    "cfgi_classification": resp.get("classification", "Neutral"),
                }
            ]

        df = pd.DataFrame(records)
        if "timestamp" in df.columns:
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

        print(f"✅ CFGI.io Sentiment: {len(df)} rows")
        return (
            df[["cfgi_score", "cfgi_classification"]]
            if "cfgi_score" in df.columns
            else pd.DataFrame()
        )

    except Exception as e:
        print(f"⚠️ CFGI.io failed: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────
# 3. GOOGLE TRENDS — via pytrends
# ─────────────────────────────────────────────────────────────────
def fetch_google_trends(keywords=["Bitcoin"], days=365):
    """
    Fetch Google Trends data for cryptocurrency keywords.
    Uses pytrends library (free, no API key needed).
    """
    try:
        from pytrends.request import TrendReq
        from pytrends import exceptions

        pytrends = TrendReq(hl="en-US", tz=0)

        # Build timeframe
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        timeframe = f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"

        all_data = []

        for keyword in keywords:
            try:
                pytrends.build_payload([keyword], timeframe=timeframe, geo="")
                data = pytrends.interest_over_time()

                if not data.empty:
                    data = data.reset_index(drop=True)
                    data.rename(
                        columns={keyword: f"gt_{keyword.lower()}"}, inplace=True
                    )
                    if f"gt_{keyword.lower()}" in data.columns:
                        all_data.append(data)

                time.sleep(2)  # Be polite to Google

            except exceptions.ResponseError as e:
                print(f"⚠️ Google Trends rate limited for '{keyword}': {e}")
                time.sleep(5)
            except Exception as e:
                print(f"⚠️ Google Trends error for '{keyword}': {e}")

        if all_data:
            combined = pd.concat(all_data, axis=1)
            combined["date"] = pd.to_datetime(combined.index)
            combined.set_index("date", inplace=True)
            combined = combined.select_dtypes(include=[np.number])

            print(f"✅ Google Trends: {combined.shape}")
            return combined
        else:
            return pd.DataFrame()

    except ImportError:
        print("⚠️ pytrends not installed. Run: uv add pytrends")
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Google Trends failed: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────
# 4. REDDIT SENTIMENT (Optional)
# ─────────────────────────────────────────────────────────────────
def fetch_reddit_sentiment(subreddits=["Bitcoin", "btc", "CryptoCurrency"], limit=100):
    """
    Fetch Reddit posts and calculate sentiment.
    Requires: praw (Python Reddit API Wrapper)

    Setup:
    1. Go to reddit.com/prefs/apps
    2. Create app (Script type)
    3. Get client_id and client_secret
    """
    try:
        import praw
        from textblob import TextBlob

        # Replace with your credentials from reddit.com/prefs/apps
        CLIENT_ID = "YOUR_CLIENT_ID"
        CLIENT_SECRET = "YOUR_CLIENT_SECRET"

        if CLIENT_ID == "YOUR_CLIENT_ID":
            print("⚠️ Reddit credentials not set. Skipping Reddit sentiment.")
            return pd.DataFrame()

        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent="neuraledge_sentiment",
        )

        all_posts = []

        for sub in subreddits:
            try:
                subreddit = reddit.subreddit(sub)
                for post in subreddit.hot(limit=limit):
                    # Calculate sentiment score
                    title_score = TextBlob(post.title).sentiment.polarity
                    selftext_score = (
                        TextBlob(post.selftext).sentiment.polarity
                        if post.selftext
                        else 0
                    )
                    combined_score = (title_score + selftext_score) / 2

                    all_posts.append(
                        {
                            "timestamp": datetime.fromtimestamp(
                                post.created_utc, tz=timezone.utc
                            ),
                            "subreddit": sub,
                            "title": post.title,
                            "sentiment_score": combined_score,
                            "score": post.score,
                            "num_comments": post.num_comments,
                        }
                    )
            except Exception as e:
                print(f"⚠️ Error fetching r/{sub}: {e}")

        if all_posts:
            df = pd.DataFrame(all_posts)
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

            # Aggregate to hourly
            hourly = (
                df.resample("1H")
                .agg({"sentiment_score": "mean", "score": "sum", "num_comments": "sum"})
                .dropna()
            )

            hourly.columns = [
                "reddit_sentiment",
                "reddit_score_sum",
                "reddit_comments_sum",
            ]

            print(f"✅ Reddit Sentiment: {len(hourly)} rows")
            return hourly
        else:
            return pd.DataFrame()

    except ImportError:
        print("⚠️ praw or textblob not installed. Skipping Reddit.")
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Reddit failed: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────
# MERGE ALL SENTIMENT DATA
# ─────────────────────────────────────────────────────────────────
def collect_all_sentiment(days=365, reddit_enabled=False):
    """Collect all sentiment sources and merge."""
    print("\n" + "=" * 55)
    print("Collecting Sentiment Data")
    print("=" * 55 + "\n")

    all_frames = []

    # 1. Alternative.me Fear & Greed
    fg_alt = fetch_alternative_me_fear_greed(days=days * 2)  # Get extra history
    if not fg_alt.empty:
        all_frames.append(fg_alt)

    # 2. CFGI.io
    cfgi = fetch_cfgi_sentiment()
    if not cfgi.empty:
        all_frames.append(cfgi)

    # 3. Google Trends
    gt = fetch_google_trends(keywords=["Bitcoin"], days=days)
    if not gt.empty:
        all_frames.append(gt)

    # 4. Reddit (optional, slow)
    if reddit_enabled:
        reddit = fetch_reddit_sentiment()
        if not reddit.empty:
            all_frames.append(reddit)

    if not all_frames:
        raise RuntimeError("No sentiment data collected!")

    # Merge all sources
    sentiment_df = all_frames[0]
    for frame in all_frames[1:]:
        sentiment_df = sentiment_df.join(frame, how="outer")

    sentiment_df.sort_index(inplace=True)

    # Resample to 8H (forward-fill for slow metrics like Fear & Greed)
    sentiment_8h = sentiment_df.resample("8H").first()
    sentiment_8h.ffill(inplace=True)

    # Drop columns that are still all NaN
    sentiment_8h = sentiment_8h.dropna(how="all", axis=1)

    print(f"\n{'=' * 55}")
    print(f"✅ Final Sentiment (8H): {sentiment_8h.shape}")
    print(f"   Columns: {list(sentiment_8h.columns)}")
    print(f"   Date range: {sentiment_8h.index[0]} → {sentiment_8h.index[-1]}")
    print(f"{'=' * 55}\n")

    return sentiment_8h


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    days = 365
    if len(sys.argv) > 1:
        days = int(sys.argv[1])

    # Collect sentiment
    sentiment_8h = collect_all_sentiment(days=days, reddit_enabled=False)

    # Save
    output_file = "btc_8h_sentiment.csv"
    sentiment_8h.to_csv(output_file)
    print(f"\n✅ Saved: {output_file}")

    # Merge with price data if exists
    price_file = "btc_8h_full.csv"
    if Path(price_file).exists():
        print(f"\n📊 Merging with price data...")
        price = pd.read_csv(price_file, index_col=0, parse_dates=True)

        # Join
        final = price.join(sentiment_8h, how="left")
        final.ffill(inplace=True)
        final.bfill(inplace=True)
        final.dropna(inplace=True)

        final_output = "btc_8h_FINAL.csv"
        final.to_csv(final_output)
        print(f"✅ Final dataset: {final_output}")
        print(f"   Shape: {final.shape}")
        print(f"   Features: {list(final.columns)}")
    else:
        print(f"\n⚠️ Price data not found. Run collect_8h_data.py first.")

    print("\n🎉 DONE")
