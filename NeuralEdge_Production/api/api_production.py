from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime
import os
import sys
import numpy as np

# Add the scripts directory to path to import predict_production
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from predict_production import ProductionPredictor

app = FastAPI(
    title="NeuralEdge Production API",
    description="Backend API serving the 100% Data Production Model to the isolated frontend.",
    version="1.0.0"
)

# Enable CORS for isolated frontend fetches
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = None
global_df = None

def load_data():
    """Load latest dataset to serve API metrics"""
    global global_df
    try:
        data_src = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'NeuralEdge', 'data')
        ohlcv = pd.read_csv(os.path.join(data_src, 'clean_ohlcv.csv'), parse_dates=['Date'])
        onchain = pd.read_csv(os.path.join(data_src, 'clean_onchain.csv'), parse_dates=['Date'])
        sentiment = pd.read_csv(os.path.join(data_src, 'clean_sentiment.csv'), parse_dates=['Date'])
        google = pd.read_csv(os.path.join(data_src, 'clean_google.csv'), parse_dates=['Date'])

        for frame in [ohlcv, onchain, sentiment, google]:
            frame['Date'] = frame['Date'].dt.normalize()

        df = ohlcv.merge(onchain, on='Date', how='left').merge(sentiment, on='Date', how='left').merge(google, on='Date', how='left')
        df = df.sort_values('Date').ffill().bfill()
        global_df = df
        print("Production Data loaded successfully.")
    except Exception as e:
        print(f"Error loading production data: {e}")


@app.on_event("startup")
async def startup_event():
    global predictor
    load_data()
    try:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        predictor = ProductionPredictor(root_dir)
        predictor.load()
        print("Neural Engine Production Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load production model: {e}")

@app.get("/")
def read_root():
    return {"status": "online", "environment": "PRODUCTION", "model_loaded": predictor is not None}

import requests

@app.get("/api/btc")
def get_btc():
    try:
        # Fetch live real-time price from CoinGecko
        res = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true", timeout=3)
        data = res.json()
        price = float(data['bitcoin']['usd'])
        change = float(data['bitcoin']['usd_24h_change'])
        return {
            "price": price,
            "change_24h": round(change, 2),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Live fetch failed, defaulting to dataset: {e}")
        if global_df is None:
            raise HTTPException(status_code=500, detail="Data not loaded.")
        latest_price = float(global_df['Close'].iloc[-1])
        yesterday_price = float(global_df['Close'].iloc[-2])
        change = ((latest_price - yesterday_price) / yesterday_price) * 100
        return {
            "price": latest_price,
            "change_24h": round(change, 2),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/signal")
def get_signal():
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    try:
        # predict() returns {"prediction": "📈 ACCUMULATE", "probability": 0.98, ...}
        res = predictor.predict()
        is_accumulate = "ACCUMULATE" in res["prediction"]
        signal_text = "ACCUMULATE" if is_accumulate else "DISTRIBUTE"
        prob = res["probability"]
        
        confidence = round(prob * 100, 1) if is_accumulate else round((1 - prob) * 100, 1)
        if confidence > 70:
            risk_level = "LOW"
        elif confidence > 55:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
            signal_text = "HOLD"

        return {
            "signal": signal_text,
            "risk_level": risk_level,
            "confidence": confidence,
            "probability": prob,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio")
def get_portfolio():
    return {"portfolio_value": 125000.00, "return_30d": 4.2}

@app.get("/api/fear-greed")
def get_fear_greed():
    if global_df is None: raise HTTPException(status_code=500, detail="Data not loaded.")
    val = int(global_df['fear_greed'].iloc[-1])
    if val > 75: cls = "Extreme Greed"
    elif val > 55: cls = "Greed"
    elif val > 45: cls = "Neutral"
    elif val > 25: cls = "Fear"
    else: cls = "Extreme Fear"
    return {"value": val, "classification": cls}

@app.get("/api/btc/history/full")
def get_btc_history():
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=365&interval=daily"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        prices = []
        for item in data['prices']:
            ts = item[0] / 1000.0
            date_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            prices.append({"date": date_str, "price": item[1]})
        return {"prices": prices}
    except Exception as e:
        print(f"Warning: Could not fetch live history. Error: {e}")
        if global_df is None: raise HTTPException(status_code=500, detail="Data not loaded.")
        subset = global_df[['Date', 'Close']].copy()
        subset['Date'] = subset['Date'].dt.strftime('%Y-%m-%d')
        subset = subset.rename(columns={'Date': 'date', 'Close': 'price'})
        return {"prices": subset.to_dict(orient='records')}

if __name__ == "__main__":
    import uvicorn
    # Isolated port for Production API
    print("Starting NeuralEdge PRODUCTION API on http://localhost:8003")
    uvicorn.run(app, host="0.0.0.0", port=8003)
