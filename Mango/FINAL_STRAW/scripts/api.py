from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from datetime import datetime
import os
import sys
import numpy as np

# Add the parent directory to the path so we can import predict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.predict import BitcoinPredictor

app = FastAPI(
    title="NeuralEdge Bitcoin API",
    description="Backend API serving the Dual-Head Transformer model to the frontend UI.",
    version="1.1.0"
)

# Enable CORS for frontend fetches
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in absolute production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Variables to hold Data and Model
predictor = None
global_df = None
historical_predictions = []

def load_data():
    """Helper to merge and load the datasets."""
    global global_df
    try:
        ohlcv     = pd.read_csv('data/clean_ohlcv.csv', parse_dates=['Date'])
        onchain   = pd.read_csv('data/clean_onchain.csv', parse_dates=['Date'])
        sentiment = pd.read_csv('data/clean_sentiment.csv', parse_dates=['Date'])
        google    = pd.read_csv('data/clean_google.csv', parse_dates=['Date'])

        for frame in [ohlcv, onchain, sentiment, google]:
            frame['Date'] = frame['Date'].dt.normalize()

        df = ohlcv.merge(onchain, on='Date', how='left')
        df = df.merge(sentiment, on='Date', how='left')
        df = df.merge(google, on='Date', how='left')
        df = df.sort_values('Date').reset_index(drop=True)
        global_df = df.ffill().bfill()
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")

@app.on_event("startup")
async def startup_event():
    global predictor
    global historical_predictions
    
    # Load data
    load_data()
    
    # Init predictor
    try:
        predictor = BitcoinPredictor(model_dir='models')
        if not os.path.exists('models/best_model.pt'):
             predictor = BitcoinPredictor(model_dir='grid_search_output')
        predictor.load_model()
        print("Neural Engine Model loaded successfully.")
        
        # Precompute historical predictions for simulator
        if global_df is not None:
             print("Precomputing batched historical inference for simulator...")
             historical_predictions = predictor.predict_historical(global_df)
             print(f"Computed {len(historical_predictions)} historical frames.")
    except Exception as e:
        print(f"Failed to load model: {e}")


@app.get("/")
def read_root():
    return {"status": "online", "model_loaded": predictor is not None}


@app.get("/api/btc")
def get_btc():
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
    if predictor is None or global_df is None:
        raise HTTPException(status_code=500, detail="Model/Data not loaded.")

    try:
        pred, prob = predictor.predict(global_df, return_probs=True)
        signal = "ACCUMULATE" if pred == 1 else "DISTRIBUTE"
        
        # Calculate heuristics out of raw probability
        confidence = round(prob * 100, 1) if pred == 1 else round((1 - prob) * 100, 1)
        if confidence > 70:
            risk_level = "LOW"
        elif confidence > 55:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
            signal = "HOLD"

        return {
            "signal": signal,
            "risk_level": risk_level,
            "confidence": confidence,
            "probability": prob,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio")
def get_portfolio():
    # Mock portfolio returns, normally hooked up to actual exchange keys
    return {
        "portfolio_value": 125000.00,
        "return_30d": 4.2
    }


@app.get("/api/fear-greed")
def get_fear_greed():
    if global_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded.")
        
    val = int(global_df['fear_greed'].iloc[-1])
    
    if val > 75: cls = "Extreme Greed"
    elif val > 55: cls = "Greed"
    elif val > 45: cls = "Neutral"
    elif val > 25: cls = "Fear"
    else: cls = "Extreme Fear"
    
    return {
        "value": val,
        "classification": cls
    }


@app.get("/api/btc/history/full")
def get_btc_history():
    if global_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded.")
        
    # Return simple date / price JSON array structure for chart UI
    subset = global_df[['Date', 'Close']].copy()
    subset['Date'] = subset['Date'].dt.strftime('%Y-%m-%d')
    subset = subset.rename(columns={'Date': 'date', 'Close': 'price'})
    return {"prices": subset.to_dict(orient='records')}


@app.get("/api/engine")
def get_engine():
    if global_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded.")
    
    latest = global_df.iloc[-1]
    
    return {
        "fusion_layer": {
            "status": "SYNCHRONIZED",
            "reconciliation": 95,
            "consensus": "STABLE"
        },
        "latency_ms": 14,
        "head1_onchain": {
            "whale_flow": f"{round(latest.get('FlowInExUSD', 0) / 1e9, 2)}B",
            "active_wallets": f"{int(latest.get('AdrActCnt', 0))}",
            "mvrv_ratio": round(latest.get('CapMVRVCur', 0), 2),
            "google_trends": int(latest.get('google_trends', 0))
        },
        "head2_sentiment": {
            "greed_index": int(latest.get('fear_greed', 0)),
            "classification": "Greed" if latest.get('fear_greed', 0) > 55 else "Fear",
            "social_volume": "142K/hr"
        }
    }


@app.get("/api/simulator/history")
def get_simulator_history():
    return historical_predictions


if __name__ == "__main__":
    import uvicorn
    # Important: Run on port 8002 specifically to match the frontend Javascript!
    print("Starting NeuralEdge API on http://localhost:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)
