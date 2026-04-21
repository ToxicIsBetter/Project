from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from datetime import datetime
import os
import sys

# Add the parent directory to the path so we can import predict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.predict import BitcoinPredictor

app = FastAPI(
    title="NeuralEdge Bitcoin Predictor API",
    description="Backend API for the Dual-Head Transformer model.",
    version="1.0.0"
)

# Load the predictor once when the server starts
try:
    predictor = BitcoinPredictor(model_dir='models')
    predictor.load_model()
except Exception as e:
    print(f"Failed to load model: {e}")
    predictor = None


class PredictionResponse(BaseModel):
    date: str
    latest_close: float
    prediction_raw: int
    direction: str
    probability: float
    confidence: str
    threshold: float


@app.get("/")
def read_root():
    return {"status": "online", "model_loaded": predictor is not None}


@app.get("/predict", response_model=PredictionResponse)
def get_prediction():
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        # Load the latest merged data from CSVs
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
        df = df.ffill().bfill()

        latest_date = df['Date'].max().strftime('%Y-%m-%d')
        latest_price = float(df['Close'].iloc[-1])

        # Run inference
        pred, prob = predictor.predict(df, return_probs=True)

        direction = "UP" if pred == 1 else "DOWN"
        if prob > 0.7:
            confidence = "HIGH"
        elif prob > 0.5:
            confidence = "MEDIUM"
        elif prob > 0.3:
            confidence = "LOW"
        else:
            confidence = "VERY LOW"

        return PredictionResponse(
            date=latest_date,
            latest_close=latest_price,
            prediction_raw=pred,
            direction=direction,
            probability=prob,
            confidence=confidence,
            threshold=predictor.THRESHOLD
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Run the API locally on port 8000
    print("Starting NeuralEdge API on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
