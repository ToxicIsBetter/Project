"""
NeuralEdge - Bitcoin Price Prediction Script
Uses the Rank 3 Grid-Search-Optimised Dual-Head Transformer
Architecture: d=32, nhead=8, num_layers=1, dropout=0.1, seq_len=7
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import os
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE (must match retrain_rank3.py exactly)
# ══════════════════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DualHeadTransformer(nn.Module):
    def __init__(self, input_dim_h1, input_dim_h2,
                 d_model=32, nhead=8, num_layers=1, dropout=0.1):
        super().__init__()
        self.head1_proj = nn.Linear(input_dim_h1, d_model)
        self.head2_proj = nn.Linear(input_dim_h2, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head1 = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.head2 = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        x1 = self.head1_proj(x1)
        x2 = self.head2_proj(x2)
        x  = (x1 + x2) / 2
        x  = self.pos_encoder(x)
        x  = self.transformer(x)
        x  = x[:, -1, :]
        h1 = self.dropout(self.head1(x))
        h2 = self.dropout(self.head2(x))
        return h1 + h2


# ══════════════════════════════════════════════════════════════════════
#  PREDICTOR CLASS
# ══════════════════════════════════════════════════════════════════════
class BitcoinPredictor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.scaler_h1 = None
        self.scaler_h2_mm = None
        self.scaler_h2_std = None
        self.feature_config = None
        self.metrics = None
        self.SEQ_LEN = 7       # Rank 3 optimal
        self.THRESHOLD = 0.44  # Rank 3 calibrated

    def load_model(self):
        """Load the trained model, scalers, and config."""
        # Load metrics (contains hyperparameters)
        with open(f'{self.model_dir}/metrics_finetuned.json', 'r') as f:
            self.metrics = json.load(f)

        # Read hyperparams from saved metrics
        hp = self.metrics.get('hyperparameters', {})
        d_model    = hp.get('d_model', 32)
        nhead      = hp.get('nhead', 8)
        num_layers = hp.get('num_layers', 1)
        dropout    = hp.get('dropout', 0.1)
        self.SEQ_LEN   = hp.get('seq_len', 7)
        self.THRESHOLD = self.metrics.get('best_threshold', 0.44)

        # Load feature config
        with open(f'{self.model_dir}/feature_sets.json', 'r') as f:
            self.feature_config = json.load(f)

        n_h1 = len(self.feature_config['final_h1'])
        n_h2 = len(self.feature_config['sentiment'])

        # Build and load model
        self.model = DualHeadTransformer(
            input_dim_h1=n_h1, input_dim_h2=n_h2,
            d_model=d_model, nhead=nhead,
            num_layers=num_layers, dropout=dropout)
        self.model.load_state_dict(
            torch.load(f'{self.model_dir}/model3_best.pt', map_location='cpu'))
        self.model.eval()

        # Load scalers
        self.scaler_h1     = joblib.load(f'{self.model_dir}/scaler_head1.pkl')
        self.scaler_h2_mm  = joblib.load(f'{self.model_dir}/scaler_head2_minmax.pkl')
        self.scaler_h2_std = joblib.load(f'{self.model_dir}/scaler_head2_std.pkl')

        print("✅ Model loaded successfully")
        print(f"   Architecture: d={d_model}, nhead={nhead}, layers={num_layers}, dropout={dropout}")
        print(f"   Accuracy: {self.metrics['accuracy']:.2%}")
        print(f"   F1 Score: {self.metrics['f1']:.4f}")
        print(f"   Threshold: {self.THRESHOLD}")
        print(f"   Sequence Length: {self.SEQ_LEN} days")

    def prepare_features(self, df):
        """Add momentum features and apply log transforms to match training."""
        df = df.copy()
        df['momentum_7d']    = df['Close'].pct_change(7)
        df['momentum_14d']   = df['Close'].pct_change(14)
        df['momentum_30d']   = df['Close'].pct_change(30)
        df['volatility_14d'] = df['Close'].pct_change().rolling(14).std() / (df['Close'].pct_change().rolling(14).mean() + 1e-6)
        df['ma_ratio']       = df['Close'] / (df['Close'].rolling(50).mean() + 1e-6)

        # Log transforms (must match training pipeline)
        log_pos = ['AdrActCnt', 'AdrBalCnt', 'TxCnt', 'TxTfrCnt', 'HashRate',
                   'BlkCnt', 'SplyCur', 'SplyExNtv', 'SplyExpFut10yr',
                   'FeeTotNtv', 'FlowInExNtv', 'FlowOutExNtv', 'IssTotNtv', 'CapMVRVCur']
        for col in [c for c in log_pos if c in df.columns]:
            df[col] = np.log1p(df[col].clip(lower=0))

        signed_log = ['TxCnt_growth7d', 'TxCnt_growth30d', 'HashRate_growth7d',
                      'HashRate_growth30d', 'AdrActCnt_growth7d', 'AdrActCnt_growth30d',
                      'CapMVRVCur_growth7d', 'CapMVRVCur_growth30d', 'gt_change7', 'gt_momentum']
        for col in [c for c in signed_log if c in df.columns]:
            df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

        # Fix Fear & Greed neutral values
        fg_cols = ['fear_greed', 'fg_ma7', 'fg_ma14', 'fg_change', 'fg_change7',
                   'fg_extreme_fear', 'fg_extreme_greed']
        for col in [c for c in fg_cols if c in df.columns]:
            df[col] = df[col].replace(50.0, np.nan).ffill(limit=3).bfill(limit=3).fillna(0.0)

        df = df.ffill().bfill()
        return df

    def predict(self, df, return_probs=False):
        """
        Make a prediction using the last SEQ_LEN rows of the DataFrame.

        Args:
            df: DataFrame with OHLCV + on-chain + sentiment + google trends
            return_probs: If True, returns (prediction, probability)

        Returns:
            prediction (int): 1 = UP, 0 = DOWN
            probability (float): only if return_probs=True
        """
        # Apply transforms to FULL dataframe first (needs history for rolling calcs)
        df = self.prepare_features(df)

        # Now take the last SEQ_LEN rows
        recent = df.tail(self.SEQ_LEN).copy()

        h1_features = self.feature_config['final_h1']
        h2_features = self.feature_config['sentiment']

        # Scale Head 1
        h1_scaled = self.scaler_h1.transform(recent[h1_features].values)

        # Scale Head 2
        bounded   = ['fear_greed', 'fg_ma7', 'fg_ma14', 'google_trends', 'gt_ma7', 'gt_ma30']
        bounded   = [c for c in bounded if c in h2_features]
        flag_cols = ['fg_extreme_fear', 'fg_extreme_greed']
        flag_cols = [c for c in flag_cols if c in h2_features]
        continuous = [c for c in h2_features if c not in bounded + flag_cols]

        h2_b = self.scaler_h2_mm.transform(recent[bounded].values) if bounded else np.empty((len(recent), 0))
        h2_c = self.scaler_h2_std.transform(recent[continuous].values) if continuous else np.empty((len(recent), 0))
        h2_f = recent[flag_cols].values if flag_cols else np.empty((len(recent), 0))

        h2_scaled = np.hstack([h2_b, h2_c, h2_f])

        # Create tensors with batch dimension
        X1 = torch.FloatTensor(h1_scaled).unsqueeze(0)
        X2 = torch.FloatTensor(h2_scaled).unsqueeze(0)

        # Predict
        with torch.no_grad():
            logits = self.model(X1, X2)
            prob   = torch.sigmoid(logits).item()
            pred   = 1 if prob > self.THRESHOLD else 0

        if return_probs:
            return pred, prob
        return pred


# ══════════════════════════════════════════════════════════════════════
#  MAIN — Example usage
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  NeuralEdge — Bitcoin Price Direction Prediction")
    print("  Rank 3 Grid-Search Model (d=32, nh=8, nl=1, seq=7)")
    print("=" * 60)

    # Initialize
    predictor = BitcoinPredictor(model_dir='models')
    predictor.load_model()

    # Load data — needs all 4 merged CSVs
    print("\n  Loading latest data...")
    ohlcv     = pd.read_csv('data/clean_ohlcv.csv',     parse_dates=['Date'])
    onchain   = pd.read_csv('data/clean_onchain.csv',   parse_dates=['Date'])
    sentiment = pd.read_csv('data/clean_sentiment.csv', parse_dates=['Date'])
    google    = pd.read_csv('data/clean_google.csv',    parse_dates=['Date'])

    for frame in [ohlcv, onchain, sentiment, google]:
        frame['Date'] = frame['Date'].dt.normalize()

    df = ohlcv.merge(onchain,   on='Date', how='left')
    df = df.merge(sentiment, on='Date', how='left')
    df = df.merge(google,    on='Date', how='left')
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.ffill().bfill()

    latest_date = df['Date'].max().strftime('%Y-%m-%d')
    latest_price = df['Close'].iloc[-1]

    # Make prediction
    print(f"  Latest date:  {latest_date}")
    print(f"  Latest close: ${latest_price:,.2f}")
    print(f"  Using last {predictor.SEQ_LEN} days for prediction...\n")

    pred, prob = predictor.predict(df, return_probs=True)

    direction = "📈 UP" if pred == 1 else "📉 DOWN"
    if prob > 0.7:
        confidence = "🟢 HIGH"
    elif prob > 0.5:
        confidence = "🟡 MEDIUM"
    elif prob > 0.3:
        confidence = "🟠 LOW"
    else:
        confidence = "🔴 VERY LOW"

    print("  ╔══════════════════════════════════════╗")
    print(f"  ║  PREDICTION:  {direction:>20s}   ║")
    print(f"  ║  PROBABILITY: {prob:>20.2%}   ║")
    print(f"  ║  CONFIDENCE:  {confidence:>20s}   ║")
    print(f"  ║  THRESHOLD:   {predictor.THRESHOLD:>20.2f}   ║")
    print("  ╚══════════════════════════════════════╝")

    return pred, prob


if __name__ == '__main__':
    main()
