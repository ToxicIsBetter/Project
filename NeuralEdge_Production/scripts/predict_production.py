import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import os
from datetime import datetime

# ── ARCHITECTURE DEFINITION ───────────────────────────────────────────
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
    def __init__(self, input_dim_h1, input_dim_h2, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.head1_proj = nn.Linear(input_dim_h1, d_model)
        self.head2_proj = nn.Linear(input_dim_h2, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head1 = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.head2 = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        x1 = self.head1_proj(x1)
        x2 = self.head2_proj(x2)
        x = (x1 + x2) / 2
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        h1 = self.dropout(self.head1(x))
        h2 = self.dropout(self.head2(x))
        return h1 + h2

# ── PREDICTOR CLASS ───────────────────────────────────────────────────
class ProductionPredictor:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.model_dir = os.path.join(root_dir, 'models')
        self.data_src = os.path.join(os.path.dirname(root_dir), 'NeuralEdge', 'data')
        self.seq_len = 7
        self.model = None
        self.scalers = {}
        self.metadata = None

    def load(self):
        # Load Metadata
        with open(os.path.join(self.model_dir, 'production_metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
        # Load Model
        self.model = DualHeadTransformer(13, 12, d_model=32, nhead=4, num_layers=2, dropout=0.1)
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, 'production_model.pt'), weights_only=True))
        self.model.eval()

        # Load Scalers
        self.scalers['h1'] = joblib.load(os.path.join(self.model_dir, 'scaler_head1.pkl'))
        self.scalers['h2_mm'] = joblib.load(os.path.join(self.model_dir, 'scaler_head2_minmax.pkl'))
        self.scalers['h2_std'] = joblib.load(os.path.join(self.model_dir, 'scaler_head2_std.pkl'))
        
        print(f"✅ Production Model Loaded ([100% Data Mode])")
        print(f"   Threshold: {self.metadata['threshold']}")

    def get_latest_data(self):
        # Merge exactly like training
        ohlcv = pd.read_csv(os.path.join(self.data_src, 'clean_ohlcv.csv'), parse_dates=['Date'])
        onchain = pd.read_csv(os.path.join(self.data_src, 'clean_onchain.csv'), parse_dates=['Date'])
        sentiment = pd.read_csv(os.path.join(self.data_src, 'clean_sentiment.csv'), parse_dates=['Date'])
        google = pd.read_csv(os.path.join(self.data_src, 'clean_google.csv'), parse_dates=['Date'])

        for frame in [ohlcv, onchain, sentiment, google]:
            frame['Date'] = frame['Date'].dt.normalize()

        df = ohlcv.merge(onchain, on='Date', how='left').merge(sentiment, on='Date', how='left').merge(google, on='Date', how='left')
        df = df.sort_values('Date').ffill().bfill()
        
        # Add momentum
        df['momentum_7d'] = df['Close'].pct_change(7)
        
        # Preprocessing
        log_pos = ['AdrActCnt', 'AdrBalCnt', 'TxCnt', 'HashRate', 'BlkCnt', 'SplyExNtv', 'FlowInExNtv', 'CapMVRVCur']
        signed_log = ['AdrActCnt_growth7d', 'AdrActCnt_growth30d', 'CapMVRVCur_growth7d', 'CapMVRVCur_growth30d', 'gt_change7', 'gt_momentum']
        
        for col in log_pos:
            if col in df.columns: df[col] = np.log1p(df[col].clip(lower=0))
        for col in signed_log:
            if col in df.columns: df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
            
        return df.tail(self.seq_len)

    def predict(self):
        recent = self.get_latest_data()
        date_used = recent['Date'].iloc[-1].strftime('%Y-%m-%d')
        
        h1_features = [
            'AdrActCnt', 'AdrBalCnt', 'TxCnt', 'HashRate', 'BlkCnt',
            'SplyExNtv', 'FlowInExNtv', 'CapMVRVCur',
            'AdrActCnt_growth7d', 'AdrActCnt_growth30d',
            'CapMVRVCur_growth7d', 'CapMVRVCur_growth30d',
            'momentum_7d'
        ]
        h2_bounded = ['fear_greed', 'fg_ma7', 'fg_ma14', 'google_trends', 'gt_ma7', 'gt_ma30']
        h2_cont = ['gt_change7', 'gt_momentum', 'fg_change', 'fg_change7']
        h2_flags = ['fg_extreme_fear', 'fg_extreme_greed']

        # Scale
        x1_scaled = self.scalers['h1'].transform(recent[h1_features].values)
        x2_b = self.scalers['h2_mm'].transform(recent[h2_bounded].values)
        x2_c = self.scalers['h2_std'].transform(recent[h2_cont].values)
        x2_f = recent[h2_flags].values
        x2_scaled = np.hstack([x2_b, x2_c, x2_f])

        # Inference
        X1 = torch.FloatTensor(x1_scaled).unsqueeze(0)
        X2 = torch.FloatTensor(x2_scaled).unsqueeze(0)
        
        with torch.no_grad():
            prob = torch.sigmoid(self.model(X1, X2)).item()
            pred = 1 if prob > self.metadata['threshold'] else 0

        return {
            "prediction": "📈 ACCUMULATE" if pred == 1 else "📉 DISTRIBUTE",
            "probability": prob,
            "threshold": self.metadata['threshold'],
            "target_date": "Next 24 Hours",
            "as_of_date": date_used
        }

if __name__ == "__main__":
    import sys
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    predictor = ProductionPredictor(ROOT)
    
    try:
        predictor.load()
        res = predictor.predict()
        
        print("\n" + "═"*40)
        print(" 🔮 NEURALEDGE PRODUCTION PREDICTION")
        print("═"*40)
        print(f" Data As Of : {res['as_of_date']}")
        print(f" Projection : {res['target_date']}")
        print(f" Direction   : {res['prediction']}")
        print(f" Confidence  : {res['probability']:.2%}")
        print("═"*40)
        
    except FileNotFoundError:
        print("❌ Error: Production models not found. Please run training first.")
