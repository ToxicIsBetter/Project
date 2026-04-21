"""
NeuralEdge - Bitcoin Price Prediction Script
Uses the fine-tuned Dual-Head Transformer model (Challenger Run 82)
Architecture: d=32, nhead=4, num_layers=2, dropout=0.1, seq_len=7
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
#  ARCHITECTURE
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


class BitcoinPredictor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.scaler_h1 = None
        self.scaler_h2_mm = None
        self.scaler_h2_std = None
        self.feature_config = None
        self.metrics = None
        self.SEQ_LEN = 7
        self.THRESHOLD = 0.65

    def load_model(self):
        """Load the trained model, scalers, and config."""
        metrics_path = os.path.join(self.model_dir, 'metrics_finetuned.json')
        
        with open(metrics_path, 'r') as f:
            self.metrics = json.load(f)
            self.THRESHOLD = self.metrics.get('threshold', 0.65)
        
        with open(os.path.join(self.model_dir, 'feature_sets.json'), 'r') as f:
            self.feature_config = json.load(f)

        n_h1 = len(self.feature_config['final_h1'])
        n_h2 = len(self.feature_config['sentiment'])

        # NeuralEdge architecture params
        self.model = DualHeadTransformer(
            input_dim_h1=n_h1, input_dim_h2=n_h2,
            d_model=32, nhead=4, num_layers=2, dropout=0.1
        )
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, 'best_model.pt'), map_location='cpu'))
        self.model.eval()

        self.scaler_h1    = joblib.load(os.path.join(self.model_dir, 'scaler_head1.pkl'))
        self.scaler_h2_mm = joblib.load(os.path.join(self.model_dir, 'scaler_head2_minmax.pkl'))
        self.scaler_h2_std = joblib.load(os.path.join(self.model_dir, 'scaler_head2_std.pkl'))

        print("✅ Production Model loaded successfully (NeuralEdge Challenger)")
        print(f"   Accuracy: {self.metrics.get('accuracy', 0):.2%}, Threshold: {self.THRESHOLD}")

    def prepare_features(self, df):
        """Add momentum and volatility features with NeuralEdge logic"""
        df = df.copy()
        df['momentum_7d']   = df['Close'].pct_change(7)
        df['momentum_14d']  = df['Close'].pct_change(14)
        df['momentum_30d']  = df['Close'].pct_change(30)
        df['volatility_14d'] = (
            df['Close'].pct_change().rolling(14).std() /
            (df['Close'].pct_change().rolling(14).mean() + 1e-6)
        )
        df['ma_ratio'] = df['Close'] / (df['Close'].rolling(50).mean() + 1e-6)
        
        # Log transforms matching NeuralEdge training
        log_pos = ['AdrActCnt', 'AdrBalCnt', 'TxCnt', 'HashRate', 'BlkCnt', 'SplyExNtv', 'FlowInExNtv', 'CapMVRVCur']
        for col in [c for c in log_pos if c in df.columns]:
            df[col] = np.log1p(df[col].clip(lower=0))
        
        signed_log = ['AdrActCnt_growth7d', 'AdrActCnt_growth30d', 'CapMVRVCur_growth7d', 'CapMVRVCur_growth30d', 'gt_change7', 'gt_momentum']
        for col in [c for c in signed_log if c in df.columns]:
            df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
            
        return df.ffill().bfill()

    def predict(self, df, return_probs=False):
        """Standard point inference"""
        df = self.prepare_features(df)
        recent = df.tail(self.SEQ_LEN).copy()
        
        h1_features = self.feature_config['final_h1']
        h2_features = self.feature_config['sentiment']

        bounded    = [c for c in ['fear_greed', 'fg_ma7', 'fg_ma14', 'google_trends', 'gt_ma7', 'gt_ma30'] if c in h2_features]
        flag_cols  = [c for c in ['fg_extreme_fear', 'fg_extreme_greed'] if c in h2_features]
        continuous = [c for c in h2_features if c not in bounded + flag_cols]

        h1_scaled  = self.scaler_h1.transform(recent[h1_features].values)
        h2_bounded = self.scaler_h2_mm.transform(recent[bounded].values) if bounded else np.empty((len(recent), 0))
        h2_cont    = self.scaler_h2_std.transform(recent[continuous].values) if continuous else np.empty((len(recent), 0))
        h2_flag    = recent[flag_cols].values if flag_cols else np.empty((len(recent), 0))
        h2_scaled  = np.hstack([h2_bounded, h2_cont, h2_flag])

        X1 = torch.FloatTensor(h1_scaled).unsqueeze(0)
        X2 = torch.FloatTensor(h2_scaled).unsqueeze(0)

        with torch.no_grad():
            prob = torch.sigmoid(self.model(X1, X2)).item()
            pred = 1 if prob > self.THRESHOLD else 0

        return (pred, prob) if return_probs else pred

    def predict_historical(self, df):
        """Batched inference for Simulator UI"""
        df = self.prepare_features(df)
        h1_features = self.feature_config['final_h1']
        h2_features = self.feature_config['sentiment']

        h1_scaled = self.scaler_h1.transform(df[h1_features].values)
        
        bounded    = [c for c in ['fear_greed', 'fg_ma7', 'fg_ma14', 'google_trends', 'gt_ma7', 'gt_ma30'] if c in h2_features]
        flag_cols  = [c for c in ['fg_extreme_fear', 'fg_extreme_greed'] if c in h2_features]
        continuous = [c for c in h2_features if c not in bounded + flag_cols]

        h2_b = self.scaler_h2_mm.transform(df[bounded].values) if bounded else np.empty((len(df), 0))
        h2_c = self.scaler_h2_std.transform(df[continuous].values) if continuous else np.empty((len(df), 0))
        h2_f = df[flag_cols].values if flag_cols else np.empty((len(df), 0))
        h2_scaled = np.hstack([h2_b, h2_c, h2_f])

        X1, X2, valid_dates, valid_prices = [], [], [], []
        dates = df['Date'].values
        prices = df['Close'].values
        
        for i in range(self.SEQ_LEN, len(h1_scaled)):
            X1.append(h1_scaled[i - self.SEQ_LEN:i])
            X2.append(h2_scaled[i - self.SEQ_LEN:i])
            valid_dates.append(dates[i])
            valid_prices.append(prices[i])

        if not X1: return []

        X1_t = torch.FloatTensor(np.array(X1))
        X2_t = torch.FloatTensor(np.array(X2))
        
        results = []
        bs = 128
        with torch.no_grad():
            for i in range(0, len(X1_t), bs):
                probs = torch.sigmoid(self.model(X1_t[i:i+bs], X2_t[i:i+bs])).cpu().numpy().flatten()
                for j, prob in enumerate(probs):
                    idx = i + j
                    p_val = 1 if prob > self.THRESHOLD else 0
                    results.append({
                        "date": str(valid_dates[idx]).split('T')[0],
                        "price": float(valid_prices[idx]),
                        "signal": "ACCUMULATE" if p_val == 1 else "DISTRIBUTE",
                        "probability": float(prob)
                    })
        return results

def main():
    print("🚀 NeuralEdge Production Predictor Test")
    predictor = BitcoinPredictor()
    predictor.load_model()
    # Mock DF for structural test only
    return True

if __name__ == '__main__':
    main()
