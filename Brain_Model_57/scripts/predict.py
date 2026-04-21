"""
NeuralEdge - Bitcoin Price Prediction Script
Uses the fine-tuned Dual-Head Transformer model
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import os
from datetime import datetime

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
    def __init__(self, input_dim_h1, input_dim_h2, d_model=64, nhead=8, num_layers=2, dropout=0.3):
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
        self.SEQ_LEN = 5
        self.THRESHOLD = 0.27
        
    def load_model(self):
        """Load the trained model and scalers"""
        # Load model
        self.model = DualHeadTransformer(13, 12)
        self.model.load_state_dict(torch.load(f'{self.model_dir}/model3_best.pt'))
        self.model.eval()
        
        # Load scalers
        self.scaler_h1 = joblib.load(f'{self.model_dir}/scaler_head1.pkl')
        self.scaler_h2_mm = joblib.load(f'{self.model_dir}/scaler_head2_minmax.pkl')
        self.scaler_h2_std = joblib.load(f'{self.model_dir}/scaler_head2_std.pkl')
        
        # Load feature config
        with open(f'{self.model_dir}/feature_sets.json', 'r') as f:
            self.feature_config = json.load(f)
        
        # Load metrics
        with open(f'{self.model_dir}/metrics_finetuned.json', 'r') as f:
            self.metrics = json.load(f)
        
        print("✅ Model loaded successfully")
        print(f"   Accuracy: {self.metrics['accuracy']:.2%}")
        print(f"   F1 Score: {self.metrics['f1']:.4f}")
        print(f"   Threshold: {self.THRESHOLD}")
        
    def prepare_features(self, df):
        """Prepare features from raw data"""
        # Add momentum features
        df['momentum_7d'] = df['Close'].pct_change(7)
        df['momentum_14d'] = df['Close'].pct_change(14)
        df['momentum_30d'] = df['Close'].pct_change(30)
        df['volatility_14d'] = df['Close'].pct_change().rolling(14).std() / (df['Close'].pct_change().rolling(14).mean() + 1e-6)
        df['ma_ratio'] = df['Close'] / (df['Close'].rolling(50).mean() + 1e-6)
        
        return df
    
    def predict(self, df, return_probs=False):
        """
        Make predictions on the last SEQ_LEN rows
        
        Args:
            df: DataFrame with required columns
            return_probs: If True, return probabilities instead of binary predictions
            
        Returns:
            Prediction (0 or 1) and probability
        """
        # Get last SEQ_LEN rows
        recent = df.tail(self.SEQ_LEN).copy()
        
        # Prepare features
        recent = self.prepare_features(recent)
        
        # Get feature lists
        h1_features = self.feature_config['final_h1']
        h2_features = self.feature_config['sentiment']
        
        # Extract and scale features
        h1_scaled = self.scaler_h1.transform(recent[h1_features].values)
        
        bounded = ['fear_greed', 'fg_ma7', 'fg_ma14', 'google_trends', 'gt_ma7', 'gt_ma30']
        bounded = [c for c in bounded if c in h2_features]
        flag_cols = ['fg_extreme_fear', 'fg_extreme_greed']
        flag_cols = [c for c in flag_cols if c in h2_features]
        continuous = [c for c in h2_features if c not in bounded + flag_cols]
        
        h2_bounded = self.scaler_h2_mm.transform(recent[bounded].values) if bounded else np.empty((len(recent), 0))
        h2_cont = self.scaler_h2_std.transform(recent[continuous].values) if continuous else np.empty((len(recent), 0))
        h2_flag = recent[flag_cols].values if flag_cols else np.empty((len(recent), 0))
        
        h2_scaled = np.hstack([h2_bounded, h2_cont, h2_flag])
        
        # Create sequences
        X1 = torch.FloatTensor(h1_scaled).unsqueeze(0)  # Add batch dimension
        X2 = torch.FloatTensor(h2_scaled).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            prob = torch.sigmoid(self.model(X1, X2)).item()
            pred = 1 if prob > self.THRESHOLD else 0
        
        if return_probs:
            return pred, prob
        return pred

def main():
    """Example usage"""
    print("="*60)
    print("NeuralEdge - Bitcoin Price Prediction")
    print("="*60)
    
    # Initialize predictor
    predictor = BitcoinPredictor()
    predictor.load_model()
    
    # Load latest data
    print("\nLoading latest data...")
    df = pd.read_csv('data/clean_ohlcv.csv', parse_dates=['Date'])
    
    # Make prediction
    print("Making prediction...")
    pred, prob = predictor.predict(df, return_probs=True)
    
    print(f"\n=== PREDICTION ===")
    print(f"Direction: {'UP' if pred == 1 else 'DOWN'}")
    print(f"Probability: {prob:.2%}")
    print(f"Confidence: {'High' if prob > 0.7 or prob < 0.3 else 'Medium' if prob > 0.5 else 'Low'}")
    print("==================")
    
    return pred, prob

if __name__ == '__main__':
    main()
