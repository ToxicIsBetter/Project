import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import json

DATA_DIR = Path('data')
LOOKBACK = 14  # Increased lookback for better pattern recognition
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SequenceDataset(Dataset):
    def __init__(self, X, y, lookback):
        self.X = []
        self.y = []
        for i in range(len(X) - lookback + 1):
            self.X.append(X[i:i+lookback])
            self.y.append(y[i+lookback-1])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class HyperCNNLSTM(nn.Module):
    def __init__(self, input_dim, cnn_filters=64, lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()
        
        # 1D Convolution over the time dimension
        # Input shape: (Batch, Lookback, Features) - Conv1d expects (Batch, Channels, Length)
        # So we permute inside the forward pass
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=cnn_filters, out_channels=cnn_filters*2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        # After Conv1d and Pool1d: Length becomes Lookback // 2
        # So LSTM input size is cnn_filters*2
        self.lstm = nn.LSTM(
            input_size=cnn_filters*2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Output layers
        self.fc1 = nn.Linear(lstm_hidden * 2, 64) # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x is (Batch, Time, Features)
        x = x.permute(0, 2, 1) # (Batch, Features, Time) for Conv1d
        
        # CNN Block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        
        # Prepare for LSTM
        x = x.permute(0, 2, 1) # (Batch, Time, Channels)
        
        # LSTM Block
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Use only the last time step from the LSTM sequence
        last_out = lstm_out[:, -1, :] 
        
        # Dense Block
        out = self.fc1(last_out)
        out = self.batch_norm(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return self.sigmoid(out)

def train_cnn_lstm():
    print(f"🔧 Device: {DEVICE}")
    print("📊 Loading data...")
    
    with open(DATA_DIR / 'feature_cols.txt') as f:
        feature_cols = f.read().strip().split('\n')
        
    train_df = pd.read_csv(DATA_DIR / 'btc_train.csv', index_col='date', parse_dates=True)
    val_df   = pd.read_csv(DATA_DIR / 'btc_val.csv', index_col='date', parse_dates=True)
    test_df  = pd.read_csv(DATA_DIR / 'btc_test.csv', index_col='date', parse_dates=True)
    
    X_train_raw = train_df[feature_cols].values
    y_train_raw = train_df['target'].values
    X_val_raw   = val_df[feature_cols].values
    y_val_raw   = val_df['target'].values
    X_test_raw  = test_df[feature_cols].values
    y_test_raw  = test_df['target'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled   = scaler.transform(X_val_raw)
    X_test_scaled  = scaler.transform(X_test_raw)

    train_data = SequenceDataset(X_train_scaled, y_train_raw, LOOKBACK)
    val_data   = SequenceDataset(X_val_scaled, y_val_raw, LOOKBACK)
    test_data  = SequenceDataset(X_test_scaled, y_test_raw, LOOKBACK)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    print(f"   Sequences — Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    model = HyperCNNLSTM(input_dim=len(feature_cols)).to(DEVICE)
    print(f"\n🏗️  Hyper CNN-LSTM Model: {sum(p.numel() for p in model.parameters())} parameters")

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []

    print(f"\n🔥 Training Hyper CNN-LSTM ({EPOCHS} epochs max, patience={PATIENCE})...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * len(X_batch)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"   ⏹ Early stopping at epoch {epoch+1} (best: {epoch+1-patience_counter})")
                break

    model.load_state_dict(best_model_state)

    # ---------------- EVALUATION ----------------
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            probs = model(X_batch).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(y_batch.numpy().flatten())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\n{'='*50}")
    print(f"  CNN-LSTM TEST RESULTS")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    results = {
        'name': 'CNN-LSTM Hybrid',
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'predictions': [int(x) for x in all_preds],
        'probabilities': [float(x) for x in all_probs],
        'labels': [float(x) for x in all_labels]
    }

    with open(DATA_DIR / 'cnnlstm_results.json', 'w') as f:
        json.dumps(results)
        json.dump(results, f)
        
    print(f"\n✅ CNN-LSTM results saved to data/cnnlstm_results.json")

if __name__ == '__main__':
    train_cnn_lstm()
