import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from hyper_cnn_lstm import HyperCNNLSTM, SequenceDataset

df = pd.read_csv('data/btc_features.csv', index_col='date', parse_dates=True)
feature_cols = [c for c in df.columns if c != 'target']
X_raw = df[feature_cols].values
y_raw = df['target'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# CREATE SEQUENCES FIRST (This introduces the sequence overlap leakage seen in 80%+ papers)
lookback = 14
X_seq, y_seq = [], []
for i in range(len(X_scaled) - lookback):
    X_seq.append(X_scaled[i:i+lookback])
    y_seq.append(y_raw[i+lookback])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# RANDOM SPLIT (Test set sequences overlap with Train set sequences)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.15, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(torch.utils.data.TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
test_loader = DataLoader(torch.utils.data.TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HyperCNNLSTM(input_dim=len(feature_cols)).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training CNN-LSTM with Sequence Overlap Leakage...")
for epoch in range(25):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        preds = (model(X_batch).cpu().numpy() > 0.5).astype(int)
        all_preds.extend(preds.flatten())
        all_labels.extend(y_batch.numpy().flatten())

from sklearn.metrics import accuracy_score
print("OVERLAPPING TEST ACCURACY:", accuracy_score(all_labels, all_preds))
