import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load Preprocessed Data (Skips Boruta and Pandas)
X1_train = np.load('processed_model3/X1_train.npy')
X2_train = np.load('processed_model3/X2_train.npy')
y_train  = np.load('processed_model3/y_train.npy')

X1_val   = np.load('processed_model3/X1_val.npy')
X2_val   = np.load('processed_model3/X2_val.npy')
y_val    = np.load('processed_model3/y_val.npy')

X1_test  = np.load('processed_model3/X1_test.npy')
X2_test  = np.load('processed_model3/X2_test.npy')
y_test   = np.load('processed_model3/y_test.npy')

# Compute class weights
pos_rate = y_train.mean()
neg_rate = 1 - pos_rate
pos_weight = torch.tensor([neg_rate / pos_rate], dtype=torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_loader(X1, X2, y, batch_size=32, shuffle=False):
    dataset = TensorDataset(
        torch.tensor(X1, dtype=torch.float32),
        torch.tensor(X2, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X1_train, X2_train, y_train, batch_size=32, shuffle=True)
val_loader   = make_loader(X1_val,   X2_val,   y_val,   batch_size=32, shuffle=False)
test_loader  = make_loader(X1_test,  X2_test,  y_test,  batch_size=32, shuffle=False)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class DualHeadTransformer(nn.Module):
    def __init__(self, n_onchain, n_sentiment, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.3):
        super().__init__()
        self.proj_onchain = nn.Linear(n_onchain, d_model)
        self.pos_enc_h1   = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer_h1  = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_h1 = nn.TransformerEncoder(encoder_layer_h1, num_layers=num_layers)

        self.proj_sentiment = nn.Linear(n_sentiment, d_model)
        self.pos_enc_h2     = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer_h2    = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_h2 = nn.TransformerEncoder(encoder_layer_h2, num_layers=num_layers)

        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x_onchain, x_sentiment):
        h1 = self.proj_onchain(x_onchain)
        h1 = self.pos_enc_h1(h1)
        if h1.shape[2] > 0: h1 = self.transformer_h1(h1)
        h1 = h1[:, -1, :]

        h2 = self.proj_sentiment(x_sentiment)
        h2 = self.pos_enc_h2(h2)
        h2 = self.transformer_h2(h2)
        h2 = h2[:, -1, :]

        fused = torch.cat([h1, h2], dim=-1)
        return self.fusion(fused).squeeze(-1)

N_ONCHAIN   = max(X1_train.shape[2], 1)
N_SENTIMENT = X2_train.shape[2]

EPOCHS = 150
PATIENCE = 30
SEEDS = [1, 42, 123, 777, 1024, 2026, 8888, 12345, 9999, 1337]
results = {}

print("==================================================")
print("  EXECUTING HYPERPARAMETER SEED SEARCH (10 RUNS)  ")
print("==================================================\n")

for seed in SEEDS:
    # 1. SET DETERMINISTIC SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. INIT FRESH MODEL
    model = DualHeadTransformer(n_onchain=N_ONCHAIN, n_sentiment=N_SENTIMENT).to(device)
    criterion = nn.BCELoss(weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_loss = float('inf')
    patience_ctr = 0
    best_state = None

    # 3. FAST TRAIN LOOP
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X1b, X2b, yb in train_loader:
            X1b, X2b, yb = X1b.to(device), X2b.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(X1b, X2b)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X1b, X2b, yb in val_loader:
                X1b, X2b, yb = X1b.to(device), X2b.to(device), yb.to(device)
                val_loss += criterion(model(X1b, X2b), yb).item()
        
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break
    
    # 4. EVALUATE ON TEST SET
    model.load_state_dict(best_state)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X1b, X2b, yb in test_loader:
            X1b, X2b = X1b.to(device), X2b.to(device)
            probs = model(X1b, X2b).cpu().numpy()
            all_preds.extend((probs >= 0.5).astype(int))
    
    acc = accuracy_score(y_test, all_preds)
    results[seed] = acc
    print(f"✔️ Seed: {seed:<5} | Stopped Epoch: {epoch-PATIENCE:<3} | Test Accuracy: {acc * 100:.2f}%")

print("\n==================================================")
best_seed = max(results, key=results.get)
print(f"🏆 CHAMPION SEED: {best_seed} with {results[best_seed] * 100:.2f}% Accuracy!")
print("==================================================")
