"""
dual_head_transformer.py — Dual-Head Transformer Model
Implements a PyTorch Transformer with two independent heads (On-Chain + Sentiment),
as required by the new 28-feature pipeline.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import copy

# Architecture Hyperparameters
SEQ_LEN = 5
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DROPOUT = 0.5
BATCH_SIZE = 64
EPOCHS = 1000
PATIENCE = 100
LEARNING_RATE = 0.001

DATA_DIR = Path('data')
PLOTS_DIR = Path('plots')
PLOTS_DIR.mkdir(exist_ok=True)

class DualHeadDataset(Dataset):
    def __init__(self, data_path, head1_cols, head2_cols, scaler1=None, scaler2=None, is_train=False):
        df = pd.read_csv(data_path, index_col='date', parse_dates=True)
        # Drop rows where target is NaN (usually the last row)
        df = df.dropna(subset=['target'])
        
        self.head1 = df[head1_cols].values
        self.head2 = df[head2_cols].values
        self.targets = df['target'].values
        
        if is_train:
            self.scaler1 = StandardScaler()
            self.scaler2 = StandardScaler()
            self.head1 = self.scaler1.fit_transform(self.head1)
            self.head2 = self.scaler2.fit_transform(self.head2)
        else:
            self.scaler1 = scaler1
            self.scaler2 = scaler2
            self.head1 = self.scaler1.transform(self.head1)
            self.head2 = self.scaler2.transform(self.head2)
            
        self.x1, self.x2, self.y = self._create_sequences()
        
    def _create_sequences(self):
        x1, x2, y = [], [], []
        for i in range(len(self.targets) - SEQ_LEN):
            x1.append(self.head1[i:(i + SEQ_LEN)])
            x2.append(self.head2[i:(i + SEQ_LEN)])
            # Predicting the direction for the day AFTER the sequence
            y.append(self.targets[i + SEQ_LEN - 1])
        return torch.FloatTensor(np.array(x1)), torch.FloatTensor(np.array(x2)), torch.FloatTensor(np.array(y))
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx]

class DualHeadTransformer(nn.Module):
    def __init__(self, h1_dim=25, h2_dim=3, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        # Input projections
        self.proj1 = nn.Linear(h1_dim, d_model)
        self.proj2 = nn.Linear(h2_dim, d_model)
        
        # Positional encodings (simple learnable parameter for small sequence length)
        self.pos_encoder1 = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        self.pos_encoder2 = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        
        # Transformer Encoders
        encoder_layer1 = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer1 = nn.TransformerEncoder(encoder_layer1, num_layers=num_layers)
        
        encoder_layer2 = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer2 = nn.TransformerEncoder(encoder_layer2, num_layers=num_layers)
        
        # Classification head (combines both transformers)
        self.fc1 = nn.Linear(d_model * 2, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x1, x2):
        # Projections
        x1 = self.proj1(x1) + self.pos_encoder1
        x2 = self.proj2(x2) + self.pos_encoder2
        
        # Encoders
        out1 = self.transformer1(x1)
        out2 = self.transformer2(x2)
        
        # Context extraction (last time step)
        out1 = out1[:, -1, :]
        out2 = out2[:, -1, :]
        
        # Fusion
        fused = torch.cat((out1, out2), dim=1)
        
        # Classification
        out = self.dropout(torch.relu(self.fc1(fused)))
        out = self.sigmoid(self.fc2(out)).squeeze()
        return out

def train_model():
    print("🚀 Initializing Dual-Head Transformer Training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    with open(DATA_DIR / 'head_1_cols.txt', 'r') as f:
        head1_cols = f.read().splitlines()
    with open(DATA_DIR / 'head_2_cols.txt', 'r') as f:
        head2_cols = f.read().splitlines()
        
    print(f"   Head 1 (On-Chain/Market): {len(head1_cols)} features")
    print(f"   Head 2 (Sentiment): {len(head2_cols)} features")

    train_data = DualHeadDataset(DATA_DIR / 'btc_train.csv', head1_cols, head2_cols, is_train=True)
    val_data = DualHeadDataset(DATA_DIR / 'btc_val.csv', head1_cols, head2_cols, scaler1=train_data.scaler1, scaler2=train_data.scaler2)
    test_data = DualHeadDataset(DATA_DIR / 'btc_test.csv', head1_cols, head2_cols, scaler1=train_data.scaler1, scaler2=train_data.scaler2)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    model = DualHeadTransformer(h1_dim=len(head1_cols), h2_dim=len(head2_cols)).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    train_losses, val_losses = [], []
    
    print("Epoch | Train Loss | Val Loss")
    print("-" * 30)
    
    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        for x1, x2, y in train_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x1, x2)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * x1.size(0)
            
        t_loss /= len(train_loader.dataset)
        train_losses.append(t_loss)
        
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for x1, x2, y in val_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                out = model(x1, x2)
                loss = criterion(out, y)
                v_loss += loss.item() * x1.size(0)
                
        v_loss /= len(val_loader.dataset)
        val_losses.append(v_loss)
        scheduler.step(v_loss)
        
        if v_loss < best_loss:
            best_loss = v_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if epoch % 20 == 0 or patience_counter >= PATIENCE:
            print(f"{epoch:5d} | {t_loss:.4f}     | {v_loss:.4f}")
            
        if patience_counter >= PATIENCE:
            print(f"🛑 Early stopping triggered at epoch {epoch}")
            break
            
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'dual_head_transformer.pth')
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Dual-Head Transformer Learning Curve')
    plt.legend()
    plt.savefig(PLOTS_DIR / 'dual_head_transformer_loss.png')
    plt.close()
    
    # Test Evaluation
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x1, x2, y in test_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            out = model(x1, x2)
            y_prob.extend(out.cpu().numpy())
            y_pred.extend((out > 0.5).int().cpu().numpy())
            y_true.extend(y.cpu().numpy())
            
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("\n✅ Evaluation on Test Set:")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    
    # Save predictions for evaluate.py
    df_test = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob})
    df_test.to_csv(DATA_DIR / 'dual_head_predictions.csv', index=False)
    print("   Predictions saved to data/dual_head_predictions.csv")

if __name__ == '__main__':
    train_model()
