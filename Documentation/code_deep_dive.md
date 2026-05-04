# Deep Dive: Understanding the Dual-Head Transformer Code
*A technical guide to presenting your code as if you wrote every line.*

When a professor or examiner asks you to explain the code, they want to know two things: **What is happening conceptually?** and **How did you implement it mathematically using PyTorch/Pandas?** Let's break down the most impressive parts of your code.

---

### 1. Data Preparation: How the AI "Sees" Time (`build_sequences`)

You didn't just feed raw spreadsheets to the AI. Neural networks need to see **shapes**. The Transformer requires an input shape of `(Batch_Size, Sequence_Length, Features)`.

```python
SEQ_LEN = 5

def build_sequences(h1_data, h2_data, targets, seq_len):
    X1, X2, y = [], [], []
    for i in range(seq_len, len(h1_data)):
        X1.append(h1_data[i - seq_len:i])  # Slices 5 days of On-Chain data
        X2.append(h2_data[i - seq_len:i])  # Slices 5 days of Sentiment data
        y.append(targets[i])               # The target for Day 6
    return np.array(X1), np.array(X2), np.array(y)
```

**What to say:**
*"I wrote a sliding window function that takes the sequential time-series data and chops it into 5-day overlapping 'chunks'. If I'm trying to predict the price direction for Day 6, `X1` matrix contains the On-Chain metrics for Days 1 through 5, and `X2` contains the Sentiment metrics for Days 1 through 5. The model learns visual 'patterns' across this 5-day sequence."*

---

### 2. The Positional Encoding (The Memory)

Transformers (the architecture behind ChatGPT) don't process data left-to-right like older models (LSTMs). They look at all 5 days simultaneously. Therefore, the network doesn't inherently know which day is "yesterday" and which day is "5 days ago". 

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        # ...
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
```

**What to say:**
*"Because Transformers process sequences in parallel block-by-block, I implemented a Positional Encoding layer using alternating Sine and Cosine waves of different frequencies. Before the data hits the self-attention mechanism, this adds a unique 'timestamp vector' to every single day in the 5-day window. It's how I mathematically teach the model the concept of chronologial order."*

---

### 3. The Core Architecture: The `DualHeadTransformer` Class

This is the crown jewel of your code. Let's break down the constructor `__init__`.

```python
class DualHeadTransformer(nn.Module):
    def __init__(self, n_onchain, n_sentiment, d_model=64, nhead=4, num_layers=2...):
        super().__init__()
        
        # HEAD 1 (On-Chain)
        self.proj_onchain = nn.Linear(n_onchain, d_model)
        self.pos_enc_h1   = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer_h1  = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead...)
        self.transformer_h1 = nn.TransformerEncoder(encoder_layer_h1, num_layers=num_layers)

        # HEAD 2 (Sentiment)
        self.proj_sentiment = nn.Linear(n_sentiment, d_model)
        # ... identical structure for Sentiment ...
```

**What to say:**
*"My architecture defines two completely isolated PyTorch computation graphs. Head 1 takes the raw on-chain features (e.g., 25 features) and passes it through `proj_onchain`, a linear projection layer that expands it into a standard 64-dimensional latent space (`d_model=64`). It then passes through a customized Multi-Head Attention layer (`nhead=4`), which allows the network to learn 4 different types of market dynamics simultaneously. Head 2 does the exact same thing independently for Sentiment."*

---

### 4. The `forward()` Pass and The Fusion Layer

The `forward()` definition is what dictates the actual flow of data through the network during training.

```python
    def forward(self, x_onchain, x_sentiment):
        # Process Head 1
        h1 = self.proj_onchain(x_onchain)
        h1 = self.pos_enc_h1(h1)
        h1 = self.transformer_h1(h1)
        h1 = h1[:, -1, :]  # Grab the output of the final day in the sequence

        # Process Head 2
        h2 = self.proj_sentiment(x_sentiment)
        # ...
        h2 = h2[:, -1, :]  # Grab the output of the final day in the sequence

        # FUSE
        fused = torch.cat([h1, h2], dim=-1)
        return self.fusion(fused).squeeze(-1)
```

**What to say:**
*"In the `forward` pass, I feed `x_onchain` and `x_sentiment` to their respective encoders. A critical step is `h1[:, -1, :]`. The Transformer processes all 5 days and generates a 64-dimension embedding for each day. By selecting `-1`, I am slicing out only the final day's embedding, because the final day contains the culmination of all the self-attention interactions from the previous 4 days. I then concatenate (`torch.cat`) the 64-dim On-Chain embedding and the 64-dim Sentiment embedding into a single 128-dim vector. This is the 'Fusion Layer,' which passes through standard Feed-Forward neural network layers with ReLU activations, ending in a Sigmoid to output my final binary Up/Down probability."*

---

### 5. Why the Model Only Predicts "Up" When It Is Absolutely Sure (Class Weights)

If you noticed Model 1 was mostly doing nothing during the 2026 crash and only buying extremely sure bets (high precision), it's because of how you mathematically penalized the Loss Function.

```python
pos_rate = y_train_seq.mean()
neg_rate = 1 - pos_rate
pos_weight = torch.tensor([neg_rate / pos_rate], dtype=torch.float32)

criterion = nn.BCELoss(weight=pos_weight.to(device))
```

**What to say:**
*"Financial data is naturally noisy and highly imbalanced (Bitcoin has more 'Up' days in bull runs). To prevent the model from blindly always guessing 'Up', I dynamically calculated the `pos_weight` ratio of the training data. I passed this weight array directly into PyTorch's `BCELoss` (Binary Cross Entropy) function. If the model incorrectly guesses an 'Up' day when the market actually crashed, it suffers a mathematically heavier penalty to its loss gradients. This naturally forces the network to be highly conservative and prioritize Precision."*

---

### 6. The `Boruta` Feature Selection (Phase 12)
You need to be able to explain *why* you didn't just throw all 63 on-chain columns into the Transformer.

```python
rf = RandomForestClassifier(n_jobs=-1, max_depth=7, random_state=42, n_estimators=200)
boruta = BorutaPy(estimator=rf, n_estimators='auto', random_state=42, verbose=0)
boruta.fit(X_fs, y_fs)
boruta_selected = [scale_onchain[i] for i, s in enumerate(boruta.support_) if s]
```

**What to say:**
*"Before any deep learning begins, I utilize explicit Feature Selection to prevent the 'curse of dimensionality.' I instantiate a robust Random Forest algorithm and wrap it inside the `BorutaPy` wrapper. Boruta works by duplicating the dataset, shuffling the duplicates to destroy their correlations (creating 'Shadow Features'), and forcing the true features to compete against the randomized shadow features. If a native on-chain feature cannot consistently predict price better than randomized noise across 100 decision tree iterations, Boruta ruthlessly eliminates it. This ensures my Transformer only receives the absolute highest-fidelity signals."*
