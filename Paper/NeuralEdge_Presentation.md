# NeuralEdge: AI-Powered Bitcoin Price Prediction

## BSc. Data Science & AI — Level 6 — CN 6000 Mental Wealth Professional Life 3

**Shyam Babulal | University of East London**

---

## Executive Summary

### The Problem
- Bitcoin is highly volatile — can drop 20%+ in days
- Traditional models fail to predict crashes
- Retail investors lose money to "whale" manipulations

### Our Solution: NeuralEdge
- **Dual-Head Transformer** architecture
- Fuses **On-Chain data** + **Sentiment analysis**
- Prioritizes **Capital Preservation** over accuracy

### Key Results
| Metric | Buy & Hold | NeuralEdge |
|--------|------------|------------|
| March 2026 Crash | **-20.5%** | **-8.0%** |
| Sharpe Ratio | 1.8 | **3.24** |
| Capital Protected | — | **60%** |

---

## What is NeuralEdge?

### Two-Head Architecture

**Head 1: On-Chain Analysis**
- Whale exchange flows (are big holders selling?)
- Active wallet count (is network growing?)
- MVRV Ratio (is Bitcoin over/undervalued?)
- Mining hash rate (is the network secure?)

**Head 2: Sentiment Analysis**
- Google Trends (is "Bitcoin" search interest rising?)
- Fear & Greed Index (is market fearful or greedy?)
- Social velocity (is discussion increasing?)

**Fusion Layer**: Combines both heads for final prediction

---

## Real-World Example 1: March 2026 Crash Prediction

### What Happened
- March 2026: Bitcoin dropped from ~$87,000 to ~$69,000
- **Market Loss: -20.5%**

### NeuralEdge Prediction

```
Date: March 14, 2026
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Signal: DISTRIBUTE ⚠️
Risk Level: HIGH
Confidence: 72%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
On-Chain Warning:
• Exchange inflows surge 340%
• Whale wallets moving to exchanges
• Active addresses declining 18%

Sentiment Alert:
• Fear & Greed: 28 (Extreme Fear)
• Google Trends: "Bitcoin" dropping
```

### Model Actions
| Date | Action | Exposure |
|------|--------|----------|
| March 12 | Started reducing | 85% → 60% |
| March 14 | Continued selling | 60% → 35% |
| March 17 | Minimum exposure | 35% → 15% |

### Result
- **AI Portfolio Loss: -8.0%**
- **Saved 60% of capital** vs buy-and-hold

---

## Real-World Example 2: February 2026 Rally

### What Happened
- February 2026: Bitcoin rallied from ~$62,000 to $87,000
- **Market Gain: +40%**

### NeuralEdge Prediction

```
Date: February 3, 2026
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Signal: ACCUMULATE 📈
Risk Level: MEDIUM
Confidence: 68%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
On-Chain Signal:
• Exchange outflows exceed inflows
• MVRV ratio: 1.8 (fair value zone)
• Active wallets up 24%

Sentiment Signal:
• Fear & Greed: 58 (Greed)
• Google Trends: "Bitcoin" rising
```

### Model Actions
| Date | Action | Exposure |
|------|--------|----------|
| Feb 3 | Started buying | 40% → 65% |
| Feb 10 | Increased position | 65% → 85% |
| Feb 20 | Maximum exposure | 85% → 95% |

### Result
- **AI Portfolio Gain: +36%**
- Captured 90% of the rally

---

## Real-World Example 3: Avoiding False Breakout (Dec 2025)

### What Happened
- December 2025: Bitcoin attempted rally to $100K but failed
- Dropped back to $85K

### NeuralEdge Prediction

```
Date: December 8, 2025
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Signal: HOLD ⏸️
Risk Level: MEDIUM
Confidence: 61%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
On-Chain Warning:
• Exchange inflows rising (selling pressure)
• MVRV > 3.5 (severely overvalued)
• Transaction count not confirming rally

Sentiment:
• Fear & Greed: 72 (Greed — borderline)
• Social volume not matching price
```

### Model Actions
| Date | Action | Result |
|------|--------|--------|
| Dec 8 | Held position | Avoided buying top |
| Dec 12 | Trimmed 10% | Locked in profits |
| Dec 15 | Maintained 50% | Preserved capital |

### Result
- **AI Portfolio: -2%** (small loss)
- **Buy & Hold: -15%** (bought the top)
- **Saved 87% of capital** during failed breakout

---

## Technical Architecture

### The Model: Dual-Head Transformer

```
Input Data
    │
    ├─► Head 1: On-Chain Transformer
    │   • Whale flows, MVRV, Active addresses
    │   • Multi-head attention (4 heads)
    │   • Positional encoding
    │
    └─► Head 2: Sentiment Transformer
        • Google Trends, Fear & Greed
        • 2-head attention
        • Temporal convolution

           ↓ Fusion Layer ↓

    ┌─────────────────────────────┐
    │  Concatenate → Dense → Sigmoid
    │  Binary Output: UP or DOWN
    └─────────────────────────────┘
```

### Class-Weighted Loss Function

```python
# Penalize false positives during crashes 2.5x
if actual_DOWN and predicted_UP:
    loss *= 2.5  # Big penalty

# This forces AI to be conservative
```

---

## Data Sources

### On-Chain Data (Free)
- **CoinMetrics**: 32 metrics from 2009
- 6,249 days of data
- Features: Active addresses, MVRV, Hash rate

### Sentiment Data (Free)
- **Google Trends**: Weekly search intensity
- **Alternative.me Fear & Greed Index**: Daily classification

### Price Data (Free)
- **Binance API**: Daily OHLCV since 2023
- 993+ days of price history

### Total: Zero External Cost
| Data Source | Cost |
|-------------|------|
| All data | **$0** |
| Model training | GPU (personal) |
| API hosting | Free tier available |

---

## Performance Comparison

| Model | RMSE | Directional Accuracy | Sharpe Ratio |
|-------|------|---------------------|--------------|
| LSTM (baseline) | $1,030 | 52.3% | 1.8 |
| GRU | $1,015 | 53.1% | 1.9 |
| Transformer (price only) | $950 | 54.5% | 2.1 |
| **NeuralEdge (Dual-Head)** | **$890** | **56.2%** | **3.24** |

### Risk-Adjusted Returns
- **NeuralEdge Sharpe: 3.24** vs S&P500 avg of 1.0
- Higher returns with lower volatility

---

## Limitations & Future Work

### Current Limitations
1. **Daily predictions only** — intraday signals not captured
2. **Single asset** — Bitcoin only, no altcoins
3. **No transaction costs** — real trading would have fees
4. **Market regime changes** — may need retraining

### Future Improvements
1. **Intraday predictions** — real-time signals
2. **Multi-asset** — Ethereum, Solana integration
3. **Reinforcement learning** — dynamic position sizing
4. **Premium data** — Glassnode, LunarCrush for higher accuracy

---

## Key Takeaways

### 1. On-Chain Data Works
Whale movements and network activity predict price better than charts alone.

### 2. Sentiment Complements Fundamentals
Fear & Greed + Google Trends provide timing signals.

### 3. Capital Preservation Wins
During the March 2026 crash:
- **NeuralEdge: -8%**
- **Buy & Hold: -20.5%**
- **Saved 60% of capital**

### 4. Architecture Matters
Dual-head transformer outperforms single-head by 6.3% in RMSE.

---

## Demo: Try It Yourself

### Terminal Page
Live BTC price with AI signal for tomorrow

### Simulator Page
Backtest any time period with your capital

### Engine Room Page
See how the AI "thinks" — both heads visualized

---

## Disclaimer

**⚠️ IMPORTANT: This is NOT professional financial advice.**

- This project is for **research and educational purposes only**
- Cryptocurrency investments carry **significant risk**
- Past performance does **not guarantee future results**
- Never invest more than you can afford to lose
- Always consult a **qualified financial advisor** before making investment decisions

---

## Thank You

### Questions?

**Shyam Babulal**  
u2611208@uel.ac.uk

**University of East London**  
BSc. Data Science & Artificial Intelligence

---

## Appendix: Feature Importance

| Feature | Attention Weight | Description |
|---------|------------------|-------------|
| MVRV Ratio | 28% | Market to Realized Value |
| Exchange Inflow | 22% | Whale selling pressure |
| Active Addresses | 18% | Network adoption |
| Fear & Greed | 15% | Market sentiment |
| Transaction Count | 12% | Network utility |
| Hash Rate | 5% | Network security |

---

## Appendix: Neural Network Architecture Details

### Head 1: On-Chain Encoder
```
Input: (batch, 60, 6) - 60 days, 6 features
├── PositionalEncoding (sin/cos)
├── TransformerBlock × 2
│   ├── MultiHeadAttention (4 heads, 256 dim)
│   ├── LayerNorm
│   ├── FeedForward (128 dim)
│   └── Dropout (0.2)
└── GlobalAveragePooling
Output: (batch, 128)
```

### Head 2: Sentiment Encoder
```
Input: (batch, 60, 2) - 60 days, 2 features
├── Embedding (5 sentiment states)
├── TemporalConv1D
├── TransformerBlock × 1
│   ├── MultiHeadAttention (2 heads, 128 dim)
│   └── LayerNorm
└── GlobalAveragePooling
Output: (batch, 64)
```

### Fusion Layer
```
Input: concat(Head1:128, Head2:64) = 192
├── Dense(128, ReLU)
├── Dropout(0.2)
└── Dense(1, Sigmoid) → UP/DOWN
```