Comprehensive Project Log: AI-Driven Cryptocurrency Valuation

Project ID: CN6000
Student Number: u2611208
Topic: The Impact and Predictive Power of On-Chain Data on Cryptocurrency Valuation using AI.

1. Introduction to On-Chain Data

On-chain data refers to information permanently recorded on a blockchain's ledger, including transaction volumes, wallet activity, and smart contract interactions. It affects cryptocurrency value by providing insights into:

Network Health: Active addresses and adoption rates.

Investor Sentiment: Movement of "whales" and exchange flows.

Supply Dynamics: Amount of supply held in "illiquid" or long-term storage.

2. Key Metrics for Prediction Models

To build a robust model, metrics are categorized into three pillars:

Utility: Active Addresses, Transaction Count, TVL.

Supply: Exchange Net Flow, Illiquid Supply, Whale Activity.

Profitability: MVRV Z-Score, SOPR (Spent Output Profit Ratio), LTH (Long-Term Holder) Supply.

3. Initial Project Proposal (CN6000 Form)

Proposed Title: AI-Driven Cryptocurrency Valuation: Assessing the Predictive Power of On-Chain Data via Temporal Transformer Networks.

Proposed Aim:
To design, implement, and evaluate a state-of-the-art Deep Learning framework utilizing Temporal Transformer architectures to forecast cryptocurrency valuation. The project aims to quantify the predictive significance of blockchain-native (on-chain) metrics compared to traditional market indicators, leveraging Self-Attention mechanisms to identify key drivers of price discovery.

Proposed Objectives:

Research & Categorize: Systematically identify impactful on-chain metrics (e.g., Realized Cap, Exchange Flow).

Data Engineering: Collect, clean, and normalize multi-source time-series data (Glassnode, CoinGecko).

Model Implementation: Develop a Temporal Transformer using Self-Attention to capture multivariate dependencies.

Comparative Analysis: Train baseline recurrent models (LSTM/GRU) to benchmark performance.

Explainability: Quantify feature importance using attention maps to identify drivers in different market cycles.

Evaluation: Use MAPE and Directional Accuracy to ensure statistically significant results for publication.

4. Deep Learning Architectures Explained

LSTM (Long Short-Term Memory)

Purpose: Handles sequential data without losing signal over time (solves Vanishing Gradient).

Mechanism: Uses a "Cell State" (conveyor belt) and three gates (Forget, Input, Output) to regulate memory.

GRU (Gated Recurrent Unit)

Purpose: A faster, more efficient version of LSTM.

Mechanism: Combines Cell and Hidden states. Uses only two gates (Update, Reset). Best for limited hardware.

Temporal Transformer

Purpose: State-of-the-art architecture for sequences using Self-Attention.

Advantage: Unlike LSTMs (step-by-step), Transformers look at the entire data window simultaneously to find relationships between distant data points.

Explainability: Provides "Attention Maps" showing exactly which metric (e.g., Whale Inflow) the model prioritized for a prediction.

5. Hardware & Feasibility

System Specs: Intel i7 12700H | 16GB RAM | RTX 3060 6GB.

Assessment: Highly capable for training, but 6GB VRAM is the primary bottleneck.

Optimization: Use Mixed Precision (FP16), smaller batch sizes (8-16), and optimized libraries (CuDNN) to prevent Out-of-Memory (OOM) errors.

6. Budget & Resources

Cost: $0 (excluding hardware/electricity).

Data APIs: Glassnode (Basic Free Tier), CoinGecko (Free API), Etherscan/Blockchain.com.

Software: Python (uv for management), PyTorch/TensorFlow, Pandas, NumPy (All Open Source).

7. Multi-modal Technical-Fundamental Fusion

Instead of noisy social media data, the model can be "Multi-modal" by combining:

Modality 1 (On-Chain): Fundamental network health.

Modality 2 (Technical Indicators): Market momentum (RSI, MACD, Bollinger Bands).

Value: This allows for an "Ablation Study" to prove if On-Chain data provides a "predictive edge" over traditional trading tools, which is highly valuable for academic publication.

8. Target Market

Institutional: Hedge funds and Quantitative traders seeking "Alpha" signals.

Retail Platforms: Analytics providers (like Nansen or Glassnode) for new features.

Academic: Researchers in Fintech and Data Science focusing on Explainable AI (XAI) in finance.