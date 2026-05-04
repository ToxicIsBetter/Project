# 👔 AI Agent Handoff Guide: Bitcoin Dual-Head Transformer
*Purpose: Provide the next agent with immediate context to resume work on the 2026 Dissertation Pipeline.*

---

## 1. Project Context & Objectives
**Goal:** Predict next-day Bitcoin price direction (Binary: Up=1, Down=0).
**Architecture:** Dual-Head Transformer (PyTorch).
*   **Head 1**: Process On-Chain/Technical data via Transformer Encoder.
*   **Head 2**: Process Sentiment (Google Trends & Fear/Greed) via Transformer Encoder.
*   **Fusion**: Concat Head 1 + Head 2 outputs -> Dense -> Sigmoid.

---

## 2. Current Model States (Version Control)
*   **Model 1 (`Mango/GoogleTrends`)**: 2018-2026 baseline. Includes **Technicals** (SMA crosses). Accuracy: **52.05%**.
*   **Model 3 (`Mango/Model3_OnChain`)**: The **Dissertation Choice**. Stripped of Technicals. Forced to use Pure On-Chain metrics + Sentiment. Accuracy: **51.43%**.
*   **Model 4 (`Mango/Model4_SingleHead`)**: **Pure On-Chain Baseline**. No Sentiment, No Technicals. Accuracy: **51.06%**.

---

## 3. Data & Feature Engineering
*   **Source Data**: `Project/BTC_Price_in_2026/` (On-Chain/Sentiment) + `Mango/GoogleTrends/google_trends_bitcoin.csv`.
*   **Windowing**: `SEQ_LEN = 5` (Sliding 5-day history).
*   **Training Range**: `2018-02-01` to `2022-12-31` (~1,795 rows).
*   **Feature Selection**:
    *   Candidate Pool: 75 features (see `dataset_dictionary.md`).
    *   Logic: `BorutaPy` for feature selection, with a **LASSO fallback** implemented in `model3` and `model4` if Boruta is over-aggressive with pure on-chain noise.

---

## 4. Key Artifacts for Onboarding
*   **`final_results_presentation.md`**: Current stats and comparative findings.
*   **`code_deep_dive.md`**: Explains the PyTorch implementation logic.
*   **`project_conversation_log.md`**: Decision history (why we dropped Santiment, why we used Model 1 for crash testing).

---

## 5. Next Steps for Incoming Agent
The user's roadmap for the next phase:

### A. Hourly Granularity Pivot
*   **Task**: Transition the daily-interval data (1d) to hourly (1h).
*   **Requirement**: Procure/Download hourly OHLCV and On-Chain/Sentiment parity data.
*   **Logic**: Update `build_sequences` for higher density training.

### B. Multi-Asset Scaling
*   **Task**: Apply the `DualHeadTransformer` architecture to **Ethereum** and **Solana**.
*   **Logic**: Modify the pipeline to handle multiple input CSVs and potentially a "Triple-Head" approach if asset-specific data is available.

---

## 6. Development Tips
*   **Logging**: Use `matplotlib.use('Agg')` for headless plotting.
*   **Dependencies**: The project uses `uv` for environment management.
*   **Feature Stripping**: When training on "True On-Chain" data, ensure the Phase 11 feature list explicitly excludes `return_`, `sma_`, `rsi_`, and `macd_`.
