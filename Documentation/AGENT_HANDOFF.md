# NeuralEdge — Agent-to-Agent Handoff Document

> **Project**: CN 6000 Mental Wealth Professional Life 3 — BSc Dissertation
> **Student**: Shyam Vijay Jagani (2611208)
> **Supervisor**: Dr. Mustansar Ghazanfar
> **University**: School of Architecture, Computing and Engineering
> **Last Updated**: 2026-06-20

---

## 1. What Is NeuralEdge?

NeuralEdge is a **Bitcoin next-day direction prediction platform** built for an undergraduate dissertation titled *"The Impact and Predictive Power of On-Chain Data on Cryptocurrency Valuation."* It uses a custom **Dual-Head Transformer** neural network that fuses **on-chain blockchain metrics** (Head 1) with **behavioral sentiment data** (Head 2) to predict whether BTC will go UP or DOWN tomorrow.

The platform consists of:
- A **production ML model** (PyTorch Dual-Head Transformer)
- A **FastAPI backend** serving live predictions
- A **web dashboard** (HTML/Tailwind/JS) with Terminal, Simulator, and Engine Room pages
- A **daily data pipeline** that fetches fresh OHLCV, on-chain, sentiment, and Google Trends data
- A **one-command prediction script** (`daily_predict.py`)

---

## 2. Project Root & Key Paths

```
PROJECT_ROOT = /home/shyam/UbuntuCode/CN 6000 Mental Wealth Professional Life 3 (Project)/Project
VENV         = $PROJECT_ROOT/.venv
PYTHON       = $PROJECT_ROOT/.venv/bin/python3
```

### Directory Layout

```
Project/
├── launch_neuraledge.sh          # One-click startup (backend + frontend)
├── launch_production.sh          # Alternative launcher
├── pyproject.toml                # Python project config
├── run.md                        # Quick-reference run commands
│
├── Backend/
│   ├── Core_API_Service/         # ★ PRIMARY WORKING DIRECTORY
│   │   ├── scripts/
│   │   │   ├── api.py            # FastAPI server (port 8002)
│   │   │   ├── predict.py        # BitcoinPredictor class (imported by api.py)
│   │   │   ├── daily_predict.py  # ★ ALL-IN-ONE: update data + predict tomorrow
│   │   │   ├── update_data_to_today.py  # Standalone data updater
│   │   │   ├── predict_tomorrow.py      # Standalone predictor (lightweight)
│   │   │   ├── grid_search.py    # Grid search training script
│   │   │   ├── retrain_rank3.py  # Retraining script
│   │   │   ├── compare_top3.py   # Model comparison script
│   │   │   └── collect_missing_data.py  # Legacy data collector
│   │   ├── data/                 # ★ CSV DATA FILES (updated by scripts)
│   │   │   ├── clean_ohlcv.csv       # 5798 rows, 2010-07-19 → 2026-06-02
│   │   │   ├── clean_onchain.csv     # ~5797 rows, 8.2MB
│   │   │   ├── clean_sentiment.csv   # ~3040 rows
│   │   │   └── clean_google.csv      # ~4903 rows
│   │   ├── models/               # Symlink/copy of NeuralEdge/models
│   │   └── grid_search_output/   # Grid search artifacts
│   │
│   ├── NeuralEdge/               # ★ MODEL ARTIFACTS (canonical source)
│   │   ├── models/
│   │   │   ├── model3_best.pt           # Production weights (1.1MB)
│   │   │   ├── scaler_head1.pkl         # StandardScaler for 13 on-chain features
│   │   │   ├── scaler_head2_minmax.pkl  # MinMaxScaler for bounded sentiment
│   │   │   ├── scaler_head2_std.pkl     # StandardScaler for continuous sentiment
│   │   │   ├── feature_sets.json        # Feature name lists
│   │   │   └── metrics_finetuned.json   # {accuracy, f1, balanced_score, threshold}
│   │   ├── plots/                # Dissertation visualization PNGs
│   │   │   ├── roc_curve_finetuned.png
│   │   │   ├── training_curves_finetuned.png
│   │   │   ├── model_comparison.png
│   │   │   ├── confusion_matrix.png
│   │   │   ├── feature_breakdown.png
│   │   │   └── trading_simulation.png
│   │   └── scripts/
│   │       └── predict.py        # Original predict script (BitcoinPredictor class)
│   │
│   ├── Performance_Evaluation/   # Evaluation notebooks/scripts
│   ├── Production/               # Production deployment artifacts
│   ├── Research_8H_Model/        # 8-hour interval research (abandoned)
│   └── Research_Transformer_Model/ # Early transformer research
│
├── Frontend/
│   ├── web-app/                  # ★ SERVED ON PORT 8000
│   │   ├── index.html            # Landing/cover page
│   │   ├── terminal.html         # ★ Main dashboard (live price, signal, chart)
│   │   ├── simulator.html        # Trading simulator page
│   │   ├── engine-room.html      # Model internals visualization
│   │   ├── cover.html            # Alternative cover
│   │   └── favicon.png           # NeuralEdge logo
│   ├── mobile-app/               # Android WebView wrapper
│   └── neuraledge-results-terminal.html  # Standalone results page
│
└── Documentation/                # Academic documentation
```

---

## 3. The Production Model

### Architecture: Dual-Head Transformer

```python
class DualHeadTransformer(nn.Module):
    # Head 1: On-chain features (13 dims) → Linear → d_model
    # Head 2: Sentiment features (12 dims) → Linear → d_model
    # Fusion: (head1 + head2) / 2
    # → PositionalEncoding → TransformerEncoder → last token → 2 MLP heads → sum
```

### Hyperparameters (Grid Search Run 82 Winner)

| Param | Value |
|-------|-------|
| `d_model` | 32 |
| `nhead` | 4 |
| `num_layers` | 2 |
| `dropout` | 0.1 |
| `seq_len` | 7 (days) |
| `batch_size` | 32 |
| `learning_rate` | 2e-3 |
| `threshold` | **0.65** |

### Performance Metrics (Frozen — from `metrics_finetuned.json`)

| Metric | Value |
|--------|-------|
| Accuracy | **63.96%** |
| F1 Score | **0.6209** |
| ROC-AUC | **0.654** |
| Balanced Score | 0.6388 |
| Threshold | 0.65 |

> [!IMPORTANT]
> These metrics are from the dissertation evaluation and are **frozen**. The model weights in `model3_best.pt` should NOT be retrained unless explicitly requested.

### Feature Sets (from `feature_sets.json`)

**Head 1 — On-Chain (13 features):**
`AdrActCnt`, `AdrBalCnt`, `TxCnt`, `HashRate`, `BlkCnt`, `SplyExNtv`, `FlowInExNtv`, `CapMVRVCur`, `AdrActCnt_growth7d`, `AdrActCnt_growth30d`, `CapMVRVCur_growth7d`, `CapMVRVCur_growth30d`, `momentum_7d`

**Head 2 — Sentiment (12 features):**
`google_trends`, `gt_ma7`, `gt_ma30`, `gt_change7`, `gt_momentum`, `fear_greed`, `fg_ma7`, `fg_ma14`, `fg_change`, `fg_change7`, `fg_extreme_fear`, `fg_extreme_greed`

### Scaler Pipeline

| Scaler | File | Applied To |
|--------|------|-----------|
| `StandardScaler` | `scaler_head1.pkl` | All 13 Head 1 features |
| `MinMaxScaler` | `scaler_head2_minmax.pkl` | Bounded: `fear_greed`, `fg_ma7`, `fg_ma14`, `google_trends`, `gt_ma7`, `gt_ma30` |
| `StandardScaler` | `scaler_head2_std.pkl` | Continuous: `gt_change7`, `gt_momentum`, `fg_change`, `fg_change7` |
| Raw (no scaling) | — | Flags: `fg_extreme_fear`, `fg_extreme_greed` |

---

## 4. Data Pipeline

### Data Sources

| Source | API | CSV | Key Columns |
|--------|-----|-----|-------------|
| **OHLCV** | Yahoo Finance (`yfinance`) | `clean_ohlcv.csv` | Date, Open, High, Low, Close, Volume |
| **On-Chain** | CoinMetrics Community v4 | `clean_onchain.csv` | 26 raw metrics + ~50 derived technicals |
| **Sentiment** | Alternative.me Fear & Greed | `clean_sentiment.csv` | fear_greed + 6 derived features |
| **Google Trends** | PyTrends | `clean_google.csv` | google_trends + 4 derived features |

### How Data Gets Updated

The script `daily_predict.py` does everything in one command:

```bash
cd Project/Backend/Core_API_Service
python scripts/daily_predict.py
```

It calls four update functions in order:
1. **`update_ohlcv()`** — yfinance `BTC-USD`, end date is `TOMORROW` (exclusive param fix)
2. **`update_onchain()`** — CoinMetrics API, strips UTC timezone (`dt.tz_localize(None)`), computes ~50 technical indicators, growth features
3. **`update_sentiment()`** — Alternative.me, recomputes rolling averages
4. **`update_google()`** — PyTrends, recomputes MAs and momentum

Then runs the model and prints the prediction box.

> [!WARNING]
> **Timezone Fix (Critical)**: CoinMetrics returns UTC-aware timestamps. Lines in both `daily_predict.py` and `update_data_to_today.py` must use `.dt.tz_localize(None).dt.normalize()` before merging with tz-naive OHLCV data. This was a bug that was fixed.

> [!WARNING]
> **yfinance `end` param is exclusive**: To get today's candle, `end` must be set to TOMORROW's date. The comparison `if start > TODAY` (not `>=`) ensures today's data is always fetched.

---

## 5. API Server (`api.py`)

**Runs on port 8002** from `Backend/Core_API_Service/` directory.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check: `{"status": "online", "model_loaded": true}` |
| `/api/btc` | GET | Live BTC price from CoinGecko (fallback: CSV) |
| `/api/signal` | GET | **★ Live model prediction** — runs `predictor.predict()` on `global_df` |
| `/api/portfolio` | GET | Mock portfolio data (hardcoded) |
| `/api/fear-greed` | GET | Live Fear & Greed from Alternative.me |
| `/api/btc/history/full` | GET | 365-day price history from CoinGecko (fallback: CSV) |
| `/api/engine` | GET | Engine room data (partially mocked) |
| `/api/simulator/history` | GET | Precomputed historical predictions for simulator |

### `/api/signal` — How It Works

This was changed from hardcoded values to **dynamic live predictions**:

```python
pred, prob = predictor.predict(global_df, return_probs=True)
signal = "ACCUMULATE" if pred == 1 else "DISTRIBUTE"
# Risk derived from distance to threshold (0.65):
#   > 0.15 = LOW risk, > 0.07 = MEDIUM, else HIGH
```

### Startup Flow

1. `load_data()` — reads all 4 CSVs, merges on Date, forward/back fills
2. `BitcoinPredictor(model_dir='models')` — loads model weights + scalers
3. `predictor.predict_historical(global_df)` — precomputes simulator data

---

## 6. Frontend Dashboard

**Served on port 8000** via `python3 -m http.server 8000` from `Frontend/web-app/`.

### Pages

| Page | URL | Purpose |
|------|-----|---------|
| `index.html` | `/` | Landing page with navigation |
| `terminal.html` | `/terminal.html` | **★ Main dashboard** — live BTC price, signal, chart, neural log |
| `simulator.html` | `/simulator.html` | Trading backtester with equity curves |
| `engine-room.html` | `/engine-room.html` | Model internals visualization |

### Tech Stack
- **TailwindCSS** (CDN)
- **Space Grotesk** + **Inter** fonts
- **Material Symbols** icons
- All API calls go to `http://localhost:8002`
- Auto-refreshes every 10 seconds

### Terminal Page Key Elements
- BTC price with 24h change (auto-updates)
- Tomorrow's Signal: ACCUMULATE / DISTRIBUTE
- Model Confidence percentage
- Interactive SVG chart with crosshair hover
- Prediction line (dashed green) showing tomorrow's projected direction
- Neural Inference Log (scrollable event feed)
- "Execute Model Vector" button

---

## 7. How to Launch Everything

### Quick Start
```bash
cd "/home/shyam/UbuntuCode/CN 6000 Mental Wealth Professional Life 3 (Project)/Project"
source .venv/bin/activate
./launch_neuraledge.sh
```
This kills existing processes, starts the API on 8002 and frontend on 8000.

### Daily Prediction (Update Data + Predict)
```bash
cd Project/Backend/Core_API_Service
python scripts/daily_predict.py
```

### After Data Update, Restart Dashboard
```bash
pkill -f "python3 scripts/api.py"; pkill -f "python3 -m http.server 8000"
sleep 2
./launch_neuraledge.sh
```

---

## 8. Known Issues & Gotchas

| Issue | Details |
|-------|---------|
| **CWD dependency** | `api.py` must run from `Backend/Core_API_Service/` — it uses relative paths like `data/`, `models/` |
| **Model dir fallback** | `api.py` checks for `models/best_model.pt` then falls back to `grid_search_output/` — the actual file is `model3_best.pt` |
| **Portfolio endpoint** | `GET /api/portfolio` is hardcoded mock data ($125K, +4.2%) |
| **Engine endpoint** | `GET /api/engine` has partially mocked on-chain values |
| **Sklearn warnings** | Scaler warnings about feature names appear on startup — cosmetic, not functional |
| **DeprecationWarning** | `@app.on_event("startup")` is deprecated in newer FastAPI — works fine but should migrate to lifespan handlers |
| **Google Trends rate limits** | PyTrends can be rate-limited; the script handles this gracefully |
| **Data lag** | On-chain data from CoinMetrics may lag 1-2 days behind; OHLCV and sentiment are usually same-day |

---

## 9. Model Architecture Code (Exact Copy Required)

When instantiating the model, attribute names **MUST match** the saved state_dict:

```python
class DualHeadTransformer(nn.Module):
    def __init__(self, input_dim_h1, input_dim_h2, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.head1_proj = nn.Linear(input_dim_h1, d_model)    # NOT "p1"
        self.head2_proj = nn.Linear(input_dim_h2, d_model)    # NOT "p2"
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)  # NOT "pe"
        encoder_layer = nn.TransformerEncoderLayer(...)
        self.transformer = nn.TransformerEncoder(...)          # NOT "t"
        self.head1 = nn.Sequential(...)                        # NOT "h1"
        self.head2 = nn.Sequential(...)                        # NOT "h2"
        self.dropout = nn.Dropout(dropout)                     # NOT "d"
```

> [!CAUTION]
> If you rename any attributes, `model.load_state_dict()` will fail with "Missing key(s)" errors. The saved weights use the exact names above.

---

## 10. Prediction Output Interpretation

| Probability | vs Threshold (0.65) | Signal | Risk | Meaning |
|------------|---------------------|--------|------|---------|
| > 0.80 | +0.15 | ACCUMULATE | LOW | Strong buy signal |
| 0.72–0.80 | +0.07–0.15 | ACCUMULATE | MEDIUM | Moderate buy |
| 0.65–0.72 | 0–0.07 | ACCUMULATE | HIGH | Weak buy, thin margin |
| < 0.65 | below | DISTRIBUTE | varies | Sell/reduce signal |

### Recent Predictions (from this conversation)

| Date Run | Prediction For | BTC Close | Direction | Prob | Confidence |
|----------|---------------|-----------|-----------|------|------------|
| May 31 | June 1 | $73,754 | 📈 UP | 67.52% | LOW |
| May 31 (re-run) | June 2 | $73,580 | 📈 UP | 68.60% | LOW |
| June 1 | June 2 | $71,069 | 📈 UP | 68.46% | LOW |
| June 2 | June 3 | $69,102 | 📈 UP | 68.76% | LOW |

---

## 11. File-by-File Reference

### Scripts You'll Use Most

| File | Purpose | When to Use |
|------|---------|-------------|
| [daily_predict.py](file:///home/shyam/UbuntuCode/CN%206000%20Mental%20Wealth%20Professional%20Life%203%20(Project)/Project/Backend/Core_API_Service/scripts/daily_predict.py) | Update all data + predict tomorrow | Daily operation |
| [api.py](file:///home/shyam/UbuntuCode/CN%206000%20Mental%20Wealth%20Professional%20Life%203%20(Project)/Project/Backend/Core_API_Service/scripts/api.py) | FastAPI server | Dashboard backend |
| [predict.py](file:///home/shyam/UbuntuCode/CN%206000%20Mental%20Wealth%20Professional%20Life%203%20(Project)/Project/Backend/Core_API_Service/scripts/predict.py) | BitcoinPredictor class | Imported by api.py |
| [terminal.html](file:///home/shyam/UbuntuCode/CN%206000%20Mental%20Wealth%20Professional%20Life%203%20(Project)/Project/Frontend/web-app/terminal.html) | Main dashboard UI | Frontend |
| [launch_neuraledge.sh](file:///home/shyam/UbuntuCode/CN%206000%20Mental%20Wealth%20Professional%20Life%203%20(Project)/Project/launch_neuraledge.sh) | Startup script | Launching platform |

### Dependencies

```
pandas, numpy, torch, joblib, scikit-learn, yfinance, pytrends, requests, fastapi, uvicorn, matplotlib
```

All installed in `.venv`.

---

## 12. Academic Context

- This is a **BSc dissertation project** for CN6000
- The research question: *Can on-chain blockchain data improve cryptocurrency price prediction?*
- The answer demonstrated: Multimodal fusion (on-chain + sentiment) outperforms single-source models
- **Boruta-LASSO** feature selection eliminated ~38 noisy features (especially technical indicators like SMA/MACD)
- The model was validated with **5-iteration cross-validation** and a **trading simulation** showing -8% drawdown vs -20.5% buy-and-hold during the Spring 2026 correction
- Dissertation plots are in `Backend/NeuralEdge/plots/`

---

## 13. What Was Done In This Conversation

1. Generated dissertation appendix plots (model comparison, confusion matrix, feature breakdown, trading simulation)
2. Created `update_data_to_today.py` — fetches missing data from all 4 sources
3. Fixed **timezone mismatch bug** (CoinMetrics UTC vs naive OHLCV dates)
4. Fixed **yfinance end-date bug** (exclusive param — need TOMORROW as end)
5. Created `daily_predict.py` — all-in-one update + predict script
6. Created `predict_tomorrow.py` — lightweight standalone predictor
7. Made `/api/signal` endpoint **dynamic** (was hardcoded, now runs live model)
8. Updated data through June 2, 2026
9. Restarted dashboard multiple times with fresh data
