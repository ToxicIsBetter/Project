# NeuralEdge - How to Run

## Quick Start

### Step 1: Start the Backend Server

```bash
cd "/home/shyam/UbuntuCode/CN 6000 Mental Wealth Professional Life 3 (Project)/Project"
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8002 --reload
```

### Step 2: Open the Frontend

Open one of these files in your browser:
- **Main Dashboard**: `showcase/web-app/index.html`
- **Engine Room**: `showcase/web-app/engine-room.html`
- **Simulator**: `showcase/web-app/simulator.html`

Or use a local server:
```bash
cd showcase/web-app
python -m http.server 8000
# Then open: http://localhost:8000
```

## What You'll See

### index.html - Main Terminal
- BTC Price with 24h change
- AI Prediction Signal (ACCUMULATE/HOLD/DISTRIBUTE)
- Confidence percentage
- Interactive price chart
- Live neural log

### engine-room.html - Engine Room
- Dual-head visualization
- On-chain metrics (Head 1)
- Sentiment analysis (Head 2)
- Fusion layer status
- Neural engine logs

### simulator.html - Simulator
- Prediction simulator
- Model comparison tools

## Model Info

- **Name**: Brain_Model_57
- **Type**: Fine-tuned Dual-Head Transformer
- **Accuracy**: 57.46%
- **F1 Score**: 0.54
- **Threshold**: 0.27

## Troubleshooting

### Backend won't start?
```bash
# Check if port 8002 is available
netstat -tulpn | grep 8002

# Test the API
curl http://localhost:8002/api/health
```

### Frontend shows errors?
- Make sure backend is running on port 8002
- Check browser console for CORS errors
- Ensure all data files exist in `Brain_Model_57/data/`

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/api/btc` | Current BTC price |
| `/api/signal` | AI prediction signal |
| `/api/fear-greed` | Fear & Greed index |
| `/api/engine` | Engine room data |
| `/api/btc/history/full` | Historical price data |
| `/api/health` | Health check |

