# NeuralEdge Backend - Brain_Model_57

## Model Connected

The backend has been updated to use the **Brain_Model_57** fine-tuned Dual-Head Transformer model with:
- **Accuracy**: 57.46%
- **F1 Score**: 0.54
- **Threshold**: 0.27
- **Features**: 13 on-chain + 12 sentiment = 25 total features

## Changes Made

1. **Updated backend/main.py**:
   - Changed model path from `Mango/Model3_OnChain` to `Brain_Model_57`
   - Updated model architecture to match DualHeadTransformer (13, 12)
   - Fixed data loading to drop NaN values properly
   - Fixed inference to calculate momentum features on full dataset
   - Updated model info in API responses

2. **Updated Frontend**:
   - `showcase/web-app/index.html` - Updated model branding
   - `showcase/web-app/engine-room.html` - Updated to reflect Brain_Model_57

## Running the Backend

```bash
cd "/home/shyam/UbuntuCode/CN 6000 Mental Wealth Professional Life 3 (Project)/Project"
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8002 --reload
```

Or in production:
```bash
uv run python -m uvicorn backend.main:app --host 0.0.0.0 --port 8002
```

## API Endpoints

- `GET /` - API info
- `GET /api/health` - Health check
- `GET /api/btc` - Current BTC price
- `GET /api/signal` - AI prediction signal
- `GET /api/fear-greed` - Fear & Greed index
- `GET /api/engine` - Engine room data
- `GET /api/dashboard` - Full dashboard data

## Testing

```bash
curl http://localhost:8002/api/health
curl http://localhost:8002/api/signal
```

## Frontend

Open `showcase/web-app/index.html` in a browser to view the terminal.
Make sure the backend is running on port 8002.
