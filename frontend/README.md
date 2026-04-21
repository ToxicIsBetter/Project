# NeuralEdge Frontend

A modern, professional dashboard for AI-powered Bitcoin price prediction.

## Features

- 🎨 **Modern Dark Theme** - Professional design with smooth animations
- 📊 **Real-time Charts** - Interactive price charts using Chart.js
- 🤖 **AI Predictions** - Live Brain_Model_57 integration
- 📈 **On-Chain Metrics** - Blockchain data visualization
- 💭 **Sentiment Analysis** - Market sentiment tracking
- 🔄 **Auto-refresh** - Updates every 30 seconds

## Quick Start

### 1. Start the Backend

```bash
cd /home/shyam/UbuntuCode/CN\ 6000\ Mental\ Wealth\ Professional\ Life\ 3\ \(Project\)/Project
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8002 --reload
```

### 2. Open the Frontend

Simply open `index.html` in your browser:

```bash
# Option 1: Direct file open
open frontend/index.html  # macOS
xdg-open frontend/index.html  # Linux

# Option 2: Using a simple HTTP server
cd frontend
python -m http.server 8000
# Then open http://localhost:8000
```

## Files

- `index.html` - Complete dashboard (single file, no build required)
- `app.js` - JavaScript application logic (optional, inline in HTML)

## API Endpoints Used

| Endpoint | Description |
|----------|-------------|
| `/api/btc` | Current Bitcoin price |
| `/api/signal` | AI prediction signal |
| `/api/fear-greed` | Fear & Greed index |
| `/api/engine` | Engine room data |
| `/api/btc/history/full` | Historical price data |

## Model Info

- **Name**: Brain_Model_57
- **Type**: Fine-tuned Dual-Head Transformer
- **Accuracy**: 57.46%
- **F1 Score**: 0.54
- **Threshold**: 0.27

## Customization

### Colors

Edit CSS variables in `<style>` section:

```css
:root {
    --bg: #0a0a0f;           /* Background */
    --bg-card: #16161f;      /* Card background */
    --primary: #6366f1;      /* Primary accent */
    --secondary: #22d3ee;    /* Secondary accent */
    --success: #10b981;      /* Success/gain */
    --danger: #ef4444;       /* Danger/loss */
}
```

### Refresh Interval

Change the auto-refresh interval (default 30s):

```javascript
setInterval(load, 30000); // Change 30000 to desired ms
```

## Screenshots

The dashboard includes:
- Price card with 24h change
- Market cap and volume cards
- Fear & Greed index
- Interactive price chart
- AI prediction signal with confidence
- On-chain analysis metrics
- Sentiment analysis metrics

## Browser Support

- Chrome/Edge (recommended)
- Firefox
- Safari

## Troubleshooting

### Backend not connecting?

Make sure the backend is running on port 8002:
```bash
curl http://localhost:8002/api/health
```

### Chart not showing?

Check browser console for errors. Ensure Chart.js CDN is accessible.

### Styles not loading?

The HTML is self-contained. Ensure you're viewing the latest version (clear cache).

## License

For educational purposes only. Not financial advice.
