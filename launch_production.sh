#!/bin/bash

# NeuralEdge - Isolated Production Startup Script
# This orchestrates the 100% Data Production Environment on isolated ports.

PROJECT_ROOT="/home/shyam/UbuntuCode/CN 6000 Mental Wealth Professional Life 3 (Project)/Project"
PROD_DIR="$PROJECT_ROOT/Backend/Production"
API_DIR="$PROD_DIR/api"
WEB_DIR="$PROD_DIR/web-app"
PYTHON_EXEC="$PROJECT_ROOT/.venv/bin/python3"

echo "========================================================="
echo "   🚀 Booting NeuralEdge Production Environment...       "
echo "========================================================="

# Cleanup previous isolated instances
pkill -f "python3 api_production.py" || true
pkill -f "python3 -m http.server 8001" || true
sleep 1

# Start API
echo "🧠 Loading 100% Data Production Engine (API)..."
cd "$API_DIR" || exit
nohup "$PYTHON_EXEC" api_production.py > production_api.log 2>&1 &
API_PID=$!

# Start Web
echo "🖥️  Starting Isolated Terminal Display (Web)..."
cd "$WEB_DIR" || exit
nohup python3 -m http.server 8001 > production_web.log 2>&1 &
WEB_PID=$!

sleep 3

# Verify
echo "---------------------------------------------------------"
if ps -p $API_PID > /dev/null; then
  echo "✅ PRODUCTION BACKEND: ONLINE (Port 8003)"
else
  echo "❌ PRODUCTION BACKEND: FAILED"
fi

if ps -p $WEB_PID > /dev/null; then
  echo "✅ PRODUCTION FRONTEND: ONLINE (Port 8001)"
else
  echo "❌ PRODUCTION FRONTEND: FAILED"
fi
echo "---------------------------------------------------------"

echo ""
echo "🔗 PRODUCTION TERMINAL: http://localhost:8001/production_index.html"
echo "🔗 PRODUCTION API:      http://localhost:8003"
echo ""
echo "Press Ctrl+C to exit monitoring (services will keep running)."

# Monitor logs
tail -f "$API_DIR/production_api.log"
