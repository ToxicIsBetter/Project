#!/bin/bash

# NeuralEdge - Magic Startup Script
# This script launches the FastAPI backend and the Frontend server simultaneously.

# --- PATHS ---
PROJECT_ROOT="/home/shyam/UbuntuCode/CN 6000 Mental Wealth Professional Life 3 (Project)/Project"
API_DIR="$PROJECT_ROOT/Backend/Core_API_Service"
WEB_DIR="$PROJECT_ROOT/Frontend/web-app"
PYTHON_EXEC="$PROJECT_ROOT/.venv/bin/python3"

# --- CLEANUP ---
echo "🚀 Initializing NeuralEdge Ecosystem..."
echo "🧹 Cleaning up existing processes..."
pkill -f "python3 scripts/api.py" || true
pkill -f "python3 -m http.server 8000" || true
sleep 1

# --- START BACKEND ---
echo "🧠 Loading NeuralEngine (API)..."
cd "$API_DIR" || exit
# Run API in background
nohup "$PYTHON_EXEC" scripts/api.py > "$API_DIR/api_startup.log" 2>&1 &
API_PID=$!

# --- START FRONTEND ---
echo "🖥️  Starting Terminal UI (Web)..."
cd "$WEB_DIR" || exit
# Run HTTP server in background
nohup python3 -m http.server 8000 > "$WEB_DIR/web_server.log" 2>&1 &
WEB_PID=$!

sleep 3

# --- VERIFICATION ---
echo "----------------------------------------------------"
if ps -p $API_PID > /dev/null; then
  echo "✅ BACKEND: ONLINE (Port 8002)"
else
  echo "❌ BACKEND: FAILED (Check $API_DIR/api_startup.log)"
fi

if ps -p $WEB_PID > /dev/null; then
  echo "✅ FRONTEND: ONLINE (Port 8000)"
else
  echo "❌ FRONTEND: FAILED (Check $WEB_DIR/web_server.log)"
fi
echo "----------------------------------------------------"

echo ""
echo "🔗 TERMINAL DASHBOARD: http://localhost:8000"
echo "🔗 API STATUS:       http://localhost:8002"
echo ""
echo "NeuralEdge is now [SYNCHRONIZED]. Press Ctrl+C in this terminal if you want to stop monitoring (services will remain running in background)."
echo "----------------------------------------------------"

# Optional: tail the logs to show activity
tail -f "$API_DIR/api_startup.log"
