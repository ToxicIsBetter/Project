#!/bin/bash
# NeuralEdge Prediction Runner
echo "=============================================="
echo "NeuralEdge - Bitcoin Price Prediction"
echo "=============================================="
echo ""
cd "$(dirname "$0")"
python scripts/predict.py
