import pandas as pd
from scripts.predict import BitcoinPredictor
from scripts.api import load_data, global_df

import scripts.api as api
api.load_data()
df = api.global_df
print(f"Loaded df shape: {df.shape}")

predictor = BitcoinPredictor(model_dir='models')
if not __import__('os').path.exists('models/best_model.pt'):
    predictor = BitcoinPredictor(model_dir='grid_search_output')
predictor.load_model()
try:
    preds = predictor.predict_historical(df)
    print(f"Computed {len(preds)}")
except Exception as e:
    print(f"ERROR: {e}")

