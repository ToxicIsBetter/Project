"""Verify that all required files are present and model can be loaded"""
import os
import sys

def verify():
    print("="*60)
    print("NeuralEdge - Installation Verification")
    print("="*60)
    
    errors = []
    
    # Check required files
    required_files = [
        'models/model3_best.pt',
        'models/scaler_head1.pkl',
        'models/scaler_head2_minmax.pkl',
        'models/scaler_head2_std.pkl',
        'models/feature_sets.json',
        'models/metrics_finetuned.json',
        'data/clean_ohlcv.csv',
        'data/clean_onchain.csv',
        'data/clean_sentiment.csv',
        'data/clean_google.csv',
        'scripts/predict.py',
        'README.md',
        'MODEL_SUMMARY.md',
        'requirements.txt'
    ]
    
    print("\n1. Checking required files...")
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - MISSING!")
            errors.append(f"Missing file: {file}")
    
    # Try to import dependencies
    print("\n2. Checking dependencies...")
    try:
        import pandas
        print("   ✅ pandas")
    except ImportError:
        errors.append("pandas not installed")
        print("   ❌ pandas - MISSING")
    
    try:
        import numpy
        print("   ✅ numpy")
    except ImportError:
        errors.append("numpy not installed")
        print("   ❌ numpy - MISSING")
    
    try:
        import torch
        print("   ✅ torch")
    except ImportError:
        errors.append("torch not installed")
        print("   ❌ torch - MISSING")
    
    try:
        import sklearn
        print("   ✅ sklearn")
    except ImportError:
        errors.append("sklearn not installed")
        print("   ❌ sklearn - MISSING")
    
    try:
        import joblib
        print("   ✅ joblib")
    except ImportError:
        errors.append("joblib not installed")
        print("   ❌ joblib - MISSING")
    
    # Try to load model
    print("\n3. Loading model...")
    try:
        import torch
        from scripts.predict import BitcoinPredictor
        predictor = BitcoinPredictor(model_dir='models')
        predictor.load_model()
        print("   ✅ Model loaded successfully")
    except Exception as e:
        errors.append(f"Model loading failed: {str(e)}")
        print(f"   ❌ Model loading failed: {str(e)}")
    
    # Summary
    print("\n" + "="*60)
    if errors:
        print(f"❌ VERIFICATION FAILED - {len(errors)} error(s)")
        for error in errors:
            print(f"   - {error}")
        return False
    else:
        print("✅ VERIFICATION PASSED - All systems ready!")
        print("\nYou can now run predictions with:")
        print("   python scripts/predict.py")
        print("   or")
        print("   bash run_prediction.sh")
        return True

if __name__ == '__main__':
    success = verify()
    sys.exit(0 if success else 1)
