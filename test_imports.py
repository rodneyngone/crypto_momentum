# test_imports.py
try:
    import pandas as pd
    import numpy as np
    import vectorbt as vbt
    print("✅ Core packages imported successfully")
    
    from crypto_momentum_backtest.utils.config import Config
    print("✅ Project modules imported successfully")
    
except ImportError as e:
    print(f"❌ Import error: {e}")