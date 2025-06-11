#!/usr/bin/env python3
"""
Create test script with proper encoding.
"""

def create_test_script():
    """Create a simple test script to verify the fix."""
    
    test_script = '''#!/usr/bin/env python3
"""
Simple test to verify BacktestEngine can initialize with current config.
"""

from pathlib import Path
from crypto_momentum_backtest.utils.config import Config
from crypto_momentum_backtest.utils.logger import setup_logger
from crypto_momentum_backtest.backtest.engine import BacktestEngine


def test_engine_init():
    """Test if BacktestEngine can initialize."""
    print("Testing BacktestEngine initialization...")
    
    try:
        # Load config
        config = Config.from_yaml(Path("config.yaml"))
        print("[OK] Config loaded successfully")
        
        # Set up logger
        logger = setup_logger("test")
        
        # Try to initialize engine
        engine = BacktestEngine(config, Path("data"), logger=logger)
        print("[OK] BacktestEngine initialized successfully!")
        
        # Check components
        print(f"  - UniverseManager: {engine.universe_manager is not None}")
        print(f"  - SignalGenerator: {engine.signal_generator is not None}")
        print(f"  - PositionSizer: {engine.position_sizer is not None}")
        print(f"  - RiskManager: {engine.risk_manager is not None}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_engine_init()
    if success:
        print("\\n[OK] All tests passed! You can now run: python run.py --no-validate")
    else:
        print("\\n[ERROR] Tests failed. Check the error messages above.")
'''
    
    # Write with UTF-8 encoding
    with open("test_engine_init.py", 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("[OK] Created test_engine_init.py")
    return True

if __name__ == "__main__":
    create_test_script()