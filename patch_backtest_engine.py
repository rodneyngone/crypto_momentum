#!/usr/bin/env python3
"""
Patch BacktestEngine to handle configuration compatibility issues.
"""

from pathlib import Path
import re


def patch_backtest_engine():
    """Patch BacktestEngine to handle config differences."""
    
    engine_file = Path("crypto_momentum_backtest/backtest/engine.py")
    
    if not engine_file.exists():
        print(f"❌ Could not find {engine_file}")
        return False
    
    # Read the file
    with open(engine_file, 'r') as f:
        content = f.read()
    
    # Store original for backup
    with open(engine_file.with_suffix('.py.bak'), 'w') as f:
        f.write(content)
    print("✅ Created backup: engine.py.bak")
    
    # Apply fixes
    fixes_applied = []
    
    # Fix 1: PositionSizer initialization
    old_position_sizer = r"""self\.position_sizer = PositionSizer\(
            atr_period=config\.risk\.atr_period,
            atr_multiplier=config\.risk\.atr_multiplier,
            max_position_size=config\.strategy\.max_position_size,
            logger=self\.logger
        \)"""
    
    new_position_sizer = """self.position_sizer = PositionSizer(
            atr_period=config.signals.adx_period,  # Use ADX period as ATR period
            atr_multiplier=2.0,  # Default for crypto
            max_position_size=config.portfolio.max_position_size,
            logger=self.logger
        )"""
    
    if re.search(old_position_sizer, content, re.DOTALL):
        content = re.sub(old_position_sizer, new_position_sizer, content, flags=re.DOTALL)
        fixes_applied.append("PositionSizer initialization")
    
    # Fix 2: RiskManager initialization
    old_risk_manager = r"""self\.risk_manager = RiskManager\(
            max_drawdown=config\.risk\.max_drawdown_threshold,
            max_position_size=config\.strategy\.max_position_size,
            max_correlation=config\.risk\.max_correlation,
            max_exchange_exposure=config\.risk\.max_exchange_exposure,
            volatility_scale_threshold=config\.risk\.volatility_regime_multiplier,
            logger=self\.logger
        \)"""
    
    new_risk_manager = """self.risk_manager = RiskManager(
            max_drawdown=config.risk.max_drawdown,
            max_position_size=config.portfolio.max_position_size,
            max_correlation=config.risk.correlation_threshold,
            max_exchange_exposure=getattr(config.risk, 'max_exchange_exposure', 0.40),
            volatility_scale_threshold=config.risk.volatility_multiple,
            logger=self.logger
        )"""
    
    if re.search(old_risk_manager, content, re.DOTALL):
        content = re.sub(old_risk_manager, new_risk_manager, content, flags=re.DOTALL)
        fixes_applied.append("RiskManager initialization")
    
    # Fix 3: Any remaining config.strategy references
    content = content.replace("config.strategy.max_position_size", "config.portfolio.max_position_size")
    content = content.replace("config.strategy.universe_size", "config.data.universe_size")
    content = content.replace("config.strategy.rebalance_frequency", "config.portfolio.rebalance_frequency")
    content = content.replace("config.strategy.long_short", "getattr(config, 'strategy', {}).get('long_short', True)")
    
    if "config.portfolio.max_position_size" in content:
        fixes_applied.append("config.strategy -> config.portfolio references")
    
    # Fix 4: Handle missing config attributes with defaults
    # Add helper method at the beginning of __init__
    init_start = content.find("def __init__(")
    init_body_start = content.find(":", init_start) + 1
    
    # Find the first line of actual code (after docstring if any)
    lines = content[init_body_start:].split('\n')
    insert_pos = init_body_start
    for i, line in enumerate(lines):
        if line.strip() and not line.strip().startswith('"""') and not line.strip().startswith('#'):
            insert_pos = init_body_start + len('\n'.join(lines[:i])) + 1
            break
    
    helper_code = '''
        # Helper function to safely get config values with defaults
        def get_config_value(obj, path, default):
            try:
                parts = path.split('.')
                value = obj
                for part in parts:
                    value = getattr(value, part)
                return value
            except AttributeError:
                return default
        
'''
    
    # Insert helper code
    content = content[:insert_pos] + helper_code + content[insert_pos:]
    fixes_applied.append("Added config helper function")
    
    # Write the patched content
    with open(engine_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Applied {len(fixes_applied)} fixes to engine.py:")
    for fix in fixes_applied:
        print(f"   - {fix}")
    
    return True


def create_simple_test_script():
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
        print("✅ Config loaded successfully")
        
        # Set up logger
        logger = setup_logger("test")
        
        # Try to initialize engine
        engine = BacktestEngine(config, Path("data"), logger=logger)
        print("✅ BacktestEngine initialized successfully!")
        
        # Check components
        print(f"  - UniverseManager: {engine.universe_manager is not None}")
        print(f"  - SignalGenerator: {engine.signal_generator is not None}")
        print(f"  - PositionSizer: {engine.position_sizer is not None}")
        print(f"  - RiskManager: {engine.risk_manager is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_engine_init()
    if success:
        print("\\n✅ All tests passed! You can now run: python run.py --no-validate")
    else:
        print("\\n❌ Tests failed. Check the error messages above.")
'''
    
    with open("test_engine_init.py", 'w') as f:
        f.write(test_script)
    
    print("✅ Created test_engine_init.py")
    return True


def main():
    """Main patch function."""
    print("🔧 Patching BacktestEngine for Configuration Compatibility")
    print("=" * 60)
    
    # Apply patches
    print("\n1️⃣ Patching BacktestEngine...")
    success = patch_backtest_engine()
    
    if success:
        # Create test script
        print("\n2️⃣ Creating test script...")
        create_simple_test_script()
        
        print("\n✅ Patches applied successfully!")
        print("\n📋 Next steps:")
        print("1. Test the initialization: python test_engine_init.py")
        print("2. If test passes, run: python run.py --no-validate")
        print("3. If still having issues, restore backup: copy engine.py.bak engine.py")
    else:
        print("\n❌ Failed to apply patches")
        print("You may need to manually edit engine.py")


if __name__ == "__main__":
    main()