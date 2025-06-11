#!/usr/bin/env python3
"""
Fix the syntax error in BacktestEngine.
"""

from pathlib import Path
import shutil


def fix_engine_syntax():
    """Fix the syntax error by properly placing the helper function."""
    
    engine_file = Path("crypto_momentum_backtest/backtest/engine.py")
    
    # First, restore from backup if it exists
    backup_file = engine_file.with_suffix('.py.bak')
    if backup_file.exists():
        shutil.copy(backup_file, engine_file)
        print("[OK] Restored from backup")
    else:
        print("[WARNING] No backup found, working with current file")
    
    # Read the file
    with open(engine_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove any existing get_config_value function that might be misplaced
    if "def get_config_value(" in content:
        # Find and remove the misplaced function
        start = content.find("def get_config_value(")
        if start != -1:
            # Find the end of the function
            end = content.find("\n        \n", start)
            if end == -1:
                end = content.find("\n\n", start)
            if end != -1:
                content = content[:start] + content[end+2:]
    
    # Now properly fix the __init__ method
    # Find the __init__ method
    init_start = content.find("def __init__(")
    if init_start == -1:
        print("[ERROR] Could not find __init__ method")
        return False
    
    # Find the start of the method body (after the docstring)
    init_body_start = content.find('"""', init_start)
    if init_body_start != -1:
        # Find the end of the docstring
        docstring_end = content.find('"""', init_body_start + 3)
        if docstring_end != -1:
            insert_pos = docstring_end + 3
            # Skip any whitespace after docstring
            while insert_pos < len(content) and content[insert_pos] in '\n\r\t ':
                insert_pos += 1
        else:
            insert_pos = init_body_start
    else:
        # No docstring, find the first line after the method signature
        colon_pos = content.find(":", init_start)
        insert_pos = colon_pos + 1
        while insert_pos < len(content) and content[insert_pos] in '\n\r\t ':
            insert_pos += 1
    
    # Insert the proper initialization code
    init_code = '''
        self.config = config
        self.data_dir = Path(data_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components with safe config access
        try:
            # Get configuration values with defaults
            universe_size = getattr(config.data, 'universe_size', 10) if hasattr(config, 'data') else 10
            
            # Signal parameters
            adx_period = getattr(config.signals, 'adx_period', 14) if hasattr(config, 'signals') else 14
            adx_threshold = getattr(config.signals, 'adx_threshold', 25.0) if hasattr(config, 'signals') else 25.0
            ewma_fast = getattr(config.signals, 'ewma_fast', 20) if hasattr(config, 'signals') else 20
            ewma_slow = getattr(config.signals, 'ewma_slow', 50) if hasattr(config, 'signals') else 50
            volume_filter = getattr(config.signals, 'volume_filter_multiple', 1.5) if hasattr(config, 'signals') else 1.5
            
            # Portfolio parameters
            max_position_size = getattr(config.portfolio, 'max_position_size', 0.10) if hasattr(config, 'portfolio') else 0.10
            rebalance_freq = getattr(config.portfolio, 'rebalance_frequency', 'weekly') if hasattr(config, 'portfolio') else 'weekly'
            
            # Risk parameters
            max_drawdown = getattr(config.risk, 'max_drawdown', 0.25) if hasattr(config, 'risk') else 0.25
            correlation_threshold = getattr(config.risk, 'correlation_threshold', 0.80) if hasattr(config, 'risk') else 0.80
            volatility_multiple = getattr(config.risk, 'volatility_multiple', 2.0) if hasattr(config, 'risk') else 2.0
            
            # Cost parameters
            maker_fee = getattr(config.costs, 'maker_fee', 0.001) if hasattr(config, 'costs') else 0.001
            taker_fee = getattr(config.costs, 'taker_fee', 0.001) if hasattr(config, 'costs') else 0.001
            base_spread = getattr(config.costs, 'base_spread', 0.0005) if hasattr(config, 'costs') else 0.0005
            funding_days = getattr(config.costs, 'funding_lookback_days', 30) if hasattr(config, 'costs') else 30
            
            # Strategy parameters (for backward compatibility)
            long_short = getattr(config.strategy, 'long_short', True) if hasattr(config, 'strategy') else True
            
        except AttributeError as e:
            self.logger.warning(f"Config attribute error: {e}, using defaults")
            # Set all defaults
            universe_size = 10
            adx_period = 14
            adx_threshold = 25.0
            ewma_fast = 20
            ewma_slow = 50
            volume_filter = 1.5
            max_position_size = 0.10
            rebalance_freq = 'weekly'
            max_drawdown = 0.25
            correlation_threshold = 0.80
            volatility_multiple = 2.0
            maker_fee = 0.001
            taker_fee = 0.001
            base_spread = 0.0005
            funding_days = 30
            long_short = True
        
        # Initialize components
        self.storage = JsonStorage(data_dir)
        self.universe_manager = UniverseManager(
            data_dir,
            universe_size=universe_size,
            logger=self.logger
        )
        
        self.signal_generator = SignalGenerator(
            adx_period=adx_period,
            adx_threshold=adx_threshold,
            ewma_fast=ewma_fast,
            ewma_slow=ewma_slow,
            volume_filter_multiple=volume_filter,
            logger=self.logger
        )
        
        self.optimizer = ERCOptimizer(
            max_position_size=max_position_size,
            allow_short=long_short,
            logger=self.logger
        )
        
        self.position_sizer = PositionSizer(
            atr_period=adx_period,  # Use ADX period as ATR period
            atr_multiplier=2.0,  # Default for crypto
            max_position_size=max_position_size,
            logger=self.logger
        )
        
        self.rebalancer = Rebalancer(
            rebalance_frequency=rebalance_freq,
            logger=self.logger
        )
        
        self.risk_manager = RiskManager(
            max_drawdown=max_drawdown,
            max_position_size=max_position_size,
            max_correlation=correlation_threshold,
            max_exchange_exposure=0.40,  # Default
            volatility_scale_threshold=volatility_multiple,
            logger=self.logger
        )
        
        self.cost_model = CostModel(
            maker_fee=maker_fee,
            taker_fee=taker_fee,
            base_spread=base_spread,
            logger=self.logger
        )
        
        self.funding_rates = FundingRates(
            lookback_days=funding_days,
            logger=self.logger
        )
'''
    
    # Find where the old initialization code starts
    # Look for "self.config = config"
    old_init_start = content.find("self.config = config", insert_pos)
    if old_init_start != -1:
        # Find where the initialization ends (look for the first method after __init__)
        next_method = content.find("\n    def ", old_init_start)
        if next_method != -1:
            # Replace the old initialization
            content = content[:insert_pos] + init_code + content[next_method:]
        else:
            # No next method found, replace to end
            content = content[:insert_pos] + init_code
    else:
        # Insert new initialization
        content = content[:insert_pos] + init_code + content[insert_pos:]
    
    # Write the fixed content
    with open(engine_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("[OK] Fixed BacktestEngine syntax and initialization")
    return True


def main():
    print("Fixing BacktestEngine syntax error...")
    print("=" * 60)
    
    success = fix_engine_syntax()
    
    if success:
        print("\n[OK] Fix applied successfully!")
        print("\nNext steps:")
        print("1. Test: python test_engine_init.py")
        print("2. Run backtest: python run.py --no-validate")
    else:
        print("\n[ERROR] Fix failed")
        print("You may need to manually restore from backup:")
        print("  copy crypto_momentum_backtest/backtest/engine.py.bak crypto_momentum_backtest/backtest/engine.py")


if __name__ == "__main__":
    main()