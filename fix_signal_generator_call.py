#!/usr/bin/env python3
"""
Fix the SignalGenerator method call in BacktestEngine.
"""

from pathlib import Path
import re


def fix_signal_generator_call():
    """Fix the generate_signals call to match the actual method signature."""
    
    engine_file = Path("crypto_momentum_backtest/backtest/engine.py")
    
    if not engine_file.exists():
        print("[ERROR] engine.py not found")
        return False
    
    # Read the file
    with open(engine_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and fix the generate_signals call
    # The new SignalGenerator expects (data, signal_type, symbol)
    
    # First, let's fix the generate_signals method in BacktestEngine
    old_method = r'''def generate_signals\(
        self,
        data: Dict\[str, pd\.DataFrame\]
    \) -> Dict\[str, pd\.DataFrame\]:
        """
        Generate trading signals for all symbols\.
        
        Args:
            data: Market data by symbol
            
        Returns:
            Signals by symbol
        """
        signals = \{\}
        
        for symbol, df in data\.items\(\):
            self\.logger\.info\(f"Generating signals for \{symbol\}"\)
            
            signal_df = self\.signal_generator\.generate_signals\(
                df,
                apply_filters=True,
                calculate_strength=True
            \)
            
            signals\[symbol\] = signal_df
        
        return signals'''
    
    new_method = '''def generate_signals(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate trading signals for all symbols.
        
        Args:
            data: Market data by symbol
            
        Returns:
            Signals by symbol
        """
        signals = {}
        
        for symbol, df in data.items():
            self.logger.info(f"Generating signals for {symbol}")
            
            # Generate signals using the configured strategy
            signal_series = self.signal_generator.generate_signals(
                df,
                signal_type=self.config.signals.signal_strategy,
                symbol=symbol
            )
            
            # Convert to DataFrame format expected by engine
            signal_df = pd.DataFrame(index=df.index)
            signal_df['position'] = signal_series
            signal_df['long_signal'] = signal_series == 1
            signal_df['short_signal'] = signal_series == -1
            
            signals[symbol] = signal_df
        
        return signals'''
    
    # Try to replace the method
    if re.search(old_method, content, re.DOTALL):
        content = re.sub(old_method, new_method, content, flags=re.DOTALL)
        print("[OK] Fixed generate_signals method using regex")
    else:
        # Try a simpler replacement
        # Find the problematic call
        old_call = '''signal_df = self.signal_generator.generate_signals(
                df,
                apply_filters=True,
                calculate_strength=True
            )'''
        
        new_call = '''# Generate signals using the configured strategy
            signal_series = self.signal_generator.generate_signals(
                df,
                signal_type=self.config.signals.signal_strategy,
                symbol=symbol
            )
            
            # Convert to DataFrame format expected by engine
            signal_df = pd.DataFrame(index=df.index)
            signal_df['position'] = signal_series
            signal_df['long_signal'] = signal_series == 1
            signal_df['short_signal'] = signal_series == -1'''
        
        if old_call in content:
            content = content.replace(old_call, new_call)
            print("[OK] Fixed generate_signals call")
        else:
            print("[WARNING] Could not find exact match, trying alternative fix")
            
            # Alternative: just remove the extra parameters
            content = re.sub(
                r'self\.signal_generator\.generate_signals\(\s*df,\s*apply_filters=True,\s*calculate_strength=True\s*\)',
                'self.signal_generator.generate_signals(df)',
                content
            )
    
    # Make sure pandas is imported
    if "import pandas as pd" not in content:
        # Add import at the top
        import_pos = content.find("import")
        if import_pos != -1:
            content = content[:import_pos] + "import pandas as pd\n" + content[import_pos:]
    
    # Write the fixed content
    with open(engine_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("[OK] Fixed signal generator calls in engine.py")
    return True


def create_test_signal_generator():
    """Create a test to verify signal generation works."""
    
    test_code = '''#!/usr/bin/env python3
"""Test signal generation."""

from pathlib import Path
import pandas as pd
from crypto_momentum_backtest.signals.signal_generator import SignalGenerator
from crypto_momentum_backtest.data.json_storage import JsonStorage

# Load some test data
storage = JsonStorage(Path('data'))
df = storage.load_range(
    symbol='BTCUSDT',
    start_date=pd.Timestamp('2022-01-01'),
    end_date=pd.Timestamp('2022-01-31')
)

if not df.empty:
    print(f"Loaded {len(df)} days of data")
    
    # Create signal generator
    generator = SignalGenerator()
    
    # Test signal generation
    signals = generator.generate_signals(df, signal_type='momentum', symbol='BTCUSDT')
    
    print(f"Generated signals: {type(signals)}")
    print(f"Signal values: {signals.value_counts()}")
    print(f"Total signals: {(signals != 0).sum()}")
else:
    print("No data found")
'''
    
    with open('test_signal_generation.py', 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    print("[OK] Created test_signal_generation.py")


def create_simple_signal_adapter():
    """Create an adapter to handle the interface mismatch."""
    
    adapter_code = '''# Add this to the BacktestEngine generate_signals method as a simpler fix:

def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Generate trading signals for all symbols."""
    signals = {}
    
    for symbol, df in data.items():
        self.logger.info(f"Generating signals for {symbol}")
        
        try:
            # Try the new interface first
            signal_series = self.signal_generator.generate_signals(
                df,
                signal_type=getattr(self.config.signals, 'signal_strategy', 'momentum'),
                symbol=symbol
            )
            
            # Convert to expected format
            signal_df = pd.DataFrame(index=df.index)
            signal_df['position'] = signal_series
            signal_df['long_signal'] = signal_series == 1
            signal_df['short_signal'] = signal_series == -1
            
        except TypeError:
            # Fall back to simple call
            self.logger.warning("Using fallback signal generation")
            
            # Just call with the dataframe
            result = self.signal_generator.generate_signals(df)
            
            if isinstance(result, pd.Series):
                # Convert Series to DataFrame
                signal_df = pd.DataFrame(index=df.index)
                signal_df['position'] = result
                signal_df['long_signal'] = result == 1
                signal_df['short_signal'] = result == -1
            else:
                # Assume it's already a DataFrame
                signal_df = result
        
        signals[symbol] = signal_df
    
    return signals
'''
    
    with open('signal_adapter_code.txt', 'w', encoding='utf-8') as f:
        f.write(adapter_code)
    
    print("[OK] Created signal_adapter_code.txt with adapter implementation")


def main():
    print("Fixing SignalGenerator Interface Mismatch")
    print("=" * 60)
    
    # Apply the fix
    success = fix_signal_generator_call()
    
    if success:
        # Create test script
        create_test_signal_generator()
        
        # Create adapter code
        create_simple_signal_adapter()
        
        print("\n[OK] Fixes applied!")
        print("\nNext steps:")
        print("1. Test signal generation: python test_signal_generation.py")
        print("2. Run backtest: python run.py --no-validate")
        print("\nIf still having issues, manually apply the adapter code from signal_adapter_code.txt")
    else:
        print("\n[ERROR] Could not apply automatic fix")
        print("Please manually edit engine.py and update the generate_signals method")
        print("See signal_adapter_code.txt for the implementation")


if __name__ == "__main__":
    main()