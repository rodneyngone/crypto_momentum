#!/usr/bin/env python3
"""
Debug script for crypto momentum signal generation issues.
This script will help identify why mean return strategies generate 0 signals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

class SignalDebugger:
    """Debug signal generation issues"""
    
    def __init__(self, data_path="data", config_path="config.yaml"):
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        
    def load_config(self):
        """Load configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            return self.get_default_config()
    
    def get_default_config(self):
        """Default configuration for testing"""
        return {
            'signals': {
                'adx_period': 14,
                'ewm_span': 20,
                'threshold': 0.02,
                'mean_return_window': 5
            }
        }
    
    def load_btc_data(self):
        """Load BTC data for debugging"""
        # Try different possible file locations/names
        possible_files = [
            self.data_path / "BTCUSDT.csv",
            self.data_path / "btc_data.csv", 
            self.data_path / "btcusdt.csv",
            "BTCUSDT.csv",
            "btc_data.csv"
        ]
        
        for file_path in possible_files:
            if Path(file_path).exists():
                print(f"Loading data from: {file_path}")
                return pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # If no file found, create sample data
        print("No BTC data file found. Creating sample data for testing...")
        return self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample BTC data for testing"""
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        
        # Simulate realistic BTC price movements
        initial_price = 40000
        returns = np.random.normal(0.001, 0.04, len(dates))  # ~4% daily volatility
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000000, 10000000, len(dates))
        }, index=dates)
        
        print("Sample data created with realistic crypto volatility")
        return data
    
    def calculate_returns(self, data):
        """Calculate various return metrics"""
        data = data.copy()
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Remove any infinite or NaN values
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        
        return data
    
    def debug_original_momentum(self, data, config):
        """Debug the original momentum strategy that works"""
        print("\n" + "="*60)
        print("🔍 DEBUGGING ORIGINAL MOMENTUM STRATEGY")
        print("="*60)
        
        # Simple momentum: price vs moving average
        ewm_span = config['signals']['ewm_span']
        threshold = config['signals']['threshold']
        
        data['ewma'] = data['close'].ewm(span=ewm_span).mean()
        data['momentum'] = (data['close'] / data['ewma'] - 1)
        
        print(f"EWM Span: {ewm_span}")
        print(f"Threshold: {threshold}")
        print(f"Momentum stats:")
        print(data['momentum'].describe())
        
        # Generate signals
        long_signals = data['momentum'] > threshold
        short_signals = data['momentum'] < -threshold
        
        print(f"\nSignal Results:")
        print(f"Long signals: {long_signals.sum()}")
        print(f"Short signals: {short_signals.sum()}")
        print(f"Total trading days: {len(data)}")
        print(f"Signal frequency: {(long_signals.sum() + short_signals.sum()) / len(data) * 100:.2f}%")
        
        return long_signals, short_signals
    
    def debug_mean_return_ewm(self, data, config):
        """Debug EWM mean return strategy"""
        print("\n" + "="*60)
        print("🔍 DEBUGGING MEAN RETURN EWM STRATEGY")
        print("="*60)
        
        ewm_span = config['signals']['ewm_span']
        threshold = config['signals']['threshold']
        
        # Calculate EWM of returns (not prices)
        ewm_returns = data['returns'].ewm(span=ewm_span).mean()
        
        print(f"EWM Span: {ewm_span}")
        print(f"Threshold: {threshold}")
        print(f"Returns stats:")
        print(data['returns'].describe())
        print(f"\nEWM Returns stats:")
        print(ewm_returns.describe())
        
        # Check if any values exceed threshold
        above_threshold = (ewm_returns > threshold).sum()
        below_threshold = (ewm_returns < -threshold).sum()
        
        print(f"\nThreshold Analysis:")
        print(f"EWM returns > {threshold}: {above_threshold}")
        print(f"EWM returns < {-threshold}: {below_threshold}")
        print(f"Max EWM return: {ewm_returns.max():.6f}")
        print(f"Min EWM return: {ewm_returns.min():.6f}")
        
        # Generate signals
        long_signals = ewm_returns > threshold
        short_signals = ewm_returns < -threshold
        
        print(f"\nSignal Results:")
        print(f"Long signals: {long_signals.sum()}")
        print(f"Short signals: {short_signals.sum()}")
        
        # Test with different thresholds
        print(f"\n📊 Threshold Sensitivity Analysis:")
        test_thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        for test_thresh in test_thresholds:
            long_test = (ewm_returns > test_thresh).sum()
            short_test = (ewm_returns < -test_thresh).sum()
            print(f"Threshold {test_thresh:5.3f}: Long={long_test:3d}, Short={short_test:3d}, Total={long_test + short_test:3d}")
        
        return long_signals, short_signals
    
    def debug_mean_return_simple(self, data, config):
        """Debug simple mean return strategy"""
        print("\n" + "="*60)
        print("🔍 DEBUGGING SIMPLE MEAN RETURN STRATEGY")
        print("="*60)
        
        window = config['signals']['mean_return_window']
        threshold = config['signals']['threshold']
        
        # Calculate rolling mean of returns
        simple_mean_returns = data['returns'].rolling(window=window).mean()
        
        print(f"Rolling Window: {window}")
        print(f"Threshold: {threshold}")
        print(f"Simple Mean Returns stats:")
        print(simple_mean_returns.describe())
        
        # Check threshold analysis
        above_threshold = (simple_mean_returns > threshold).sum()
        below_threshold = (simple_mean_returns < -threshold).sum()
        
        print(f"\nThreshold Analysis:")
        print(f"Mean returns > {threshold}: {above_threshold}")
        print(f"Mean returns < {-threshold}: {below_threshold}")
        print(f"Max mean return: {simple_mean_returns.max():.6f}")
        print(f"Min mean return: {simple_mean_returns.min():.6f}")
        
        # Generate signals
        long_signals = simple_mean_returns > threshold
        short_signals = simple_mean_returns < -threshold
        
        print(f"\nSignal Results:")
        print(f"Long signals: {long_signals.sum()}")
        print(f"Short signals: {short_signals.sum()}")
        
        return long_signals, short_signals
    
    def suggest_optimal_thresholds(self, data, config):
        """Suggest optimal thresholds based on data analysis"""
        print("\n" + "="*60)
        print("💡 OPTIMAL THRESHOLD SUGGESTIONS")
        print("="*60)
        
        ewm_span = config['signals']['ewm_span']
        window = config['signals']['mean_return_window']
        
        # Calculate both mean return types
        ewm_returns = data['returns'].ewm(span=ewm_span).mean()
        simple_returns = data['returns'].rolling(window=window).mean()
        
        # Calculate percentile-based thresholds
        percentiles = [70, 75, 80, 85, 90, 95]
        
        print("📊 Percentile-Based Threshold Suggestions:")
        print("(These will generate signals on X% of trading days)")
        print()
        
        for pct in percentiles:
            ewm_thresh = np.percentile(np.abs(ewm_returns.dropna()), pct)
            simple_thresh = np.percentile(np.abs(simple_returns.dropna()), pct)
            
            # Test signal counts
            ewm_signals = ((ewm_returns > ewm_thresh) | (ewm_returns < -ewm_thresh)).sum()
            simple_signals = ((simple_returns > simple_thresh) | (simple_returns < -simple_thresh)).sum()
            
            print(f"{pct}th percentile:")
            print(f"  EWM threshold: {ewm_thresh:.5f} -> {ewm_signals} signals ({ewm_signals/len(data)*100:.1f}% of days)")
            print(f"  Simple threshold: {simple_thresh:.5f} -> {simple_signals} signals ({simple_signals/len(data)*100:.1f}% of days)")
            print()
        
        # Volatility-based suggestions
        return_volatility = data['returns'].std()
        
        print("📈 Volatility-Based Suggestions:")
        vol_multiples = [0.5, 1.0, 1.5, 2.0, 2.5]
        for mult in vol_multiples:
            vol_thresh = return_volatility * mult
            signals = ((data['returns'] > vol_thresh) | (data['returns'] < -vol_thresh)).sum()
            print(f"  {mult}x volatility ({vol_thresh:.5f}): {signals} signals ({signals/len(data)*100:.1f}% of days)")
    
    def create_fixed_config(self, config):
        """Create a fixed configuration with optimal thresholds"""
        print("\n" + "="*60)
        print("🔧 CREATING FIXED CONFIGURATION")
        print("="*60)
        
        # Use more realistic thresholds for crypto
        fixed_config = config.copy()
        fixed_config['signals']['threshold'] = 0.005  # 0.5% instead of 2%
        fixed_config['signals']['ewm_span'] = 10      # Shorter EWM for crypto
        fixed_config['signals']['mean_return_window'] = 3  # Shorter window
        
        print("Original configuration:")
        print(f"  Threshold: {config['signals']['threshold']}")
        print(f"  EWM Span: {config['signals']['ewm_span']}")
        print(f"  Window: {config['signals']['mean_return_window']}")
        
        print("\nFixed configuration:")
        print(f"  Threshold: {fixed_config['signals']['threshold']}")
        print(f"  EWM Span: {fixed_config['signals']['ewm_span']}")
        print(f"  Window: {fixed_config['signals']['mean_return_window']}")
        
        return fixed_config
    
    def test_fixed_strategies(self, data, fixed_config):
        """Test strategies with fixed configuration"""
        print("\n" + "="*60)
        print("✅ TESTING FIXED STRATEGIES")
        print("="*60)
        
        # Test EWM strategy with new config
        ewm_span = fixed_config['signals']['ewm_span']
        threshold = fixed_config['signals']['threshold']
        
        ewm_returns = data['returns'].ewm(span=ewm_span).mean()
        long_signals = ewm_returns > threshold
        short_signals = ewm_returns < -threshold
        
        print(f"Fixed EWM Strategy:")
        print(f"  Long signals: {long_signals.sum()}")
        print(f"  Short signals: {short_signals.sum()}")
        print(f"  Total signals: {long_signals.sum() + short_signals.sum()}")
        
        # Test simple strategy with new config
        window = fixed_config['signals']['mean_return_window']
        simple_returns = data['returns'].rolling(window=window).mean()
        long_simple = simple_returns > threshold
        short_simple = simple_returns < -threshold
        
        print(f"\nFixed Simple Strategy:")
        print(f"  Long signals: {long_simple.sum()}")
        print(f"  Short signals: {short_simple.sum()}")
        print(f"  Total signals: {long_simple.sum() + short_simple.sum()}")
        
        return long_signals, short_signals, long_simple, short_simple
    
    def save_fixed_config(self, fixed_config, filename="config_fixed.yaml"):
        """Save the fixed configuration to file"""
        with open(filename, 'w') as f:
            yaml.dump(fixed_config, f, default_flow_style=False)
        print(f"\n💾 Fixed configuration saved to: {filename}")
        print("You can now use this config with:")
        print(f"  copy {filename} config.yaml")
        print("  python run.py --no-validate")
    
    def run_full_debug(self):
        """Run complete debugging process"""
        print("🚀 STARTING SIGNAL DEBUG PROCESS")
        print("="*60)
        
        # Load configuration and data
        config = self.load_config()
        data = self.load_btc_data()
        data = self.calculate_returns(data)
        
        print(f"Loaded data: {len(data)} rows")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        # Debug each strategy
        self.debug_original_momentum(data, config)
        self.debug_mean_return_ewm(data, config)
        self.debug_mean_return_simple(data, config)
        
        # Suggest optimal parameters
        self.suggest_optimal_thresholds(data, config)
        
        # Create and test fixed configuration
        fixed_config = self.create_fixed_config(config)
        self.test_fixed_strategies(data, fixed_config)
        
        # Save fixed config
        self.save_fixed_config(fixed_config)
        
        print("\n" + "="*60)
        print("🎯 DEBUG COMPLETE!")
        print("="*60)
        print("Next steps:")
        print("1. Use the fixed config file generated")
        print("2. Run your backtest again")
        print("3. Adjust thresholds based on suggestions above")

def main():
    """Main debugging function"""
    debugger = SignalDebugger()
    debugger.run_full_debug()

if __name__ == "__main__":
    main()