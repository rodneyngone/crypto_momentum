#!/usr/bin/env python3
"""
Debug mean return signals using the EXACT same data loading as test_mean_return_signals.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add parent directory to path (same as your test script)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_momentum_backtest.signals.signal_generator import SignalGenerator
from crypto_momentum_backtest.data.json_storage import JsonStorage


def load_test_data(symbol='BTCUSDT', start_date='2022-01-01', end_date='2023-12-31'):
    """Load test data - EXACT same function as your test script"""
    storage = JsonStorage(Path('data'))
    
    df = storage.load_range(
        symbol=symbol,
        start_date=pd.Timestamp(start_date),
        end_date=pd.Timestamp(end_date)
    )
    
    return df

def debug_signal_generator():
    """Debug the SignalGenerator to understand threshold issues"""
    
    print("🔍 DEBUGGING SIGNAL GENERATOR WITH REAL DATA")
    print("="*60)
    
    # Load the same data as test script
    try:
        print("📊 Loading BTC data...")
        data = load_test_data()
        print(f"Loaded {len(data)} days of data")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        print(f"Columns: {list(data.columns)}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Make sure you have data in the 'data' directory")
        return None
    
    return data

def analyze_returns_distribution(data):
    """Analyze the returns to understand what thresholds make sense"""
    
    print("\n📈 ANALYZING RETURNS DISTRIBUTION")
    print("="*50)
    
    # Calculate returns (same as SignalGenerator would)
    data['returns'] = data['close'].pct_change()
    returns = data['returns'].dropna()
    
    print(f"Returns statistics:")
    print(f"  Count: {len(returns):,}")
    print(f"  Mean: {returns.mean():.6f} ({returns.mean()*100:.4f}%)")
    print(f"  Std: {returns.std():.6f} ({returns.std()*100:.4f}%)")
    print(f"  Min: {returns.min():.6f} ({returns.min()*100:.2f}%)")
    print(f"  Max: {returns.max():.6f} ({returns.max()*100:.2f}%)")
    
    # Show percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    print(f"\nReturn percentiles:")
    for p in percentiles:
        value = np.percentile(returns, p)
        print(f"  {p:2d}%: {value:.6f} ({value*100:.3f}%)")
    
    return returns

def test_mean_return_signals(data):
    """Test mean return signal generation with different parameters"""
    
    print("\n🔍 TESTING MEAN RETURN SIGNAL GENERATION")
    print("="*50)
    
    returns = data['returns'].dropna()
    
    # Test different EWM parameters
    ewm_spans = [5, 10, 15, 20, 30]
    thresholds = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02]
    
    print("EWM Mean Return Signal Testing:")
    print(f"{'Span':<6} {'Thresh':<8} {'Long':<6} {'Short':<6} {'Total':<6} {'Freq%':<7}")
    print("-" * 45)
    
    best_combinations = []
    
    for span in ewm_spans:
        # Calculate EWM of returns
        ewm_returns = returns.ewm(span=span).mean()
        
        for threshold in thresholds:
            long_signals = (ewm_returns > threshold).sum()
            short_signals = (ewm_returns < -threshold).sum()
            total_signals = long_signals + short_signals
            frequency = (total_signals / len(returns)) * 100
            
            print(f"{span:<6} {threshold:<8.4f} {long_signals:<6} {short_signals:<6} {total_signals:<6} {frequency:<7.1f}")
            
            # Save good combinations (10-50 signals is reasonable)
            if 10 <= total_signals <= 50:
                best_combinations.append({
                    'type': 'EWM',
                    'span': span,
                    'threshold': threshold,
                    'signals': total_signals,
                    'frequency': frequency
                })
    
    # Test rolling mean parameters
    print(f"\n\nSimple Rolling Mean Signal Testing:")
    print(f"{'Window':<8} {'Thresh':<8} {'Long':<6} {'Short':<6} {'Total':<6} {'Freq%':<7}")
    print("-" * 47)
    
    windows = [3, 5, 7, 10, 15, 20]
    
    for window in windows:
        # Calculate rolling mean of returns
        rolling_mean = returns.rolling(window=window).mean()
        
        for threshold in thresholds:
            long_signals = (rolling_mean > threshold).sum()
            short_signals = (rolling_mean < -threshold).sum()
            total_signals = long_signals + short_signals
            frequency = (total_signals / len(returns)) * 100
            
            print(f"{window:<8} {threshold:<8.4f} {long_signals:<6} {short_signals:<6} {total_signals:<6} {frequency:<7.1f}")
            
            # Save good combinations
            if 10 <= total_signals <= 50:
                best_combinations.append({
                    'type': 'Rolling',
                    'window': window,
                    'threshold': threshold,
                    'signals': total_signals,
                    'frequency': frequency
                })
    
    return best_combinations

def suggest_signal_generator_fixes(best_combinations):
    """Suggest specific fixes for the SignalGenerator"""
    
    print("\n💡 SIGNAL GENERATOR FIX RECOMMENDATIONS")
    print("="*50)
    
    if not best_combinations:
        print("❌ No good combinations found in tested ranges.")
        print("Try even smaller thresholds: 0.0001, 0.0002, 0.0003")
        return
    
    print("✅ WORKING PARAMETER COMBINATIONS:")
    print()
    
    # Group by type
    ewm_combos = [c for c in best_combinations if c['type'] == 'EWM']
    rolling_combos = [c for c in best_combinations if c['type'] == 'Rolling']
    
    if ewm_combos:
        print("🔧 EWM Mean Return Strategy:")
        best_ewm = max(ewm_combos, key=lambda x: x['signals'])
        print(f"   RECOMMENDED: span={best_ewm['span']}, threshold={best_ewm['threshold']:.4f}")
        print(f"   This generates: {best_ewm['signals']} signals ({best_ewm['frequency']:.1f}% of days)")
        print()
        
        print("   Top 5 EWM combinations:")
        for combo in sorted(ewm_combos, key=lambda x: x['signals'], reverse=True)[:5]:
            print(f"     span={combo['span']:2d}, threshold={combo['threshold']:.4f} -> {combo['signals']:2d} signals")
        print()
    
    if rolling_combos:
        print("🔧 Simple Rolling Mean Strategy:")
        best_rolling = max(rolling_combos, key=lambda x: x['signals'])
        print(f"   RECOMMENDED: window={best_rolling['window']}, threshold={best_rolling['threshold']:.4f}")
        print(f"   This generates: {best_rolling['signals']} signals ({best_rolling['frequency']:.1f}% of days)")
        print()
        
        print("   Top 5 Rolling combinations:")
        for combo in sorted(rolling_combos, key=lambda x: x['signals'], reverse=True)[:5]:
            print(f"     window={combo['window']:2d}, threshold={combo['threshold']:.4f} -> {combo['signals']:2d} signals")

def test_current_signal_generator(data):
    """Test what the current SignalGenerator actually produces"""
    
    print("\n🧪 TESTING CURRENT SIGNAL GENERATOR")
    print("="*50)
    
    try:
        # Create SignalGenerator with default parameters
        signal_gen = SignalGenerator()
        
        print("Testing current SignalGenerator with different strategy types...")
        
        strategies = ['momentum', 'mean_return_ewm', 'mean_return_simple']
        
        for strategy in strategies:
            try:
                print(f"\nTesting {strategy}:")
                
                # This might require specific parameters - adjust as needed
                if hasattr(signal_gen, 'generate'):
                    signals = signal_gen.generate(data, strategy_type=strategy)
                    if hasattr(signals, 'sum'):
                        print(f"  Signals generated: {signals.sum()}")
                    else:
                        print(f"  Signals: {signals}")
                else:
                    print("  SignalGenerator.generate method not found")
                    
            except Exception as e:
                print(f"  Error testing {strategy}: {e}")
                
    except Exception as e:
        print(f"❌ Error creating SignalGenerator: {e}")
        print("This might help us understand what parameters it expects")

def create_fixed_config():
    """Create a config with the right threshold values"""
    
    print("\n🔧 CREATING FIXED CONFIG RECOMMENDATIONS")
    print("="*50)
    
    fixed_config = {
        'signals': {
            # Momentum strategy (currently working)
            'momentum_threshold': 0.01,  # Keep working value
            
            # Mean return strategies (FIXED thresholds)
            'mean_return_ewm_threshold': 0.002,  # 0.2% instead of 2%
            'mean_return_ewm_span': 10,
            
            'mean_return_simple_threshold': 0.003,  # 0.3% instead of 2%
            'mean_return_simple_window': 5,
            
            # General parameters
            'adx_period': 14,
            'lookback': 20
        }
    }
    
    print("Recommended config.yaml changes:")
    print("```yaml")
    print("signals:")
    for key, value in fixed_config['signals'].items():
        print(f"  {key}: {value}")
    print("```")
    
    return fixed_config

def main():
    """Main debugging function"""
    print("🔍 DEBUGGING MEAN RETURN SIGNALS WITH REAL DATA")
    print("="*70)
    
    # Step 1: Load real data using same method as test script
    data = debug_signal_generator()
    if data is None:
        return
    
    # Step 2: Analyze returns distribution
    returns = analyze_returns_distribution(data)
    
    # Step 3: Test different mean return parameters
    best_combinations = test_mean_return_signals(data)
    
    # Step 4: Suggest fixes
    suggest_signal_generator_fixes(best_combinations)
    
    # Step 5: Test current SignalGenerator
    test_current_signal_generator(data)
    
    # Step 6: Create config recommendations
    create_fixed_config()
    
    print("\n🎯 SUMMARY")
    print("="*30)
    print("The issue is that mean return thresholds are too high for crypto data.")
    print("Crypto mean returns are typically 0.001-0.005, not 0.02 like stocks.")
    print("\nUpdate your SignalGenerator or config with the recommended thresholds above.")

if __name__ == "__main__":
    main()