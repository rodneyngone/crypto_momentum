#!/usr/bin/env python3
"""
Test and compare different mean return signal strategies.
FIXED VERSION: Compatible with new SignalGenerator class.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_momentum_backtest.signals.signal_generator import SignalGenerator, SignalType
from crypto_momentum_backtest.data.json_storage import JsonStorage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_data(symbol='BTCUSDT', start_date='2022-01-01', end_date='2023-12-31'):
    """Load test data for signal comparison."""
    storage = JsonStorage(Path('data'))
    
    df = storage.load_range(
        symbol=symbol,
        start_date=pd.Timestamp(start_date),
        end_date=pd.Timestamp(end_date)
    )
    
    return df


def test_original_momentum_strategy(data, signal_gen):
    """Test the original momentum strategy (baseline)."""
    try:
        signals = signal_gen.generate_signals(data, SignalType.MOMENTUM)
        returns = data['close'].pct_change().fillna(0)
        
        # Calculate strategy returns
        strategy_returns = signals.shift(1) * returns
        strategy_returns = strategy_returns.dropna()
        
        # Calculate performance metrics
        if len(strategy_returns) > 0 and strategy_returns.std() > 0:
            total_return = (1 + strategy_returns).prod() - 1
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            win_rate = (strategy_returns > 0).mean()
            
            long_signals = (signals == 1).sum()
            short_signals = (signals == -1).sum()
        else:
            total_return = 0
            sharpe_ratio = 0
            win_rate = 0
            long_signals = 0
            short_signals = 0
        
        return {
            'strategy': 'Original Momentum',
            'total_return': total_return,
            'sharpe': sharpe_ratio,
            'win_rate': win_rate,
            'signals': {'long': long_signals, 'short': short_signals}
        }
        
    except Exception as e:
        logger.error(f"Error testing original momentum strategy: {e}")
        return {
            'strategy': 'Original Momentum',
            'total_return': 0,
            'sharpe': 0,
            'win_rate': 0,
            'signals': {'long': 0, 'short': 0}
        }


def test_mean_return_strategy(data, signal_gen, strategy_type, strategy_name):
    """Test a mean return strategy."""
    try:
        signals = signal_gen.generate_signals(data, strategy_type)
        returns = data['close'].pct_change().fillna(0)
        
        # Calculate strategy returns
        strategy_returns = signals.shift(1) * returns
        strategy_returns = strategy_returns.dropna()
        
        # Calculate performance metrics
        if len(strategy_returns) > 0 and strategy_returns.std() > 0:
            total_return = (1 + strategy_returns).prod() - 1
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            win_rate = (strategy_returns > 0).mean()
            
            long_signals = (signals == 1).sum()
            short_signals = (signals == -1).sum()
        else:
            total_return = 0
            sharpe_ratio = 0
            win_rate = 0
            long_signals = 0
            short_signals = 0
        
        return {
            'strategy': strategy_name,
            'total_return': total_return,
            'sharpe': sharpe_ratio,
            'win_rate': win_rate,
            'signals': {'long': long_signals, 'short': short_signals}
        }
        
    except Exception as e:
        logger.error(f"Error testing {strategy_name}: {e}")
        return {
            'strategy': strategy_name,
            'total_return': 0,
            'sharpe': 0,
            'win_rate': 0,
            'signals': {'long': 0, 'short': 0}
        }


def test_signal_strategies(data):
    """Test all signal strategies and return results."""
    
    # Create SignalGenerator with crypto-optimized defaults
    signal_gen = SignalGenerator(
        # Crypto-optimized parameters based on empirical analysis
        momentum_threshold=0.01,
        mean_return_ewm_threshold=0.015,    # Fixed: was likely 0.02
        mean_return_ewm_span=10,            # Fixed: more responsive for crypto
        mean_return_simple_threshold=0.015, # Fixed: was likely 0.02
        mean_return_simple_window=10        # Fixed: balanced for crypto
    )
    
    results = []
    
    # Test Original Momentum Strategy
    print("📊 Testing Original Momentum Strategy...")
    momentum_result = test_original_momentum_strategy(data, signal_gen)
    results.append(momentum_result)
    
    # Test Mean Return EWM Strategy
    print("📊 Testing Mean Return EWM Strategy...")
    ewm_result = test_mean_return_strategy(
        data, signal_gen, SignalType.MEAN_RETURN_EWM, "Mean Return EWM"
    )
    results.append(ewm_result)
    
    # Test Mean Return Simple Strategy
    print("📊 Testing Mean Return Simple Strategy...")
    simple_result = test_mean_return_strategy(
        data, signal_gen, SignalType.MEAN_RETURN_SIMPLE, "Mean Return Simple"
    )
    results.append(simple_result)
    
    # Test Hybrid AND Strategy
    print("📊 Testing Hybrid AND Strategy...")
    hybrid_and_result = test_mean_return_strategy(
        data, signal_gen, SignalType.HYBRID_AND, "Hybrid AND"
    )
    results.append(hybrid_and_result)
    
    # Test Hybrid OR Strategy
    print("📊 Testing Hybrid OR Strategy...")
    hybrid_or_result = test_mean_return_strategy(
        data, signal_gen, SignalType.HYBRID_OR, "Hybrid OR"
    )
    results.append(hybrid_or_result)
    
    return results


def calculate_buy_hold_return(data):
    """Calculate buy and hold return for comparison."""
    try:
        start_price = data['close'].iloc[0]
        end_price = data['close'].iloc[-1]
        buy_hold_return = (end_price / start_price) - 1
        return buy_hold_return
    except:
        return 0


def print_results(results, buy_hold_return):
    """Print formatted results."""
    
    print("\n" + "="*60)
    print("📋 DETAILED RESULTS")
    print("="*60)
    
    for result in results:
        strategy = result['strategy']
        total_return = result['total_return']
        sharpe = result['sharpe']
        win_rate = result['win_rate']
        long_signals = result['signals']['long']
        short_signals = result['signals']['short']
        
        print(f"\n📊 {strategy}:")
        
        # Handle NaN values gracefully
        if np.isnan(total_return) or total_return == 0:
            print(f"  Total Return: 0.00%")
        else:
            print(f"  Total Return: {total_return:.2%}")
        
        if np.isnan(sharpe):
            print(f"  Sharpe Ratio: nan")
        else:
            print(f"  Sharpe Ratio: {sharpe:.2f}")
        
        if np.isnan(win_rate):
            print(f"  Win Rate: 0.00%") 
        else:
            print(f"  Win Rate: {win_rate:.2%}")
        
        print(f"  Signals - Long: {long_signals}, Short: {short_signals}")


def create_comparison_plot(results, buy_hold_return):
    """Create comparison plot of strategy performance."""
    try:
        strategies = [r['strategy'] for r in results]
        returns = [r['total_return'] for r in results]
        
        # Replace NaN and 0 values for plotting
        returns = [r if not (np.isnan(r) or r == 0) else -0.01 for r in returns]
        
        plt.figure(figsize=(12, 8))
        
        # Create bar plot
        bars = plt.bar(strategies, [r * 100 for r in returns], 
                      color=['green' if r > 0 else 'red' for r in returns])
        
        # Add buy & hold line
        plt.axhline(y=buy_hold_return * 100, color='blue', linestyle='--', 
                   label=f'Buy & Hold ({buy_hold_return:.2%})')
        
        plt.title('Signal Strategy Performance Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Total Return (%)', fontsize=12)
        plt.xlabel('Strategy', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, ret in zip(bars, returns):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{ret:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'signal_strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved comparison plot to output/signal_strategy_comparison.png")
        
    except Exception as e:
        logger.error(f"Error creating comparison plot: {e}")


def print_summary(results, buy_hold_return):
    """Print summary and insights."""
    
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    # Find best performing strategies
    valid_results = [r for r in results if not np.isnan(r['total_return']) and r['total_return'] != 0]
    
    if valid_results:
        best_return = max(valid_results, key=lambda x: x['total_return'])
        best_sharpe = max([r for r in valid_results if not np.isnan(r['sharpe'])], 
                         key=lambda x: x['sharpe'], default=None)
        
        print(f"\n🏆 Best Total Return: {best_return['strategy']} ({best_return['total_return']:.2%})")
        
        if best_sharpe:
            print(f"🏆 Best Sharpe Ratio: {best_sharpe['strategy']} ({best_sharpe['sharpe']:.2f})")
        else:
            print("🏆 Best Sharpe Ratio: None (all strategies have 0 or NaN Sharpe)")
    else:
        print("🏆 Best Total Return: None (all strategies returned 0%)")
        print("🏆 Best Sharpe Ratio: None (all strategies have 0 or NaN Sharpe)")
    
    print(f"\n📊 Buy & Hold Return: {buy_hold_return:.2%}")
    
    # Print insights
    print(f"\n💡 Key Insights:")
    
    # Check if mean return strategies are now working
    mean_return_strategies = [r for r in results if 'Mean Return' in r['strategy']]
    working_strategies = [r for r in mean_return_strategies if r['signals']['long'] + r['signals']['short'] > 0]
    
    if working_strategies:
        print("✅ Mean return strategies are now generating signals!")
        for strategy in working_strategies:
            total_signals = strategy['signals']['long'] + strategy['signals']['short']
            print(f"   - {strategy['strategy']}: {total_signals} signals")
    else:
        print("❌ Mean return strategies still generating 0 signals")
        print("   - Consider further reducing thresholds (try 0.010 or 0.005)")
    
    print("- Mean return signals can capture different market dynamics")
    print("- EWM gives more weight to recent returns vs simple average") 
    print("- Combining signals (AND/OR) affects trade frequency and quality")
    print("- Consider market regime when choosing signal type")
    
    # Threshold recommendations
    print(f"\n🔧 Current Thresholds:")
    print("- Mean Return EWM: 0.015 (1.5%)")
    print("- Mean Return Simple: 0.015 (1.5%)")
    print("- If still getting 0 signals, try 0.010 (1.0%) or 0.005 (0.5%)")


def main():
    """Main test function."""
    print("🔍 Mean Return Signal Strategy Test")
    print("="*60)
    
    try:
        # Load test data
        print("📊 Loading BTC data...")
        df = load_test_data()
        print(f"Loaded {len(df)} days of data")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        # Test signal strategies  
        results = test_signal_strategies(df)
        
        # Calculate buy & hold benchmark
        buy_hold_return = calculate_buy_hold_return(df)
        
        # Print results
        print_results(results, buy_hold_return)
        
        # Create visualization
        create_comparison_plot(results, buy_hold_return)
        
        # Print summary
        print_summary(results, buy_hold_return)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()