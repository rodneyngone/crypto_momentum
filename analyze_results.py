#!/usr/bin/env python3
"""
Analyze backtest results and suggest improvements.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(output_dir='output'):
    """Load backtest results from output directory."""
    output_path = Path(output_dir)
    
    results = {}
    
    # Load portfolio values
    portfolio_path = output_path / 'portfolio_values.csv'
    if portfolio_path.exists():
        results['portfolio'] = pd.read_csv(portfolio_path, index_col=0, parse_dates=True)
        print(f"✓ Loaded portfolio values: {len(results['portfolio'])} days")
    
    # Load trades
    trades_path = output_path / 'trades.csv'
    if trades_path.exists():
        results['trades'] = pd.read_csv(trades_path, parse_dates=['date'])
        print(f"✓ Loaded trades: {len(results['trades'])} trades")
    
    # Load metrics
    metrics_path = output_path / 'metrics.csv'
    if metrics_path.exists():
        results['metrics'] = pd.read_csv(metrics_path, index_col=0)
        print(f"✓ Loaded metrics")
    
    return results


def analyze_performance(results):
    """Analyze performance and identify issues."""
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    portfolio = results.get('portfolio', pd.DataFrame())
    trades = results.get('trades', pd.DataFrame())
    
    if portfolio.empty:
        print("No portfolio data found!")
        return
    
    # Calculate returns
    returns = portfolio.pct_change().dropna()
    
    # 1. Basic Statistics
    print("\n1. BASIC STATISTICS:")
    print(f"   - Starting Value: ${portfolio.iloc[0].values[0]:,.0f}")
    print(f"   - Ending Value: ${portfolio.iloc[-1].values[0]:,.0f}")
    print(f"   - Total Return: {((portfolio.iloc[-1] / portfolio.iloc[0]) - 1).values[0]:.2%}")
    print(f"   - Trading Days: {len(portfolio)}")
    
    # 2. Risk Metrics
    print("\n2. RISK METRICS:")
    print(f"   - Volatility (Annual): {returns.std().values[0] * np.sqrt(252):.2%}")
    print(f"   - Best Day: {returns.max().values[0]:.2%}")
    print(f"   - Worst Day: {returns.min().values[0]:.2%}")
    print(f"   - % Positive Days: {(returns > 0).sum().values[0] / len(returns):.1%}")
    
    # 3. Trading Analysis
    if not trades.empty:
        print("\n3. TRADING ANALYSIS:")
        print(f"   - Total Trades: {len(trades)}")
        print(f"   - Unique Assets Traded: {trades['symbol'].nunique()}")
        print(f"   - Average Trades per Month: {len(trades) / (len(portfolio) / 21):.1f}")
        print(f"   - Average Trade Size: ${trades['value'].mean():,.0f}")
        
        # Most traded assets
        print("\n   Most Traded Assets:")
        top_traded = trades['symbol'].value_counts().head(5)
        for symbol, count in top_traded.items():
            print(f"   - {symbol}: {count} trades")
        
        # Trade distribution by side
        print(f"\n   Trade Distribution:")
        print(f"   - Buys: {(trades['side'] == 'buy').sum()} ({(trades['side'] == 'buy').sum() / len(trades):.1%})")
        print(f"   - Sells: {(trades['side'] == 'sell').sum()} ({(trades['side'] == 'sell').sum() / len(trades):.1%})")
    
    # 4. Period Analysis
    print("\n4. PERIOD ANALYSIS:")
    
    # Monthly returns
    monthly_returns = portfolio.resample('M').last().pct_change().dropna()
    print(f"   - Positive Months: {(monthly_returns > 0).sum().values[0]} / {len(monthly_returns)}")
    print(f"   - Best Month: {monthly_returns.max().values[0]:.2%}")
    print(f"   - Worst Month: {monthly_returns.min().values[0]:.2%}")
    
    # 5. Issues Identified
    print("\n5. POTENTIAL ISSUES IDENTIFIED:")
    
    issues = []
    
    # Check if strategy is trading enough
    if not trades.empty and len(trades) < 50:
        issues.append("Low trading frequency - strategy might be too conservative")
    
    # Check if returns are consistently negative
    if (monthly_returns < 0).sum().values[0] > len(monthly_returns) * 0.7:
        issues.append("Consistently negative returns - strategy may not suit bear market")
    
    # Check if volatility is too low
    if returns.std().values[0] * np.sqrt(252) < 0.10:
        issues.append("Very low volatility - positions might be too small")
    
    # Check concentration
    if not trades.empty and trades['symbol'].nunique() < 10:
        issues.append("Low diversification - trading too few assets")
    
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("   No major issues identified")
    
    return returns, trades


def suggest_improvements():
    """Suggest parameter improvements."""
    print("\n" + "="*60)
    print("SUGGESTED IMPROVEMENTS")
    print("="*60)
    
    print("\n1. SIGNAL GENERATION:")
    print("   - Consider lowering momentum_threshold from 0.01 to 0.005")
    print("   - Try multi_timeframe or ensemble signal strategies")
    print("   - Enable adaptive_ewma for volatile markets")
    
    print("\n2. PORTFOLIO OPTIMIZATION:")
    print("   - Increase max_position_size from 0.25 to 0.30")
    print("   - Enable concentration_mode with top_n_assets=5")
    print("   - Increase momentum_tilt_strength to 0.7")
    
    print("\n3. REBALANCING:")
    print("   - Change from 'weekly' to 'daily' for trending markets")
    print("   - Enable use_dynamic_rebalancing")
    
    print("\n4. RISK MANAGEMENT:")
    print("   - Consider using trailing stops")
    print("   - Adjust regime parameters for bear markets")
    
    print("\n5. ALTERNATIVE CONFIGURATIONS:")
    print("   Try these config changes:")
    print("""
signals:
  signal_strategy: multi_timeframe  # or ensemble
  momentum_threshold: 0.005
  min_score_threshold: 0.2
  use_volume_confirmation: true
  
portfolio:
  max_position_size: 0.30
  concentration_mode: true
  top_n_assets: 5
  momentum_tilt_strength: 0.7
  
risk:
  use_trailing_stops: true
  trailing_stop_atr_multiplier: 2.0
""")


def create_enhanced_config():
    """Create an enhanced configuration file."""
    enhanced_config = """# Enhanced configuration for better performance
data:
  universe_size: 30
  selection_size: 15

signals:
  signal_strategy: ensemble  # Use ensemble of strategies
  momentum_threshold: 0.005  # More sensitive
  min_score_threshold: 0.2   # Lower threshold
  adx_periods: [7, 14, 21]
  adaptive_ewma: true
  use_volume_confirmation: true
  volume_threshold: 1.1      # Lower volume requirement

portfolio:
  optimization_method: enhanced_risk_parity
  max_position_size: 0.30    # Larger positions
  concentration_mode: true
  top_n_assets: 5
  concentration_weight: 0.7
  momentum_tilt_strength: 0.7
  base_rebalance_frequency: daily
  use_dynamic_rebalancing: true

risk:
  max_drawdown: 0.25         # Allow more drawdown
  use_trailing_stops: true
  trailing_stop_atr_multiplier: 2.0
  regime_lookback: 30        # Faster regime detection

backtest:
  walk_forward_analysis: false
  output_directory: output_enhanced
"""
    
    with open('config_enhanced.yaml', 'w') as f:
        f.write(enhanced_config)
    
    print("\n✓ Created 'config_enhanced.yaml' with optimized parameters")
    print("\nRun with: python run.py --config config_enhanced.yaml --no-validate")


def main():
    """Main analysis function."""
    print("BACKTEST RESULTS ANALYSIS")
    print("="*60)
    
    # Load results
    results = load_results()
    
    if not results:
        print("No results found in output directory!")
        return
    
    # Analyze performance
    returns, trades = analyze_performance(results)
    
    # Suggest improvements
    suggest_improvements()
    
    # Create enhanced config
    create_enhanced_config()
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("1. Review the analysis above")
    print("2. Try the enhanced configuration")
    print("3. Consider testing different time periods")
    print("4. Experiment with different signal strategies")
    print("="*60)


if __name__ == "__main__":
    main()