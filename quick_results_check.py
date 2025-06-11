#!/usr/bin/env python3
"""
Quick check of the backtest results from the improved configuration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def check_results():
    """Quick check of the latest results."""
    
    print("📊 Checking Latest Backtest Results")
    print("=" * 60)
    
    # Check if we have new results
    output_dir = Path("output")
    
    # List files with modification times
    print("\n📁 Output files:")
    for file in output_dir.glob("*"):
        if file.is_file():
            mod_time = pd.Timestamp.fromtimestamp(file.stat().st_mtime)
            print(f"  {file.name}: {mod_time}")
    
    # Load and analyze portfolio values
    try:
        portfolio_df = pd.read_csv('output/portfolio_values.csv', index_col=0, parse_dates=True)
        
        initial = portfolio_df.iloc[0].values[0]
        final = portfolio_df.iloc[-1].values[0]
        peak = portfolio_df.max().values[0]
        trough = portfolio_df.min().values[0]
        
        total_return = (final - initial) / initial
        max_dd = (trough - peak) / peak
        
        print(f"\n💰 Performance Summary:")
        print(f"Initial Value: ${initial:,.2f}")
        print(f"Final Value: ${final:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Peak Value: ${peak:,.2f}")
        print(f"Max Drawdown: {max_dd:.2%}")
        
        # Calculate Sharpe
        returns = portfolio_df.pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        print(f"Sharpe Ratio: {sharpe.values[0]:.2f}")
        
        # Quick plot
        plt.figure(figsize=(12, 6))
        portfolio_df.plot(linewidth=2)
        plt.axhline(y=initial, color='red', linestyle='--', alpha=0.5, label='Initial')
        plt.title('Portfolio Value - Updated Strategy')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('output/quick_check.png')
        print("\n📊 Saved quick_check.png")
        
        # Compare to previous run
        print("\n📈 Improvement Analysis:")
        print("Previous run (20 assets, slow signals): -43.74% return")
        print(f"Current run (10 assets, faster signals): {total_return:.2%} return")
        print(f"Improvement: {total_return - (-0.4374):.2%}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure the backtest has completed and saved results.")

def check_metrics():
    """Check detailed metrics if available."""
    try:
        metrics_df = pd.read_csv('output/metrics.csv', index_col=0)
        
        print("\n📊 Key Metrics:")
        key_metrics = ['total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown', 
                      'win_rate', 'annual_volatility']
        
        for metric in key_metrics:
            if metric in metrics_df.index:
                value = metrics_df.loc[metric].values[0]
                if 'return' in metric or 'rate' in metric or 'volatility' in metric:
                    print(f"{metric}: {value:.2%}")
                else:
                    print(f"{metric}: {value:.2f}")
                    
    except:
        pass

if __name__ == "__main__":
    check_results()
    check_metrics()