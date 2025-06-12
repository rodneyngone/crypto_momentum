#!/usr/bin/env python3
"""
Analyze long/short backtest results and suggest improvements.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


def load_and_analyze_trades(output_dir):
    """Load and analyze trade data."""
    trades_path = Path(output_dir) / 'trades.csv'
    
    if not trades_path.exists():
        print("No trades file found")
        return None
    
    trades = pd.read_csv(trades_path)
    trades['date'] = pd.to_datetime(trades['date'])
    
    # Analyze trade distribution
    analysis = {
        'total_trades': len(trades),
        'long_trades': len(trades[trades['position_type'] == 'long']),
        'short_trades': len(trades[trades['position_type'] == 'short']),
        'symbols_traded': trades['symbol'].nunique(),
        'avg_trade_value': trades['value'].mean(),
        'total_volume': trades['value'].sum()
    }
    
    # Analyze by symbol
    symbol_stats = trades.groupby('symbol').agg({
        'position_type': lambda x: (x == 'long').sum(),
        'value': ['count', 'sum', 'mean']
    })
    
    # Most traded symbols
    most_traded = trades['symbol'].value_counts().head(10)
    
    return trades, analysis, symbol_stats, most_traded


def analyze_position_performance(trades, portfolio_values):
    """Analyze performance of long vs short positions."""
    portfolio_values['date'] = pd.to_datetime(portfolio_values.index)
    
    # Create position tracking
    position_tracker = {}
    position_pnl = []
    
    for idx, trade in trades.iterrows():
        symbol = trade['symbol']
        date = trade['date']
        
        if trade['side'] == 'buy':
            # Opening or adding to position
            if symbol not in position_tracker:
                position_tracker[symbol] = {
                    'entry_date': date,
                    'entry_price': trade['price'],
                    'position_type': trade['position_type'],
                    'size': trade['units']
                }
            else:
                # Average in
                position_tracker[symbol]['size'] += trade['units']
        
        elif trade['side'] == 'sell' and symbol in position_tracker:
            # Closing position
            entry = position_tracker[symbol]
            
            # Calculate P&L
            if entry['position_type'] == 'long':
                pnl_pct = (trade['price'] - entry['entry_price']) / entry['entry_price']
            else:  # short
                pnl_pct = (entry['entry_price'] - trade['price']) / entry['entry_price']
            
            position_pnl.append({
                'symbol': symbol,
                'position_type': entry['position_type'],
                'entry_date': entry['entry_date'],
                'exit_date': date,
                'holding_days': (date - entry['entry_date']).days,
                'pnl_pct': pnl_pct,
                'pnl_value': pnl_pct * trade['value']
            })
            
            # Remove from tracker
            del position_tracker[symbol]
    
    # Convert to DataFrame
    pnl_df = pd.DataFrame(position_pnl)
    
    if not pnl_df.empty:
        # Calculate statistics
        long_pnl = pnl_df[pnl_df['position_type'] == 'long']
        short_pnl = pnl_df[pnl_df['position_type'] == 'short']
        
        stats = {
            'long': {
                'count': len(long_pnl),
                'win_rate': (long_pnl['pnl_pct'] > 0).mean() if len(long_pnl) > 0 else 0,
                'avg_pnl': long_pnl['pnl_pct'].mean() if len(long_pnl) > 0 else 0,
                'total_pnl': long_pnl['pnl_value'].sum() if len(long_pnl) > 0 else 0
            },
            'short': {
                'count': len(short_pnl),
                'win_rate': (short_pnl['pnl_pct'] > 0).mean() if len(short_pnl) > 0 else 0,
                'avg_pnl': short_pnl['pnl_pct'].mean() if len(short_pnl) > 0 else 0,
                'total_pnl': short_pnl['pnl_value'].sum() if len(short_pnl) > 0 else 0
            }
        }
        
        return pnl_df, stats
    
    return None, None


def create_analysis_plots(trades, portfolio_values, output_dir):
    """Create analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Long/Short Strategy Analysis', fontsize=16)
    
    # 1. Portfolio value with market regime shading
    ax = axes[0, 0]
    portfolio_values.plot(ax=ax, linewidth=2)
    ax.set_title('Portfolio Value Over Time')
    ax.set_ylabel('Portfolio Value ($)')
    ax.grid(True, alpha=0.3)
    
    # 2. Long vs Short trade distribution over time
    ax = axes[0, 1]
    trades['month'] = trades['date'].dt.to_period('M')
    monthly_trades = trades.groupby(['month', 'position_type']).size().unstack(fill_value=0)
    
    if not monthly_trades.empty:
        monthly_trades.plot(kind='bar', ax=ax, color=['green', 'red'])
        ax.set_title('Monthly Long vs Short Trades')
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Trades')
        ax.legend(['Long', 'Short'])
        ax.tick_params(axis='x', rotation=45)
    
    # 3. Top traded symbols
    ax = axes[1, 0]
    top_symbols = trades['symbol'].value_counts().head(10)
    top_symbols.plot(kind='barh', ax=ax)
    ax.set_title('Top 10 Most Traded Symbols')
    ax.set_xlabel('Number of Trades')
    
    # 4. Trade size distribution
    ax = axes[1, 1]
    trades['value'].hist(bins=50, ax=ax, alpha=0.7)
    ax.set_title('Trade Size Distribution')
    ax.set_xlabel('Trade Value ($)')
    ax.set_ylabel('Frequency')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'long_short_analysis.png', dpi=150)
    plt.close()


def generate_recommendations(analysis, pnl_stats):
    """Generate recommendations based on analysis."""
    recommendations = []
    
    # Check long/short balance
    if analysis['short_trades'] > analysis['long_trades'] * 2:
        recommendations.append("Consider reducing short bias - market may be oversold")
    
    # Check win rates
    if pnl_stats and pnl_stats['long']['win_rate'] < 0.3:
        recommendations.append("Long positions have low win rate - review entry criteria")
    
    if pnl_stats and pnl_stats['short']['win_rate'] < 0.3:
        recommendations.append("Short positions have low win rate - consider tighter stops")
    
    # Check trade frequency
    if analysis['total_trades'] < 50:
        recommendations.append("Low trade count - consider lowering signal thresholds")
    
    return recommendations


def main():
    output_dir = 'output_long_short'
    
    print("="*60)
    print("LONG/SHORT STRATEGY ANALYSIS")
    print("="*60)
    
    # Load portfolio values
    portfolio_path = Path(output_dir) / 'portfolio_values.csv'
    portfolio_values = pd.read_csv(portfolio_path, index_col=0, parse_dates=True)
    
    # Load and analyze trades
    trades, analysis, symbol_stats, most_traded = load_and_analyze_trades(output_dir)
    
    if trades is not None:
        print("\n1. TRADE SUMMARY")
        print("-"*40)
        for key, value in analysis.items():
            print(f"{key}: {value}")
        
        print("\n2. MOST TRADED SYMBOLS")
        print("-"*40)
        print(most_traded)
        
        # Analyze position performance
        pnl_df, pnl_stats = analyze_position_performance(trades, portfolio_values)
        
        if pnl_stats:
            print("\n3. POSITION PERFORMANCE")
            print("-"*40)
            print("\nLONG POSITIONS:")
            for key, value in pnl_stats['long'].items():
                print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
            
            print("\nSHORT POSITIONS:")
            for key, value in pnl_stats['short'].items():
                print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # Create plots
        create_analysis_plots(trades, portfolio_values['portfolio_value'], output_dir)
        print("\n✓ Analysis plots saved to output_long_short/long_short_analysis.png")
        
        # Generate recommendations
        recommendations = generate_recommendations(analysis, pnl_stats)
        if recommendations:
            print("\n4. RECOMMENDATIONS")
            print("-"*40)
            for rec in recommendations:
                print(f"• {rec}")
    
    print("\n5. NEXT STEPS")
    print("-"*40)
    print("1. Try the improved configuration: config_improved_long_short.yaml")
    print("2. Test on different time periods (2020-2021 bull market)")
    print("3. Consider adding market regime filters")
    print("4. Experiment with different signal generation methods")
    print("5. Add pairs trading for market-neutral strategies")


if __name__ == "__main__":
    main()