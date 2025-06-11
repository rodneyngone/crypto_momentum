#!/usr/bin/env python3
"""
Parameter optimization script for crypto momentum strategies.
This will help you find optimal thresholds and parameters.
"""

import pandas as pd
import numpy as np
import yaml
from itertools import product
from pathlib import Path

class ParameterOptimizer:
    """Optimize strategy parameters"""
    
    def __init__(self, data_path="data"):
        self.data_path = Path(data_path)
        
    def load_data(self, symbol="BTCUSDT"):
        """Load crypto data"""
        possible_files = [
            self.data_path / f"{symbol}.csv",
            f"{symbol}.csv",
            "btc_data.csv"
        ]
        
        for file_path in possible_files:
            if Path(file_path).exists():
                return pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Create sample data if no file found
        return self.create_sample_data()
    
    def create_sample_data(self):
        """Create realistic sample crypto data"""
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        
        initial_price = 40000
        returns = np.random.normal(0.001, 0.04, len(dates))  # Crypto-like volatility
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        return pd.DataFrame({
            'close': prices,
            'volume': np.random.uniform(1000000, 10000000, len(dates))
        }, index=dates)
    
    def calculate_strategy_returns(self, data, strategy_type, **params):
        """Calculate returns for a given strategy"""
        data = data.copy()
        data['returns'] = data['close'].pct_change().dropna()
        
        if strategy_type == "momentum":
            return self.momentum_strategy(data, **params)
        elif strategy_type == "mean_return_ewm":
            return self.mean_return_ewm_strategy(data, **params)
        elif strategy_type == "mean_return_simple":
            return self.mean_return_simple_strategy(data, **params)
        else:
            raise ValueError(f"Unknown strategy: {strategy_type}")
    
    def momentum_strategy(self, data, ewm_span=20, threshold=0.02):
        """Original momentum strategy"""
        data['ewma'] = data['close'].ewm(span=ewm_span).mean()
        data['momentum'] = (data['close'] / data['ewma'] - 1)
        
        long_signals = data['momentum'] > threshold
        short_signals = data['momentum'] < -threshold
        
        # Calculate strategy returns
        positions = long_signals.astype(int) - short_signals.astype(int)
        strategy_returns = positions.shift(1) * data['returns']
        
        return strategy_returns.dropna()
    
    def mean_return_ewm_strategy(self, data, ewm_span=20, threshold=0.005):
        """EWM mean return strategy"""
        ewm_returns = data['returns'].ewm(span=ewm_span).mean()
        
        long_signals = ewm_returns > threshold
        short_signals = ewm_returns < -threshold
        
        positions = long_signals.astype(int) - short_signals.astype(int)
        strategy_returns = positions.shift(1) * data['returns']
        
        return strategy_returns.dropna()
    
    def mean_return_simple_strategy(self, data, window=5, threshold=0.005):
        """Simple mean return strategy"""
        simple_returns = data['returns'].rolling(window=window).mean()
        
        long_signals = simple_returns > threshold
        short_signals = simple_returns < -threshold
        
        positions = long_signals.astype(int) - short_signals.astype(int)
        strategy_returns = positions.shift(1) * data['returns']
        
        return strategy_returns.dropna()
    
    def calculate_metrics(self, returns):
        """Calculate performance metrics"""
        if len(returns) == 0 or returns.std() == 0:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'num_trades': 0
            }
        
        total_return = (1 + returns).prod() - 1
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        
        # Calculate max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Number of trades (position changes)
        positions = np.sign(returns)
        num_trades = (positions != positions.shift(1)).sum()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': num_trades
        }
    
    def optimize_momentum_strategy(self, data):
        """Optimize momentum strategy parameters"""
        print("🔧 Optimizing Momentum Strategy Parameters...")
        
        # Parameter ranges
        ewm_spans = [5, 10, 15, 20, 30, 50]
        thresholds = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05]
        
        results = []
        
        for ewm_span, threshold in product(ewm_spans, thresholds):
            strategy_returns = self.momentum_strategy(data, ewm_span, threshold)
            metrics = self.calculate_metrics(strategy_returns)
            
            results.append({
                'strategy': 'momentum',
                'ewm_span': ewm_span,
                'threshold': threshold,
                **metrics
            })
        
        return pd.DataFrame(results)
    
    def optimize_mean_return_ewm_strategy(self, data):
        """Optimize EWM mean return strategy parameters"""
        print("🔧 Optimizing Mean Return EWM Strategy Parameters...")
        
        # Parameter ranges (smaller thresholds for mean returns)
        ewm_spans = [3, 5, 10, 15, 20, 30]
        thresholds = [0.001, 0.002, 0.005, 0.007, 0.01, 0.015, 0.02]
        
        results = []
        
        for ewm_span, threshold in product(ewm_spans, thresholds):
            strategy_returns = self.mean_return_ewm_strategy(data, ewm_span, threshold)
            metrics = self.calculate_metrics(strategy_returns)
            
            results.append({
                'strategy': 'mean_return_ewm',
                'ewm_span': ewm_span,
                'threshold': threshold,
                **metrics
            })
        
        return pd.DataFrame(results)
    
    def optimize_mean_return_simple_strategy(self, data):
        """Optimize simple mean return strategy parameters"""
        print("🔧 Optimizing Simple Mean Return Strategy Parameters...")
        
        # Parameter ranges
        windows = [2, 3, 5, 7, 10, 15, 20]
        thresholds = [0.001, 0.002, 0.005, 0.007, 0.01, 0.015, 0.02]
        
        results = []
        
        for window, threshold in product(windows, thresholds):
            strategy_returns = self.mean_return_simple_strategy(data, window, threshold)
            metrics = self.calculate_metrics(strategy_returns)
            
            results.append({
                'strategy': 'mean_return_simple',
                'window': window,
                'threshold': threshold,
                **metrics
            })
        
        return pd.DataFrame(results)
    
    def find_best_parameters(self, results_df, metric='sharpe_ratio', min_trades=10):
        """Find best parameters based on specified metric"""
        # Filter for minimum number of trades
        filtered = results_df[results_df['num_trades'] >= min_trades].copy()
        
        if len(filtered) == 0:
            print(f"⚠️  No results with minimum {min_trades} trades found!")
            return results_df.loc[results_df[metric].idxmax()]
        
        return filtered.loc[filtered[metric].idxmax()]
    
    def create_optimized_config(self, best_params_dict):
        """Create optimized configuration file"""
        config = {
            'data': {
                'universe_size': 10,
                'start_date': '2022-01-01',
                'end_date': '2023-12-31'
            },
            'signals': {},
            'portfolio': {
                'max_position_size': 0.10,
                'rebalance_frequency': 'daily'
            },
            'risk': {
                'max_drawdown': 0.20
            }
        }
        
        # Add optimized parameters for each strategy
        for strategy, params in best_params_dict.items():
            if strategy == 'momentum':
                config['signals']['momentum'] = {
                    'ewm_span': int(params['ewm_span']),
                    'threshold': float(params['threshold'])
                }
            elif strategy == 'mean_return_ewm':
                config['signals']['mean_return_ewm'] = {
                    'ewm_span': int(params['ewm_span']),
                    'threshold': float(params['threshold'])
                }
            elif strategy == 'mean_return_simple':
                config['signals']['mean_return_simple'] = {
                    'window': int(params['window']),
                    'threshold': float(params['threshold'])
                }
        
        return config
    
    def run_optimization(self):
        """Run complete parameter optimization"""
        print("🚀 STARTING PARAMETER OPTIMIZATION")
        print("="*60)
        
        # Load data
        data = self.load_data()
        print(f"Loaded data: {len(data)} rows")
        
        # Optimize each strategy
        momentum_results = self.optimize_momentum_strategy(data)
        ewm_results = self.optimize_mean_return_ewm_strategy(data)
        simple_results = self.optimize_mean_return_simple_strategy(data)
        
        # Find best parameters for each strategy
        print("\n📊 OPTIMIZATION RESULTS")
        print("="*60)
        
        best_params = {}
        
        # Momentum strategy
        best_momentum = self.find_best_parameters(momentum_results)
        best_params['momentum'] = best_momentum
        print(f"\n🏆 Best Momentum Strategy:")
        print(f"  EWM Span: {best_momentum['ewm_span']}")
        print(f"  Threshold: {best_momentum['threshold']:.4f}")
        print(f"  Sharpe Ratio: {best_momentum['sharpe_ratio']:.2f}")
        print(f"  Total Return: {best_momentum['total_return']:.2%}")
        print(f"  Max Drawdown: {best_momentum['max_drawdown']:.2%}")
        print(f"  Number of Trades: {best_momentum['num_trades']}")
        
        # EWM mean return strategy
        best_ewm = self.find_best_parameters(ewm_results)
        best_params['mean_return_ewm'] = best_ewm
        print(f"\n🏆 Best Mean Return EWM Strategy:")
        print(f"  EWM Span: {best_ewm['ewm_span']}")
        print(f"  Threshold: {best_ewm['threshold']:.4f}")
        print(f"  Sharpe Ratio: {best_ewm['sharpe_ratio']:.2f}")
        print(f"  Total Return: {best_ewm['total_return']:.2%}")
        print(f"  Max Drawdown: {best_ewm['max_drawdown']:.2%}")
        print(f"  Number of Trades: {best_ewm['num_trades']}")
        
        # Simple mean return strategy
        best_simple = self.find_best_parameters(simple_results)
        best_params['mean_return_simple'] = best_simple
        print(f"\n🏆 Best Simple Mean Return Strategy:")
        print(f"  Window: {best_simple['window']}")
        print(f"  Threshold: {best_simple['threshold']:.4f}")
        print(f"  Sharpe Ratio: {best_simple['sharpe_ratio']:.2f}")
        print(f"  Total Return: {best_simple['total_return']:.2%}")
        print(f"  Max Drawdown: {best_simple['max_drawdown']:.2%}")
        print(f"  Number of Trades: {best_simple['num_trades']}")
        
        # Save detailed results
        all_results = pd.concat([momentum_results, ewm_results, simple_results], ignore_index=True)
        all_results.to_csv('optimization_results.csv', index=False)
        print(f"\n💾 Detailed results saved to: optimization_results.csv")
        
        # Create optimized config
        optimized_config = self.create_optimized_config(best_params)
        with open('config_optimized.yaml', 'w') as f:
            yaml.dump(optimized_config, f, default_flow_style=False)
        print(f"💾 Optimized config saved to: config_optimized.yaml")
        
        # Show top 5 for each strategy
        print(f"\n📈 TOP 5 CONFIGURATIONS BY SHARPE RATIO")
        print("="*60)
        
        for strategy in ['momentum', 'mean_return_ewm', 'mean_return_simple']:
            strategy_results = all_results[all_results['strategy'] == strategy]
            top5 = strategy_results.nlargest(5, 'sharpe_ratio')
            
            print(f"\n{strategy.upper()}:")
            for i, row in top5.iterrows():
                if strategy == 'momentum':
                    print(f"  {row['ewm_span']:2.0f} span, {row['threshold']:.4f} thresh -> Sharpe: {row['sharpe_ratio']:.2f}, Return: {row['total_return']:.1%}")
                elif strategy == 'mean_return_ewm':
                    print(f"  {row['ewm_span']:2.0f} span, {row['threshold']:.4f} thresh -> Sharpe: {row['sharpe_ratio']:.2f}, Return: {row['total_return']:.1%}")
                else:
                    print(f"  {row['window']:2.0f} window, {row['threshold']:.4f} thresh -> Sharpe: {row['sharpe_ratio']:.2f}, Return: {row['total_return']:.1%}")
        
        print("\n" + "="*60)
        print("🎯 OPTIMIZATION COMPLETE!")
        print("="*60)
        print("Next steps:")
        print("1. Review optimization_results.csv for detailed analysis")
        print("2. Use config_optimized.yaml for your backtests")
        print("3. Test the optimized parameters with:")
        print("   copy config_optimized.yaml config.yaml")
        print("   python run.py --no-validate")

def main():
    optimizer = ParameterOptimizer()
    optimizer.run_optimization()

if __name__ == "__main__":
    main()