#!/usr/bin/env python3
"""
Advanced parameter optimization for crypto momentum strategies.
Includes walk-forward optimization and regime-aware parameter selection.
"""

import pandas as pd
import numpy as np
import yaml
from itertools import product
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container for optimization results"""
    strategy: str
    params: Dict
    metrics: Dict
    in_sample_performance: float
    out_sample_performance: float
    stability_score: float


class AdvancedParameterOptimizer:
    """Advanced parameter optimizer with walk-forward analysis."""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.results_cache = {}
        
    def load_data(self, symbol: str = "BTCUSDT") -> pd.DataFrame:
        """Load crypto data with caching."""
        if symbol in self.results_cache:
            return self.results_cache[symbol]
            
        # Try to load from your data structure
        try:
            from crypto_momentum_backtest.data.json_storage import JsonStorage
            storage = JsonStorage(self.data_path)
            df = storage.load_range(
                symbol=symbol,
                start_date=pd.Timestamp('2021-01-01'),
                end_date=pd.Timestamp('2023-12-31')
            )
            self.results_cache[symbol] = df
            return df
        except:
            # Fallback to CSV or create sample data
            logger.warning(f"Could not load {symbol} from storage, creating sample data")
            return self.create_sample_data()
    
    def create_sample_data(self) -> pd.DataFrame:
        """Create realistic crypto sample data."""
        dates = pd.date_range('2021-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        
        # Multi-regime price simulation
        regimes = self._generate_market_regimes(len(dates))
        prices = self._generate_regime_prices(regimes, initial_price=40000)
        
        return pd.DataFrame({
            'open': prices * np.random.uniform(0.99, 1.01, len(prices)),
            'high': prices * np.random.uniform(1.0, 1.02, len(prices)),
            'low': prices * np.random.uniform(0.98, 1.0, len(prices)),
            'close': prices,
            'volume': np.random.uniform(1e9, 1e10, len(dates)) * (1 + regimes['volatility'])
        }, index=dates)
    
    def _generate_market_regimes(self, n_days: int) -> pd.DataFrame:
        """Generate market regime labels."""
        # Define regimes: Bull, Bear, Sideways, High Vol
        regime_probs = np.array([0.3, 0.2, 0.3, 0.2])  # Probabilities
        regime_names = ['bull', 'bear', 'sideways', 'high_vol']
        
        # Generate regime sequence with persistence
        regimes = []
        current_regime = np.random.choice(4, p=regime_probs)
        
        for _ in range(n_days):
            regimes.append(current_regime)
            # 95% chance to stay in current regime
            if np.random.random() > 0.95:
                current_regime = np.random.choice(4, p=regime_probs)
        
        # Create regime features
        regime_df = pd.DataFrame({
            'regime': [regime_names[r] for r in regimes],
            'trend': [0.001 if r == 0 else -0.001 if r == 1 else 0 for r in regimes],
            'volatility': [0.03 if r < 3 else 0.06 for r in regimes]
        })
        
        return regime_df
    
    def _generate_regime_prices(self, regimes: pd.DataFrame, initial_price: float) -> np.ndarray:
        """Generate prices based on market regimes."""
        prices = [initial_price]
        
        for i in range(1, len(regimes)):
            trend = regimes.iloc[i]['trend']
            vol = regimes.iloc[i]['volatility']
            
            # Generate return with regime characteristics
            ret = np.random.normal(trend, vol)
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        return np.array(prices)
    
    def walk_forward_optimization(
        self,
        data: pd.DataFrame,
        strategy: str,
        param_grid: Dict[str, List],
        n_splits: int = 5,
        test_pct: float = 0.2
    ) -> List[OptimizationResult]:
        """
        Perform walk-forward optimization.
        
        Args:
            data: Price data
            strategy: Strategy name
            param_grid: Parameter grid to search
            n_splits: Number of walk-forward splits
            test_pct: Percentage of data for out-of-sample testing
            
        Returns:
            List of optimization results
        """
        results = []
        n_days = len(data)
        test_days = int(n_days * test_pct)
        
        # Create time splits
        for split in range(n_splits):
            # Calculate split boundaries
            split_size = (n_days - test_days) // n_splits
            train_start = split * split_size
            train_end = train_start + split_size
            test_start = train_end
            test_end = min(test_start + test_days, n_days)
            
            # Split data
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # Find best parameters on training data
            best_params, best_metrics = self._optimize_on_period(
                train_data, strategy, param_grid
            )
            
            # Test on out-of-sample data
            test_performance = self._test_strategy(
                test_data, strategy, best_params
            )
            
            # Calculate stability score
            stability = self._calculate_stability(best_metrics, test_performance)
            
            result = OptimizationResult(
                strategy=strategy,
                params=best_params,
                metrics=best_metrics,
                in_sample_performance=best_metrics['sharpe_ratio'],
                out_sample_performance=test_performance['sharpe_ratio'],
                stability_score=stability
            )
            
            results.append(result)
            
            logger.info(f"Split {split+1}/{n_splits}: "
                       f"IS Sharpe={result.in_sample_performance:.2f}, "
                       f"OOS Sharpe={result.out_sample_performance:.2f}")
        
        return results
    
    def _optimize_on_period(
        self,
        data: pd.DataFrame,
        strategy: str,
        param_grid: Dict[str, List]
    ) -> Tuple[Dict, Dict]:
        """Optimize parameters on a single period."""
        best_sharpe = -np.inf
        best_params = {}
        best_metrics = {}
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for params in product(*param_values):
            param_dict = dict(zip(param_names, params))
            
            # Test strategy with these parameters
            metrics = self._test_strategy(data, strategy, param_dict)
            
            if metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                best_params = param_dict
                best_metrics = metrics
        
        return best_params, best_metrics
    
    def _test_strategy(self, data: pd.DataFrame, strategy: str, params: Dict) -> Dict:
        """Test a strategy with given parameters."""
        data = data.copy()
        data['returns'] = data['close'].pct_change().fillna(0)
        
        # Generate signals based on strategy
        if strategy == 'momentum':
            signals = self._momentum_signals(data, **params)
        elif strategy == 'mean_return_ewm':
            signals = self._mean_return_ewm_signals(data, **params)
        elif strategy == 'mean_return_simple':
            signals = self._mean_return_simple_signals(data, **params)
        elif strategy == 'rsi':
            signals = self._rsi_signals(data, **params)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Calculate strategy returns
        strategy_returns = signals.shift(1) * data['returns']
        strategy_returns = strategy_returns.dropna()
        
        # Calculate metrics
        return self._calculate_metrics(strategy_returns)
    
    def _momentum_signals(self, data: pd.DataFrame, ewm_span: int, threshold: float) -> pd.Series:
        """Generate momentum signals."""
        ewma = data['close'].ewm(span=ewm_span).mean()
        momentum = (data['close'] / ewma - 1)
        
        signals = pd.Series(0, index=data.index)
        signals[momentum > threshold] = 1
        signals[momentum < -threshold] = -1
        
        return signals
    
    def _mean_return_ewm_signals(self, data: pd.DataFrame, ewm_span: int, threshold: float) -> pd.Series:
        """Generate EWM mean return signals."""
        ewm_returns = data['returns'].ewm(span=ewm_span).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[ewm_returns > threshold] = 1
        signals[ewm_returns < -threshold] = -1
        
        return signals
    
    def _mean_return_simple_signals(self, data: pd.DataFrame, window: int, threshold: float) -> pd.Series:
        """Generate simple mean return signals."""
        mean_returns = data['returns'].rolling(window=window).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[mean_returns > threshold] = 1
        signals[mean_returns < -threshold] = -1
        
        return signals
    
    def _rsi_signals(self, data: pd.DataFrame, period: int, oversold: float, overbought: float) -> pd.Series:
        """Generate RSI signals."""
        # Calculate RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=data.index)
        signals[rsi < oversold] = 1
        signals[rsi > overbought] = -1
        
        return signals
    
    def _calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if len(returns) == 0 or returns.std() == 0:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'num_trades': 0
            }
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate and profit factor
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        
        win_rate = len(winning_returns) / len(returns) if len(returns) > 0 else 0
        profit_factor = winning_returns.sum() / abs(losing_returns.sum()) if len(losing_returns) > 0 and losing_returns.sum() != 0 else 0
        
        # Number of trades
        positions = np.sign(returns)
        num_trades = (positions != positions.shift(1)).sum()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': num_trades
        }
    
    def _calculate_stability(self, in_sample_metrics: Dict, out_sample_metrics: Dict) -> float:
        """Calculate stability score between in-sample and out-of-sample performance."""
        # Compare key metrics
        sharpe_diff = abs(in_sample_metrics['sharpe_ratio'] - out_sample_metrics['sharpe_ratio'])
        return_diff = abs(in_sample_metrics['total_return'] - out_sample_metrics['total_return'])
        
        # Stability score (0-1, higher is better)
        stability = 1 / (1 + sharpe_diff + return_diff * 10)
        
        return stability
    
    def create_optimization_report(self, all_results: Dict[str, List[OptimizationResult]]) -> None:
        """Create comprehensive optimization report with visualizations."""
        # Create output directory
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        # 1. Parameter stability heatmap
        self._create_stability_heatmap(all_results, output_dir)
        
        # 2. Walk-forward performance chart
        self._create_walk_forward_chart(all_results, output_dir)
        
        # 3. Parameter importance analysis
        self._create_parameter_importance(all_results, output_dir)
        
        # 4. Generate report
        self._generate_text_report(all_results, output_dir)
    
    def _create_stability_heatmap(self, results: Dict, output_dir: Path) -> None:
        """Create heatmap showing parameter stability across strategies."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, (strategy, strategy_results) in enumerate(results.items()):
            if idx >= 4:
                break
                
            # Extract parameter values and stability scores
            param_data = []
            for result in strategy_results:
                param_dict = result.params.copy()
                param_dict['stability'] = result.stability_score
                param_data.append(param_dict)
            
            df = pd.DataFrame(param_data)
            
            # Create pivot table for heatmap
            if len(df.columns) >= 3:
                param_cols = [col for col in df.columns if col != 'stability'][:2]
                pivot = df.pivot_table(
                    values='stability',
                    index=param_cols[0],
                    columns=param_cols[1] if len(param_cols) > 1 else 'stability',
                    aggfunc='mean'
                )
                
                sns.heatmap(pivot, ax=axes[idx], cmap='RdYlGn', 
                           annot=True, fmt='.2f', cbar_kws={'label': 'Stability'})
                axes[idx].set_title(f'{strategy} Parameter Stability')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_stability_heatmap.png', dpi=300)
        plt.close()
    
    def _create_walk_forward_chart(self, results: Dict, output_dir: Path) -> None:
        """Create walk-forward performance comparison chart."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for strategy, strategy_results in results.items():
            splits = list(range(len(strategy_results)))
            is_sharpes = [r.in_sample_performance for r in strategy_results]
            oos_sharpes = [r.out_sample_performance for r in strategy_results]
            
            ax.plot(splits, is_sharpes, 'o-', label=f'{strategy} (IS)', linewidth=2)
            ax.plot(splits, oos_sharpes, 's--', label=f'{strategy} (OOS)', linewidth=2)
        
        ax.set_xlabel('Walk-Forward Split')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Walk-Forward Optimization Results')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'walk_forward_performance.png', dpi=300)
        plt.close()
    
    def _create_parameter_importance(self, results: Dict, output_dir: Path) -> None:
        """Analyze parameter importance across strategies."""
        # Placeholder for more sophisticated analysis
        pass
    
    def _generate_text_report(self, results: Dict, output_dir: Path) -> None:
        """Generate detailed text report."""
        report_lines = [
            "# CRYPTO MOMENTUM PARAMETER OPTIMIZATION REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            ""
        ]
        
        for strategy, strategy_results in results.items():
            report_lines.extend([
                f"\n## {strategy.upper()} STRATEGY",
                "-" * 40
            ])
            
            # Find most stable parameters
            most_stable = max(strategy_results, key=lambda x: x.stability_score)
            
            report_lines.extend([
                f"\n### Most Stable Parameters:",
                f"Parameters: {most_stable.params}",
                f"In-Sample Sharpe: {most_stable.in_sample_performance:.3f}",
                f"Out-Sample Sharpe: {most_stable.out_sample_performance:.3f}",
                f"Stability Score: {most_stable.stability_score:.3f}",
                ""
            ])
            
            # Average performance
            avg_is = np.mean([r.in_sample_performance for r in strategy_results])
            avg_oos = np.mean([r.out_sample_performance for r in strategy_results])
            
            report_lines.extend([
                f"### Average Performance:",
                f"Average IS Sharpe: {avg_is:.3f}",
                f"Average OOS Sharpe: {avg_oos:.3f}",
                f"Performance Degradation: {(avg_is - avg_oos) / avg_is * 100:.1f}%",
                ""
            ])
        
        # Write report
        with open(output_dir / 'optimization_report.txt', 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Report saved to {output_dir / 'optimization_report.txt'}")
    
    def run_comprehensive_optimization(self) -> Dict[str, List[OptimizationResult]]:
        """Run comprehensive optimization for all strategies."""
        print("🚀 ADVANCED PARAMETER OPTIMIZATION")
        print("=" * 60)
        
        # Load data
        data = self.load_data()
        print(f"Loaded {len(data)} days of data")
        
        # Define parameter grids for each strategy
        param_grids = {
            'momentum': {
                'ewm_span': [10, 15, 20, 30],
                'threshold': [0.005, 0.01, 0.015, 0.02, 0.03]
            },
            'mean_return_ewm': {
                'ewm_span': [3, 5, 7, 10, 15],
                'threshold': [0.001, 0.002, 0.003, 0.005, 0.007, 0.01]
            },
            'mean_return_simple': {
                'window': [2, 3, 5, 7, 10],
                'threshold': [0.002, 0.003, 0.005, 0.007, 0.01, 0.015]
            },
            'rsi': {
                'period': [9, 14, 21],
                'oversold': [20, 25, 30],
                'overbought': [70, 75, 80]
            }
        }
        
        all_results = {}
        
        # Run walk-forward optimization for each strategy
        for strategy, param_grid in param_grids.items():
            print(f"\n📊 Optimizing {strategy} strategy...")
            
            results = self.walk_forward_optimization(
                data=data,
                strategy=strategy,
                param_grid=param_grid,
                n_splits=5,
                test_pct=0.2
            )
            
            all_results[strategy] = results
        
        # Create comprehensive report
        self.create_optimization_report(all_results)
        
        # Find overall best configuration
        best_overall = None
        best_stability = -np.inf
        
        for strategy, results in all_results.items():
            for result in results:
                if result.stability_score > best_stability:
                    best_stability = result.stability_score
                    best_overall = (strategy, result)
        
        if best_overall:
            strategy, result = best_overall
            print(f"\n🏆 BEST OVERALL CONFIGURATION:")
            print(f"Strategy: {strategy}")
            print(f"Parameters: {result.params}")
            print(f"Stability Score: {result.stability_score:.3f}")
            print(f"Expected Sharpe: {result.out_sample_performance:.3f}")
        
        # Generate optimized config file
        self._generate_optimized_config(all_results)
        
        return all_results
    
    def _generate_optimized_config(self, results: Dict[str, List[OptimizationResult]]) -> None:
        """Generate optimized configuration file based on results."""
        # Find best parameters for each strategy
        best_params = {}
        
        for strategy, strategy_results in results.items():
            # Use most stable parameters
            most_stable = max(strategy_results, key=lambda x: x.stability_score)
            best_params[strategy] = most_stable.params
        
        # Create config structure
        config = {
            'signals': {
                # Momentum parameters
                'momentum_threshold': best_params['momentum']['threshold'],
                'momentum_ewma_span': best_params['momentum']['ewm_span'],
                
                # Mean return EWM parameters
                'mean_return_ewm_threshold': best_params['mean_return_ewm']['threshold'],
                'mean_return_ewm_span': best_params['mean_return_ewm']['ewm_span'],
                
                # Mean return simple parameters
                'mean_return_simple_threshold': best_params['mean_return_simple']['threshold'],
                'mean_return_simple_window': best_params['mean_return_simple']['window'],
                
                # RSI parameters
                'rsi_period': best_params['rsi']['period'],
                'rsi_oversold': best_params['rsi']['oversold'],
                'rsi_overbought': best_params['rsi']['overbought'],
                
                # Default strategy selection
                'signal_strategy': 'mean_return_ewm'  # Based on typical crypto performance
            },
            # Add other config sections with defaults
            'portfolio': {
                'max_position_size': 0.10,
                'rebalance_frequency': 'weekly'
            },
            'risk': {
                'max_drawdown': 0.25
            }
        }
        
        # Save config
        with open('config_optimized_advanced.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"\n💾 Saved optimized configuration to config_optimized_advanced.yaml")


def main():
    """Run advanced optimization."""
    optimizer = AdvancedParameterOptimizer()
    results = optimizer.run_comprehensive_optimization()
    
    print("\n✅ OPTIMIZATION COMPLETE!")
    print("Check the output directory for detailed reports and visualizations.")
    print("\nNext steps:")
    print("1. Review optimization_report.txt for detailed analysis")
    print("2. Use config_optimized_advanced.yaml for production")
    print("3. Monitor parameter stability in live trading")


if __name__ == "__main__":
    main()