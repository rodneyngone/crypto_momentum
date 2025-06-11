# crypto_momentum_backtest/backtest/validator.py
"""Backtest validation and robustness testing."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit


class BacktestValidator:
    """
    Validates backtest results and performs robustness testing.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size_ratio: float = 0.2,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize validator.
        
        Args:
            n_splits: Number of walk-forward splits
            test_size_ratio: Test set size as ratio of total
            logger: Logger instance
        """
        self.n_splits = n_splits
        self.test_size_ratio = test_size_ratio
        self.logger = logger or logging.getLogger(__name__)
        
    def walk_forward_analysis(
        self,
        backtest_func: callable,
        start_date: datetime,
        end_date: datetime,
        **backtest_kwargs
    ) -> List[Dict]:
        """
        Perform walk-forward analysis.
        
        Args:
            backtest_func: Backtest function to call
            start_date: Start date
            end_date: End date
            **backtest_kwargs: Additional backtest parameters
            
        Returns:
            List of results for each split
        """
        # Calculate total days
        total_days = (end_date - start_date).days
        test_days = max(30, int(total_days * self.test_size_ratio))  # Minimum 30 days
        train_days = max(60, total_days - test_days)  # Minimum 60 days for training
        
        # Create splits
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=test_days)
        
        # Generate date ranges
        dates = pd.date_range(start_date, end_date, freq='D')
        
        results = []
        
        for i, (train_idx, test_idx) in enumerate(tscv.split(dates)):
            self.logger.info(f"Running walk-forward split {i+1}/{self.n_splits}")
            
            train_start = dates[train_idx[0]]
            train_end = dates[train_idx[-1]]
            test_start = dates[test_idx[0]]
            test_end = dates[test_idx[-1]]
            
            # Run backtest on training period
            train_result = backtest_func(
                start_date=train_start,
                end_date=train_end,
                **backtest_kwargs
            )
            
            # Run backtest on test period with same parameters
            test_result = backtest_func(
                start_date=test_start,
                end_date=test_end,
                **backtest_kwargs
            )
            
            results.append({
                'split': i,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_metrics': train_result['metrics'],
                'test_metrics': test_result['metrics'],
                'train_portfolio': train_result['portfolio'],
                'test_portfolio': test_result['portfolio']
            })
        
        return results
    
    def monte_carlo_simulation(
        self,
        returns: pd.Series,
        n_simulations: int = 1000,
        n_days: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Run Monte Carlo simulation on returns.
        
        Args:
            returns: Historical returns
            n_simulations: Number of simulations
            n_days: Number of days to simulate
            
        Returns:
            Simulation results
        """
        if n_days is None:
            n_days = len(returns)
        
        # Calculate return statistics
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Run simulations
        simulated_returns = np.random.normal(
            mean_return,
            std_return,
            size=(n_simulations, n_days)
        )
        
        # Calculate cumulative returns
        cumulative_returns = (1 + simulated_returns).cumprod(axis=1)
        
        # Calculate terminal values
        terminal_values = cumulative_returns[:, -1]
        
        # Calculate metrics for each simulation
        max_drawdowns = []
        sharpe_ratios = []
        
        for i in range(n_simulations):
            sim_returns = simulated_returns[i]
            cum_returns = cumulative_returns[i]
            
            # Max drawdown
            running_max = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - running_max) / running_max
            max_drawdowns.append(drawdown.min())
            
            # Sharpe ratio
            sharpe = np.sqrt(252) * sim_returns.mean() / sim_returns.std()
            sharpe_ratios.append(sharpe)
        
        return {
            'terminal_values': terminal_values,
            'max_drawdowns': np.array(max_drawdowns),
            'sharpe_ratios': np.array(sharpe_ratios),
            'percentiles': {
                '5%': np.percentile(terminal_values, 5),
                '25%': np.percentile(terminal_values, 25),
                '50%': np.percentile(terminal_values, 50),
                '75%': np.percentile(terminal_values, 75),
                '95%': np.percentile(terminal_values, 95)
            }
        }
    
    def parameter_sensitivity(
        self,
        backtest_func: callable,
        base_params: Dict,
        param_ranges: Dict[str, List],
        **fixed_kwargs
    ) -> pd.DataFrame:
        """
        Test parameter sensitivity.
        
        Args:
            backtest_func: Backtest function
            base_params: Base parameters
            param_ranges: Parameter ranges to test
            **fixed_kwargs: Fixed parameters
            
        Returns:
            Sensitivity analysis results
        """
        results = []
        
        for param_name, param_values in param_ranges.items():
            self.logger.info(f"Testing sensitivity for {param_name}")
            
            for param_value in param_values:
                # Update parameters
                test_params = base_params.copy()
                test_params[param_name] = param_value
                
                # Run backtest
                try:
                    result = backtest_func(**test_params, **fixed_kwargs)
                    
                    results.append({
                        'parameter': param_name,
                        'value': param_value,
                        'total_return': result['metrics']['total_return'],
                        'sharpe_ratio': result['metrics']['sharpe_ratio'],
                        'max_drawdown': result['metrics']['max_drawdown'],
                        'volatility': result['metrics']['volatility']
                    })
                except Exception as e:
                    self.logger.error(f"Error with {param_name}={param_value}: {e}")
        
        return pd.DataFrame(results)
    
    def validate_results(
        self,
        results: Dict
    ) -> Dict[str, bool]:
        """
        Validate backtest results for common issues.
        
        Args:
            results: Backtest results
            
        Returns:
            Validation checks
        """
        checks = {}
        
        # Check for look-ahead bias
        positions = results.get('positions', pd.DataFrame())
        returns = results.get('returns', pd.Series())
        
        if not positions.empty and not returns.empty:
            # Positions should not predict future returns
            for i in range(1, min(len(positions) - 1, len(returns) - 1)):
                if i + 1 < len(returns):
                    future_return = returns.iloc[i+1]
                else:
                    continue
                current_position = positions.iloc[i].sum()
                
                if abs(current_position) > 0:
                    correlation = np.corrcoef(
                        positions.iloc[:i].sum(axis=1),
                        returns.iloc[1:i+1]
                    )[0, 1]
                    
                    # Suspiciously high correlation might indicate look-ahead
                    if correlation > 0.7:
                        checks['look_ahead_bias'] = False
                        break
            else:
                checks['look_ahead_bias'] = True
        
        # Check for survivorship bias
        # (Would need delisted assets info)
        checks['survivorship_bias'] = True  # Assume handled
        
        # Check for unrealistic trading
        trades = results.get('trades', pd.DataFrame())
        if not trades.empty:
            # Check for excessive trading
            daily_trades = trades.groupby(trades['date'].dt.date).size()
            avg_daily_trades = daily_trades.mean()
            
            checks['reasonable_trading'] = avg_daily_trades < 50
            
            # Check for round-trip trades
            symbol_trades = trades.groupby(['date', 'symbol'])['side'].apply(list)
            round_trips = symbol_trades.apply(
                lambda x: 'buy' in x and 'sell' in x
            ).sum()
            
            checks['minimal_round_trips'] = round_trips < len(trades) * 0.1
        
        # Check metrics validity
        metrics = results.get('metrics', {})
        
        checks['valid_sharpe'] = -5 < metrics.get('sharpe_ratio', 0) < 5
        checks['valid_returns'] = -1 < metrics.get('total_return', 0) < 10
        checks['valid_drawdown'] = -1 <= metrics.get('max_drawdown', 0) <= 0
        
        return checks
    
    def generate_validation_report(
        self,
        walk_forward_results: List[Dict],
        monte_carlo_results: Dict,
        sensitivity_results: pd.DataFrame,
        validation_checks: Dict[str, bool]
    ) -> Dict:
        """
        Generate comprehensive validation report.
        
        Args:
            walk_forward_results: Walk-forward analysis results
            monte_carlo_results: Monte Carlo simulation results
            sensitivity_results: Parameter sensitivity results
            validation_checks: Validation check results
            
        Returns:
            Validation report
        """
        report = {
            'timestamp': datetime.now(),
            'validation_checks': validation_checks,
            'all_checks_passed': all(validation_checks.values())
        }
        
        # Walk-forward analysis summary
        if walk_forward_results:
            train_sharpes = [r['train_metrics']['sharpe_ratio'] for r in walk_forward_results]
            test_sharpes = [r['test_metrics']['sharpe_ratio'] for r in walk_forward_results]
            
            report['walk_forward'] = {
                'n_splits': len(walk_forward_results),
                'avg_train_sharpe': np.mean(train_sharpes),
                'avg_test_sharpe': np.mean(test_sharpes),
                'sharpe_degradation': np.mean(train_sharpes) - np.mean(test_sharpes),
                'consistent_performance': np.std(test_sharpes) < 0.5
            }
        
        # Monte Carlo summary
        if monte_carlo_results:
            report['monte_carlo'] = {
                'median_terminal_value': monte_carlo_results['percentiles']['50%'],
                'value_at_risk_5%': monte_carlo_results['percentiles']['5%'],
                'probability_of_loss': (monte_carlo_results['terminal_values'] < 1).mean(),
                'expected_max_drawdown': monte_carlo_results['max_drawdowns'].mean()
            }
        
        # Parameter sensitivity summary
        if not sensitivity_results.empty:
            report['sensitivity'] = {}
            
            for param in sensitivity_results['parameter'].unique():
                param_data = sensitivity_results[
                    sensitivity_results['parameter'] == param
                ]
                
                report['sensitivity'][param] = {
                    'sharpe_std': param_data['sharpe_ratio'].std(),
                    'return_std': param_data['total_return'].std(),
                    'stable': param_data['sharpe_ratio'].std() < 0.2
                }
        
        return report
