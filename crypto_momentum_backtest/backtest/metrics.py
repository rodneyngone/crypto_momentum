"""Performance metrics calculation for backtesting - Updated without empyrical dependency."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats


class MetricsCalculator:
    """
    Calculates comprehensive performance metrics for backtest results.
    Implements all metrics without empyrical dependency.
    """
    
    @staticmethod
    def calculate_returns_metrics(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> Dict[str, float]:
        """
        Calculate return-based metrics.
        
        Args:
            returns: Returns series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            
        Returns:
            Dictionary of metrics
        """
        # Adjust risk-free rate to period
        period_risk_free = risk_free_rate / periods_per_year
        
        # Basic calculations
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(periods_per_year)
        
        # Downside volatility
        downside_returns = returns[returns < period_risk_free]
        downside_vol = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
        
        # Max drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = drawdown.min()
        
        # Sharpe ratio
        excess_returns = returns - period_risk_free
        sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Sortino ratio
        sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        # Calmar ratio
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        # Win rate and profit factor
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        win_rate = len(winning_returns) / len(returns) if len(returns) > 0 else 0
        
        profit_factor = winning_returns.sum() / abs(losing_returns.sum()) if len(losing_returns) > 0 and losing_returns.sum() != 0 else np.inf
        
        # Tail ratio
        percentile_95 = np.percentile(returns, 95)
        percentile_5 = np.percentile(returns, 5)
        tail_ratio = abs(percentile_95 / percentile_5) if percentile_5 != 0 else np.inf
        
        # Gain to pain ratio
        total_gains = winning_returns.sum()
        total_losses = abs(losing_returns.sum())
        gain_to_pain = total_gains / total_losses if total_losses > 0 else np.inf
        
        metrics = {
            # Basic returns
            'total_return': total_return,
            'annual_return': annual_return,
            'monthly_return': ((1 + annual_return) ** (1/12) - 1),
            
            # Risk metrics
            'annual_volatility': annual_vol,
            'downside_volatility': downside_vol,
            'max_drawdown': max_dd,
            
            # Risk-adjusted returns
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            
            # Other metrics
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'tail_ratio': tail_ratio,
            'gain_to_pain_ratio': gain_to_pain
        }
        
        # Add distribution metrics
        metrics.update({
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'var_95': np.percentile(returns, 5),
            'cvar_95': returns[returns <= np.percentile(returns, 5)].mean() if len(returns[returns <= np.percentile(returns, 5)]) > 0 else np.percentile(returns, 5)
        })
        
        return metrics
    
    @staticmethod
    def calculate_drawdown_metrics(returns: pd.Series) -> Dict[str, any]:
        """
        Calculate detailed drawdown metrics.
        
        Args:
            returns: Returns series
            
        Returns:
            Dictionary of drawdown metrics
        """
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Calculate drawdown series
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        
        # Find drawdown periods
        is_drawdown = drawdown < 0
        drawdown_starts = (~is_drawdown).shift(1) & is_drawdown
        drawdown_ends = is_drawdown.shift(1) & (~is_drawdown)
        
        # Calculate metrics
        metrics = {
            'max_drawdown': drawdown.min(),
            'max_drawdown_duration': 0,
            'avg_drawdown': drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0,
            'avg_drawdown_duration': 0,
            'drawdown_periods': []
        }
        
        # Analyze each drawdown period
        start_dates = drawdown.index[drawdown_starts]
        end_dates = drawdown.index[drawdown_ends]
        
        if len(start_dates) > 0 and len(end_dates) > 0:
            # Handle edge cases
            if start_dates[0] > end_dates[0]:
                end_dates = end_dates[1:]
            if len(start_dates) > len(end_dates):
                start_dates = start_dates[:-1]
            
            durations = []
            
            for start, end in zip(start_dates, end_dates):
                duration = (end - start).days
                durations.append(duration)
                
                period_drawdown = drawdown[start:end].min()
                recovery_date = cum_returns[end:].loc[
                    cum_returns[end:] >= running_max[start]
                ].index
                
                recovery_days = (recovery_date[0] - end).days if len(recovery_date) > 0 else None
                
                metrics['drawdown_periods'].append({
                    'start': start,
                    'end': end,
                    'drawdown': period_drawdown,
                    'duration_days': duration,
                    'recovery_days': recovery_days
                })
            
            metrics['max_drawdown_duration'] = max(durations) if durations else 0
            metrics['avg_drawdown_duration'] = np.mean(durations) if durations else 0
        
        return metrics
    
    @staticmethod
    def calculate_trade_metrics(trades: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate trade-based metrics.
        
        Args:
            trades: DataFrame with trade data
            
        Returns:
            Dictionary of trade metrics
        """
        if trades.empty:
            return {
                'total_trades': 0,
                'avg_trade_size': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'largest_win': 0,
                'largest_loss': 0
            }
        
        metrics = {
            'total_trades': len(trades),
            'avg_trade_size': trades['value'].mean(),
            'total_volume': trades['value'].sum(),
            'avg_trades_per_day': len(trades) / trades['date'].nunique(),
            'unique_symbols_traded': trades['symbol'].nunique()
        }
        
        # Trade frequency by symbol
        symbol_counts = trades['symbol'].value_counts()
        metrics['most_traded_symbol'] = symbol_counts.index[0]
        metrics['most_traded_count'] = symbol_counts.iloc[0]
        
        return metrics
    
    @staticmethod
    def calculate_risk_metrics(
        returns: pd.Series,
        positions: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate risk-related metrics.
        
        Args:
            returns: Portfolio returns
            positions: Position data
            
        Returns:
            Dictionary of risk metrics
        """
        metrics = {}
        
        # Portfolio concentration
        if not positions.empty:
            position_values = positions.abs()
            total_exposure = position_values.sum(axis=1)
            
            # Herfindahl index (concentration)
            position_weights = position_values.div(total_exposure, axis=0)
            herfindahl = (position_weights ** 2).sum(axis=1).mean()
            
            metrics['avg_herfindahl_index'] = herfindahl
            metrics['avg_positions'] = (position_values > 0).sum(axis=1).mean()
            metrics['max_positions'] = (position_values > 0).sum(axis=1).max()
            metrics['avg_exposure'] = total_exposure.mean()
        
        # Risk metrics
        metrics['information_ratio'] = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Omega ratio
        threshold = 0
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        omega = gains.sum() / losses.sum() if losses.sum() > 0 else np.inf
        metrics['omega_ratio'] = omega
        
        # Stability metrics
        rolling_sharpe = returns.rolling(window=252).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        
        metrics['sharpe_stability'] = rolling_sharpe.std() if len(rolling_sharpe) > 0 else 0
        
        # Calculate rolling max drawdown
        def calc_max_dd(returns_window):
            cum = (1 + returns_window).cumprod()
            running_max = cum.expanding().max()
            dd = (cum - running_max) / running_max
            return dd.min()
        
        rolling_max_dd = returns.rolling(window=252).apply(calc_max_dd)
        metrics['rolling_max_dd'] = rolling_max_dd.min() if len(rolling_max_dd) > 0 else 0
        
        return metrics
    
    @staticmethod
    def calculate_benchmark_metrics(
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate metrics relative to benchmark.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Dictionary of relative metrics
        """
        # Align series
        aligned_returns, aligned_benchmark = returns.align(
            benchmark_returns, join='inner'
        )
        
        # Calculate beta and alpha
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = aligned_benchmark.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Alpha (annualized)
        strategy_annual = aligned_returns.mean() * 252
        benchmark_annual = aligned_benchmark.mean() * 252
        alpha = strategy_annual - beta * benchmark_annual
        
        # Tracking error
        excess_returns = aligned_returns - aligned_benchmark
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        # Information ratio
        info_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Capture ratios
        up_periods = aligned_benchmark > 0
        down_periods = aligned_benchmark < 0
        
        up_capture = (aligned_returns[up_periods].mean() / aligned_benchmark[up_periods].mean() 
                     if up_periods.any() and aligned_benchmark[up_periods].mean() != 0 else 0)
        
        down_capture = (aligned_returns[down_periods].mean() / aligned_benchmark[down_periods].mean() 
                       if down_periods.any() and aligned_benchmark[down_periods].mean() != 0 else 0)
        
        capture_ratio = up_capture / down_capture if down_capture != 0 else np.inf
        
        metrics = {
            'alpha': alpha,
            'beta': beta,
            'correlation': aligned_returns.corr(aligned_benchmark),
            'tracking_error': tracking_error,
            'information_ratio': info_ratio,
            'up_capture': up_capture,
            'down_capture': down_capture,
            'capture_ratio': capture_ratio
        }
        
        return metrics
    
    @classmethod
    def calculate_all_metrics(
        cls,
        portfolio_values: pd.Series,
        returns: pd.Series,
        positions: pd.DataFrame,
        trades: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02
    ) -> Dict[str, any]:
        """
        Calculate all metrics.
        
        Args:
            portfolio_values: Portfolio value series
            returns: Returns series
            positions: Positions DataFrame
            trades: Trades DataFrame
            benchmark_returns: Optional benchmark returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Comprehensive metrics dictionary
        """
        all_metrics = {}
        
        # Returns metrics
        all_metrics['returns'] = cls.calculate_returns_metrics(
            returns, risk_free_rate
        )
        
        # Drawdown metrics
        all_metrics['drawdowns'] = cls.calculate_drawdown_metrics(returns)
        
        # Trade metrics
        all_metrics['trades'] = cls.calculate_trade_metrics(trades)
        
        # Risk metrics
        all_metrics['risk'] = cls.calculate_risk_metrics(returns, positions)
        
        # Benchmark metrics
        if benchmark_returns is not None:
            all_metrics['benchmark'] = cls.calculate_benchmark_metrics(
                returns, benchmark_returns
            )
        
        # Summary statistics
        all_metrics['summary'] = {
            'start_value': portfolio_values.iloc[0],
            'end_value': portfolio_values.iloc[-1],
            'total_return': (portfolio_values.iloc[-1] - portfolio_values.iloc[0]) / portfolio_values.iloc[0],
            'best_month': returns.resample('ME').apply(lambda x: (1 + x).prod() - 1).max(),
            'worst_month': returns.resample('ME').apply(lambda x: (1 + x).prod() - 1).min(),
            'positive_months': (returns.resample('ME').apply(lambda x: (1 + x).prod() - 1) > 0).sum(),
            'total_months': len(returns.resample('ME').apply(lambda x: (1 + x).prod() - 1))
        }
        
        return all_metrics