# crypto_momentum_backtest/portfolio/rebalancer.py
"""Portfolio rebalancing logic."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging


class Rebalancer:
    """
    Handles portfolio rebalancing decisions and execution.
    """
    
    def __init__(
        self,
        rebalance_frequency: str = 'monthly',
        deviation_threshold: float = 0.10,
        min_trade_size: float = 0.01,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize rebalancer.
        
        Args:
            rebalance_frequency: Rebalancing frequency
            deviation_threshold: Threshold for deviation-triggered rebalancing
            min_trade_size: Minimum trade size as fraction of portfolio
            logger: Logger instance
        """
        self.rebalance_frequency = rebalance_frequency
        self.deviation_threshold = deviation_threshold
        self.min_trade_size = min_trade_size
        self.logger = logger or logging.getLogger(__name__)
        
        # Map frequency to pandas offset
        self.frequency_map = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'ME',
            'quarterly': 'QE'
        }
        
    def get_rebalance_dates(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[datetime]:
        """
        Get scheduled rebalance dates.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of rebalance dates
        """
        freq = self.frequency_map.get(self.rebalance_frequency, 'ME')
        dates = pd.date_range(start_date, end_date, freq=freq)
        
        return dates.tolist()
    
    def check_deviation_trigger(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series
    ) -> bool:
        """
        Check if deviation from target triggers rebalancing.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            
        Returns:
            True if rebalancing should be triggered
        """
        # Calculate absolute deviations
        deviations = abs(current_weights - target_weights)
        
        # Check if any position deviates more than threshold
        max_deviation = deviations.max()
        
        if max_deviation > self.deviation_threshold:
            self.logger.info(
                f"Deviation trigger: max deviation {max_deviation:.2%} > "
                f"threshold {self.deviation_threshold:.2%}"
            )
            return True
        
        return False
    
    def calculate_trades(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        current_values: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate required trades to rebalance.
        
        Args:
            current_weights: Current weights
            target_weights: Target weights
            current_values: Current position values
            
        Returns:
            DataFrame with trade details
        """
        total_value = current_values.sum()
        
        # Calculate target values
        target_values = target_weights * total_value
        
        # Calculate required trades
        trade_values = target_values - current_values
        
        # Apply minimum trade size filter
        min_trade_value = total_value * self.min_trade_size
        trade_values[abs(trade_values) < min_trade_value] = 0
        
        # Create trade DataFrame
        trades = pd.DataFrame({
            'symbol': trade_values.index,
            'current_weight': current_weights,
            'target_weight': target_weights,
            'current_value': current_values,
            'target_value': target_values,
            'trade_value': trade_values,
            'trade_pct': trade_values / total_value
        })
        
        # Remove zero trades
        trades = trades[trades['trade_value'] != 0]
        
        return trades
    
    def apply_constraints(
        self,
        trades: pd.DataFrame,
        constraints: Dict
    ) -> pd.DataFrame:
        """
        Apply trading constraints to rebalancing trades.
        
        Args:
            trades: Proposed trades
            constraints: Trading constraints
            
        Returns:
            Adjusted trades
        """
        adjusted_trades = trades.copy()
        
        # Maximum trade size constraint
        if 'max_trade_size' in constraints:
            max_size = constraints['max_trade_size']
            mask = adjusted_trades['trade_pct'].abs() > max_size
            adjusted_trades.loc[mask, 'trade_value'] *= (
                max_size / adjusted_trades.loc[mask, 'trade_pct'].abs()
            )
        
        # Sector constraints
        if 'sector_limits' in constraints:
            # This would require sector mapping
            pass
        
        # Liquidity constraints
        if 'min_liquidity' in constraints:
            # Filter out illiquid assets
            pass
        
        return adjusted_trades
    
    def optimize_trade_execution(
        self,
        trades: pd.DataFrame,
        market_impact_model: Optional[callable] = None
    ) -> List[Dict]:
        """
        Optimize trade execution order.
        
        Args:
            trades: Trades to execute
            market_impact_model: Function to estimate market impact
            
        Returns:
            Ordered list of trades
        """
        # Sort by absolute trade size (largest first)
        trades_sorted = trades.sort_values('trade_value', key=abs, ascending=False)
        
        execution_plan = []
        
        for idx, trade in trades_sorted.iterrows():
            execution = {
                'symbol': trade['symbol'],
                'side': 'buy' if trade['trade_value'] > 0 else 'sell',
                'value': abs(trade['trade_value']),
                'urgency': 'normal'
            }
            
            # Estimate market impact if model provided
            if market_impact_model:
                impact = market_impact_model(
                    symbol=trade['symbol'],
                    trade_value=trade['trade_value']
                )
                execution['expected_impact'] = impact
            
            execution_plan.append(execution)
        
        return execution_plan
    
    def calculate_turnover(
        self,
        trades: pd.DataFrame,
        total_value: float
    ) -> Dict[str, float]:
        """
        Calculate portfolio turnover metrics.
        
        Args:
            trades: Executed trades
            total_value: Total portfolio value
            
        Returns:
            Turnover metrics
        """
        # One-way turnover
        total_traded = trades['trade_value'].abs().sum()
        one_way_turnover = total_traded / (2 * total_value)
        
        # Two-way turnover
        two_way_turnover = total_traded / total_value
        
        # Number of trades
        num_trades = len(trades)
        
        return {
            'one_way_turnover': one_way_turnover,
            'two_way_turnover': two_way_turnover,
            'num_trades': num_trades,
            'avg_trade_size': total_traded / num_trades if num_trades > 0 else 0
        }
