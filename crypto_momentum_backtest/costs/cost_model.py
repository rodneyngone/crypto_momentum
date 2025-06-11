# crypto_momentum_backtest/costs/cost_model.py
"""Transaction cost modeling for realistic backtesting."""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
import logging


class CostModel:
    """
    Comprehensive transaction cost model including fees, spreads, and slippage.
    """
    
    def __init__(
        self,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0004,
        base_spread: float = 0.0001,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize cost model.
        
        Args:
            maker_fee: Maker fee rate
            taker_fee: Taker fee rate
            base_spread: Base bid-ask spread
            logger: Logger instance
        """
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.base_spread = base_spread
        self.logger = logger or logging.getLogger(__name__)
        
    def calculate_trading_costs(
        self,
        trade_value: float,
        trade_type: str = 'taker',
        volatility: float = 0.0,
        volume: float = 0.0,
        urgency: str = 'normal'
    ) -> Dict[str, float]:
        """
        Calculate total trading costs.
        
        Args:
            trade_value: Absolute trade value
            trade_type: 'maker' or 'taker'
            volatility: Current volatility
            volume: Current volume
            urgency: Trade urgency ('low', 'normal', 'high')
            
        Returns:
            Dictionary with cost components
        """
        costs = {}
        
        # Exchange fees
        if trade_type == 'maker':
            costs['exchange_fee'] = trade_value * self.maker_fee
        else:
            costs['exchange_fee'] = trade_value * self.taker_fee
        
        # Spread cost
        spread = self.calculate_spread(volatility, volume)
        costs['spread_cost'] = trade_value * spread / 2  # Half spread
        
        # Slippage
        slippage = self.calculate_slippage(trade_value, volume, urgency)
        costs['slippage'] = trade_value * slippage
        
        # Total cost
        costs['total'] = sum(costs.values())
        costs['total_bps'] = costs['total'] / trade_value * 10000
        
        return costs
    
    def calculate_spread(
        self,
        volatility: float,
        volume: float,
        vol_coefficient: float = 0.5,
        volume_coefficient: float = -0.2
    ) -> float:
        """
        Calculate dynamic spread based on market conditions.
        
        Args:
            volatility: Current volatility (annualized)
            volume: Current volume (normalized)
            vol_coefficient: Volatility impact coefficient
            volume_coefficient: Volume impact coefficient
            
        Returns:
            Estimated spread
        """
        # Base spread + volatility adjustment + volume adjustment
        vol_adjustment = volatility * vol_coefficient
        
        # Volume impact (higher volume = tighter spread)
        volume_ratio = volume / (volume + 1)  # Normalize to 0-1
        volume_adjustment = volume_coefficient * (1 - volume_ratio) * self.base_spread
        
        spread = self.base_spread + vol_adjustment + volume_adjustment
        
        # Ensure minimum spread
        return max(spread, self.base_spread * 0.5)
    
    def calculate_slippage(
        self,
        trade_value: float,
        average_daily_volume: float,
        urgency: str = 'normal',
        impact_coefficient: float = 0.1
    ) -> float:
        """
        Calculate market impact/slippage.
        
        Args:
            trade_value: Trade value
            average_daily_volume: Average daily volume
            urgency: Trade urgency
            impact_coefficient: Market impact coefficient
            
        Returns:
            Slippage rate
        """
        # Calculate participation rate
        if average_daily_volume > 0:
            participation_rate = trade_value / average_daily_volume
        else:
            participation_rate = 0.01  # Default for low volume
        
        # Square root market impact model
        base_impact = impact_coefficient * np.sqrt(participation_rate)
        
        # Urgency multiplier
        urgency_multipliers = {
            'low': 0.5,
            'normal': 1.0,
            'high': 2.0,
            'immediate': 3.0
        }
        
        urgency_mult = urgency_multipliers.get(urgency, 1.0)
        
        return base_impact * urgency_mult
    
    def estimate_portfolio_turnover_cost(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        portfolio_value: float,
        market_conditions: Dict[str, pd.Series]
    ) -> Dict[str, float]:
        """
        Estimate costs for portfolio rebalancing.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value
            market_conditions: Dict with volatility and volume data
            
        Returns:
            Cost breakdown
        """
        # Calculate trades
        weight_changes = target_weights - current_weights
        trade_values = abs(weight_changes) * portfolio_value
        
        total_costs = {
            'exchange_fees': 0.0,
            'spread_costs': 0.0,
            'slippage': 0.0,
            'total': 0.0
        }
        
        # Calculate costs for each trade
        for asset in trade_values[trade_values > 0].index:
            trade_value = trade_values[asset]
            
            # Get market conditions
            volatility = market_conditions.get('volatility', pd.Series()).get(asset, 0.15)
            volume = market_conditions.get('volume', pd.Series()).get(asset, 1e6)
            
            # Calculate costs
            costs = self.calculate_trading_costs(
                trade_value=trade_value,
                trade_type='taker',  # Assume taker for conservative estimate
                volatility=volatility,
                volume=volume
            )
            
            # Aggregate
            total_costs['exchange_fees'] += costs['exchange_fee']
            total_costs['spread_costs'] += costs['spread_cost']
            total_costs['slippage'] += costs['slippage']
        
        total_costs['total'] = sum([
            total_costs['exchange_fees'],
            total_costs['spread_costs'],
            total_costs['slippage']
        ])
        
        # Calculate turnover
        total_costs['turnover'] = trade_values.sum() / portfolio_value
        total_costs['cost_per_turnover'] = (
            total_costs['total'] / total_costs['turnover']
            if total_costs['turnover'] > 0 else 0
        )
        
        return total_costs
