# crypto_momentum_backtest/costs/slippage_model.py
"""Advanced slippage modeling for realistic execution simulation."""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging


class SlippageModel:
    """
    Models slippage based on order size, market conditions, and microstructure.
    """
    
    def __init__(
        self,
        base_spread_bps: float = 1.0,  # basis points
        temporary_impact_coef: float = 0.1,
        permanent_impact_coef: float = 0.05,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize slippage model.
        
        Args:
            base_spread_bps: Base bid-ask spread in basis points
            temporary_impact_coef: Temporary market impact coefficient
            permanent_impact_coef: Permanent market impact coefficient
            logger: Logger instance
        """
        self.base_spread_bps = base_spread_bps
        self.temporary_impact_coef = temporary_impact_coef
        self.permanent_impact_coef = permanent_impact_coef
        self.logger = logger or logging.getLogger(__name__)
        
    def calculate_spread(
        self,
        symbol: str,
        volatility: float,
        volume: float,
        time_of_day: Optional[int] = None
    ) -> float:
        """
        Calculate dynamic bid-ask spread.
        
        Args:
            symbol: Trading symbol
            volatility: Current volatility
            volume: Current volume
            time_of_day: Hour of day (0-23)
            
        Returns:
            Spread in basis points
        """
        # Base spread
        spread = self.base_spread_bps
        
        # Volatility adjustment (higher vol = wider spread)
        vol_adjustment = volatility / 0.15  # Normalized to 15% annual vol
        spread *= vol_adjustment
        
        # Volume adjustment (lower volume = wider spread)
        volume_normalized = volume / 1e9  # Normalize to $1B
        volume_factor = 1 / (1 + volume_normalized)
        spread *= (1 + volume_factor)
        
        # Time of day adjustment
        if time_of_day is not None:
            # Wider spreads during off-hours
            if time_of_day < 8 or time_of_day > 20:
                spread *= 1.5
        
        return spread
    
    def calculate_market_impact(
        self,
        order_size: float,
        adv: float,  # Average Daily Volume
        volatility: float,
        duration_minutes: int = 30
    ) -> Tuple[float, float]:
        """
        Calculate market impact using Almgren-Chriss model.
        
        Args:
            order_size: Order size in currency
            adv: Average daily volume
            volatility: Daily volatility
            duration_minutes: Order duration
            
        Returns:
            Tuple of (temporary_impact_bps, permanent_impact_bps)
        """
        # Participation rate
        daily_minutes = 24 * 60
        participation_rate = order_size / (adv * duration_minutes / daily_minutes)
        
        # Temporary impact (square root model)
        temp_impact = self.temporary_impact_coef * np.sqrt(participation_rate) * volatility
        
        # Permanent impact (linear model)
        perm_impact = self.permanent_impact_coef * participation_rate
        
        # Convert to basis points
        temp_impact_bps = temp_impact * 10000
        perm_impact_bps = perm_impact * 10000
        
        return temp_impact_bps, perm_impact_bps
    
    def simulate_execution(
        self,
        order_size: float,
        side: str,
        mid_price: float,
        spread_bps: float,
        impact_bps: Tuple[float, float]
    ) -> Dict[str, float]:
        """
        Simulate order execution with slippage.
        
        Args:
            order_size: Order size
            side: 'buy' or 'sell'
            mid_price: Mid market price
            spread_bps: Spread in basis points
            impact_bps: (temporary, permanent) impact in bps
            
        Returns:
            Execution details
        """
        temp_impact_bps, perm_impact_bps = impact_bps
        
        # Calculate execution price
        if side == 'buy':
            # Pay half spread + impacts
            total_slippage_bps = spread_bps / 2 + temp_impact_bps + perm_impact_bps
            execution_price = mid_price * (1 + total_slippage_bps / 10000)
        else:
            # Receive half spread - impacts
            total_slippage_bps = spread_bps / 2 + temp_impact_bps + perm_impact_bps
            execution_price = mid_price * (1 - total_slippage_bps / 10000)
        
        # Calculate costs
        executed_value = order_size * execution_price
        ideal_value = order_size * mid_price
        slippage_cost = abs(executed_value - ideal_value)
        
        return {
            'mid_price': mid_price,
            'execution_price': execution_price,
            'spread_bps': spread_bps,
            'temporary_impact_bps': temp_impact_bps,
            'permanent_impact_bps': perm_impact_bps,
            'total_slippage_bps': total_slippage_bps,
            'slippage_cost': slippage_cost,
            'executed_value': executed_value
        }
    
    def estimate_portfolio_slippage(
        self,
        trades: pd.DataFrame,
        market_data: pd.DataFrame,
        execution_style: str = 'normal'
    ) -> pd.DataFrame:
        """
        Estimate slippage for a portfolio of trades.
        
        Args:
            trades: DataFrame with trade details
            market_data: Market data (prices, volumes, volatility)
            execution_style: 'aggressive', 'normal', 'passive'
            
        Returns:
            DataFrame with slippage estimates
        """
        slippage_results = []
        
        # Execution style parameters
        style_params = {
            'aggressive': {'duration': 15, 'spread_cross': 1.0},
            'normal': {'duration': 30, 'spread_cross': 0.5},
            'passive': {'duration': 60, 'spread_cross': 0.0}
        }
        
        params = style_params.get(execution_style, style_params['normal'])
        
        for idx, trade in trades.iterrows():
            symbol = trade['symbol']
            
            # Get market data
            if symbol in market_data.columns:
                volatility = market_data[f'{symbol}_volatility'].iloc[-1]
                volume = market_data[f'{symbol}_volume'].iloc[-1]
                price = market_data[symbol].iloc[-1]
            else:
                # Default values
                volatility = 0.20
                volume = 1e8
                price = trade.get('price', 100)
            
            # Calculate spread
            spread_bps = self.calculate_spread(symbol, volatility, volume)
            
            # Calculate market impact
            order_value = abs(trade['value'])
            impact_bps = self.calculate_market_impact(
                order_value,
                volume,
                volatility,
                params['duration']
            )
            
            # Simulate execution
            execution = self.simulate_execution(
                order_size=abs(trade['units']),
                side=trade['side'],
                mid_price=price,
                spread_bps=spread_bps * params['spread_cross'],
                impact_bps=impact_bps
            )
            
            execution['symbol'] = symbol
            execution['trade_id'] = idx
            
            slippage_results.append(execution)
        
        return pd.DataFrame(slippage_results)
