# crypto_momentum_backtest/costs/funding_rates.py
"""Funding rate calculations for perpetual futures."""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
import logging
from datetime import datetime, timedelta


class FundingRates:
    """
    Handles funding rate calculations for perpetual futures positions.
    """
    
    def __init__(
        self,
        funding_interval: int = 8,  # hours
        lookback_days: int = 30,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize funding rate calculator.
        
        Args:
            funding_interval: Funding interval in hours
            lookback_days: Days to look back for historical rates
            logger: Logger instance
        """
        self.funding_interval = funding_interval
        self.lookback_days = lookback_days
        self.logger = logger or logging.getLogger(__name__)
        
        # Funding happens 3 times per day for 8-hour intervals
        self.funding_times_per_day = 24 / funding_interval
        
    def load_historical_funding_rates(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.Series:
        """
        Load historical funding rates.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            Series of funding rates
        """
        # In production, this would load from a real data source
        # For now, generate synthetic funding rates
        
        # Generate timestamps at funding intervals
        timestamps = pd.date_range(
            start=start_date,
            end=end_date,
            freq=f'{self.funding_interval}H'
        )
        
        # Generate synthetic funding rates
        # Typical range: -0.01% to 0.01% per 8 hours
        np.random.seed(hash(symbol) % 2**32)
        
        # Base funding rate with mean reversion
        base_rate = 0.0001  # 0.01% positive funding on average
        volatility = 0.0002
        
        # Generate auto-correlated funding rates
        rates = [base_rate]
        for _ in range(len(timestamps) - 1):
            # Mean reversion
            mean_reversion = 0.1 * (base_rate - rates[-1])
            # Random component
            random_component = np.random.normal(0, volatility)
            # New rate
            new_rate = rates[-1] + mean_reversion + random_component
            # Bound rates to realistic range
            new_rate = np.clip(new_rate, -0.003, 0.003)
            rates.append(new_rate)
        
        funding_series = pd.Series(rates, index=timestamps, name=f'{symbol}_funding')
        
        return funding_series
    
    def calculate_funding_cost(
        self,
        positions: pd.DataFrame,
        funding_rates: Dict[str, pd.Series],
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate funding costs for positions.
        
        Args:
            positions: Position sizes (units)
            funding_rates: Dict of symbol -> funding rate series
            prices: Price data
            
        Returns:
            DataFrame with funding costs
        """
        funding_costs = pd.DataFrame(
            0.0,
            index=positions.index,
            columns=positions.columns
        )
        
        for symbol in positions.columns:
            if symbol not in funding_rates:
                continue
            
            # Get funding rate series
            rates = funding_rates[symbol]
            
            # Align funding rates with position index
            aligned_rates = rates.reindex(positions.index, method='ffill')
            
            # Calculate position value
            position_value = positions[symbol] * prices[symbol]
            
            # Funding cost = position_value * funding_rate
            # Positive position pays funding when rate is positive
            # Negative position receives funding when rate is positive
            funding_costs[symbol] = position_value * aligned_rates
        
        return funding_costs
    
    def aggregate_daily_funding(
        self,
        funding_costs: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate funding costs to daily level.
        
        Args:
            funding_costs: Funding costs at funding intervals
            
        Returns:
            Daily funding costs
        """
        # Resample to daily, summing costs
        daily_funding = funding_costs.resample('D').sum()
        
        return daily_funding
    
    def estimate_funding_impact(
        self,
        symbol: str,
        position_side: str,
        holding_period_days: int,
        historical_rates: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Estimate funding impact for a position.
        
        Args:
            symbol: Trading symbol
            position_side: 'long' or 'short'
            holding_period_days: Expected holding period
            historical_rates: Historical funding rates
            
        Returns:
            Funding impact estimates
        """
        if historical_rates is None:
            # Use synthetic rates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            historical_rates = self.load_historical_funding_rates(
                symbol, start_date, end_date
            )
        
        # Calculate statistics
        avg_rate = historical_rates.mean()
        std_rate = historical_rates.std()
        
        # Funding cost depends on position side
        if position_side == 'long':
            # Longs pay when funding is positive
            expected_cost_per_interval = avg_rate
        else:
            # Shorts receive when funding is positive
            expected_cost_per_interval = -avg_rate
        
        # Calculate expected cost for holding period
        intervals_per_day = self.funding_times_per_day
        total_intervals = holding_period_days * intervals_per_day
        
        expected_total_cost = expected_cost_per_interval * total_intervals
        cost_std = std_rate * np.sqrt(total_intervals)
        
        return {
            'expected_cost_pct': expected_total_cost * 100,
            'cost_std_pct': cost_std * 100,
            'worst_case_cost_pct': (expected_total_cost + 2 * cost_std) * 100,
            'avg_funding_rate': avg_rate,
            'funding_volatility': std_rate
        }
    
    def calculate_funding_pnl(
        self,
        positions: pd.DataFrame,
        funding_rates: Dict[str, pd.Series],
        prices: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate P&L from funding rates.
        
        Args:
            positions: Position sizes
            funding_rates: Funding rates by symbol
            prices: Price data
            
        Returns:
            Total funding P&L series
        """
        # Calculate funding costs
        funding_costs = self.calculate_funding_cost(
            positions, funding_rates, prices
        )
        
        # Sum across all positions
        total_funding_pnl = -funding_costs.sum(axis=1)  # Negative because costs reduce P&L
        
        return total_funding_pnl
