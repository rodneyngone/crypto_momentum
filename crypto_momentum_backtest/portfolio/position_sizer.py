"""Position sizing based on ATR and risk management."""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
import logging


class PositionSizer:
    """
    ATR-based position sizing with volatility adjustments.
    """
    
    def __init__(
        self,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        max_position_size: float = 0.20,
        volatility_lookback: int = 10,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize position sizer.
        
        Args:
            atr_period: ATR calculation period
            atr_multiplier: Risk per position in ATR units
            max_position_size: Maximum position size
            volatility_lookback: Lookback for volatility calculation
            logger: Logger instance
        """
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.max_position_size = max_position_size
        self.volatility_lookback = volatility_lookback
        self.logger = logger or logging.getLogger(__name__)
        
    def calculate_position_sizes(
        self,
        prices: Union[pd.DataFrame, pd.Series],
        atr: Union[pd.DataFrame, pd.Series],
        signals: Union[pd.DataFrame, pd.Series],
        capital: float = 1000000,
        risk_per_position: float = 0.02
    ) -> pd.DataFrame:
        """
        Calculate position sizes based on ATR.
        
        Args:
            prices: Price DataFrame or Series
            atr: ATR DataFrame or Series
            signals: Signal DataFrame or Series (1, -1, 0)
            capital: Total capital
            risk_per_position: Risk per position as fraction of capital
            
        Returns:
            Position sizes in units
        """
        # Handle both Series and DataFrame inputs
        if isinstance(prices, pd.Series):
            # Convert Series to DataFrame for consistent handling
            prices_df = prices.to_frame().T
            atr_df = atr.to_frame().T if isinstance(atr, pd.Series) else atr
            signals_df = signals.to_frame().T if isinstance(signals, pd.Series) else signals
            
            # Get columns
            columns = prices.index if isinstance(prices, pd.Series) else prices.columns
        else:
            prices_df = prices
            atr_df = atr
            signals_df = signals
            columns = prices.columns
        
        position_sizes = pd.DataFrame(
            0.0,
            index=prices_df.index,
            columns=columns
        )
        
        for symbol in columns:
            if symbol not in atr_df.columns or symbol not in signals_df.columns:
                continue
            
            # Get values for this symbol
            price_val = prices_df[symbol].iloc[0] if len(prices_df) > 0 else prices_df[symbol]
            atr_val = atr_df[symbol].iloc[0] if len(atr_df) > 0 else atr_df[symbol]
            signal_val = signals_df[symbol].iloc[0] if len(signals_df) > 0 else signals_df[symbol]
            
            # Calculate position size based on ATR
            # Position size = (Capital * Risk%) / (ATR * Multiplier)
            risk_amount = capital * risk_per_position
            
            if atr_val > 0 and price_val > 0:
                position_value = risk_amount / (atr_val * self.atr_multiplier)
                
                # Convert to units
                position_units = position_value / price_val
                
                # Apply maximum position size constraint
                max_units = (capital * self.max_position_size) / price_val
                position_units = min(position_units, max_units)
                
                # Apply signal
                position_sizes.iloc[0, position_sizes.columns.get_loc(symbol)] = position_units * signal_val
        
        # If input was a Series, return a Series
        if isinstance(prices, pd.Series):
            return position_sizes.iloc[0]
        
        return position_sizes
    
    def calculate_volatility_adjusted_sizes(
        self,
        base_sizes: pd.DataFrame,
        returns: pd.DataFrame,
        target_volatility: float = 0.15
    ) -> pd.DataFrame:
        """
        Adjust position sizes based on realized volatility.
        
        Args:
            base_sizes: Base position sizes
            returns: Returns DataFrame
            target_volatility: Target portfolio volatility
            
        Returns:
            Adjusted position sizes
        """
        adjusted_sizes = base_sizes.copy()
        
        # Calculate rolling volatility for each asset
        asset_vols = returns.rolling(
            window=self.volatility_lookback,
            min_periods=self.volatility_lookback
        ).std() * np.sqrt(252)  # Annualized
        
        # Calculate volatility scaling factor
        vol_scale = target_volatility / asset_vols
        vol_scale = vol_scale.clip(lower=0.5, upper=2.0)  # Limit scaling
        
        # Apply scaling
        for symbol in base_sizes.columns:
            if symbol in vol_scale.columns:
                adjusted_sizes[symbol] = base_sizes[symbol] * vol_scale[symbol]
        
        return adjusted_sizes
    
    def apply_correlation_adjustment(
        self,
        position_sizes: pd.DataFrame,
        returns: pd.DataFrame,
        lookback: int = 60,
        max_correlation: float = 0.8
    ) -> pd.DataFrame:
        """
        Adjust sizes based on correlation between positions.
        
        Args:
            position_sizes: Current position sizes
            returns: Returns DataFrame
            lookback: Correlation lookback period
            max_correlation: Maximum allowed correlation
            
        Returns:
            Adjusted position sizes
        """
        adjusted_sizes = position_sizes.copy()
        
        # Calculate correlation matrix
        corr_matrix = returns.iloc[-lookback:].corr()
        
        # For each date
        for idx in position_sizes.index:
            active_positions = position_sizes.columns[
                position_sizes.loc[idx].abs() > 0
            ]
            
            if len(active_positions) <= 1:
                continue
            
            # Check correlations between active positions
            for i, sym1 in enumerate(active_positions):
                for sym2 in active_positions[i+1:]:
                    if corr_matrix.loc[sym1, sym2] > max_correlation:
                        # Reduce position sizes proportionally
                        reduction = (corr_matrix.loc[sym1, sym2] - max_correlation) / (1 - max_correlation)
                        reduction = min(reduction, 0.5)  # Max 50% reduction
                        
                        adjusted_sizes.loc[idx, sym1] *= (1 - reduction)
                        adjusted_sizes.loc[idx, sym2] *= (1 - reduction)
        
        return adjusted_sizes
    
    def calculate_kelly_sizes(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        lookback: int = 252,
        kelly_fraction: float = 0.25
    ) -> pd.DataFrame:
        """
        Calculate position sizes using Kelly Criterion.
        
        Args:
            signals: Trading signals
            returns: Historical returns
            lookback: Lookback period for win rate calculation
            kelly_fraction: Fraction of Kelly to use (for safety)
            
        Returns:
            Kelly-based position sizes
        """
        kelly_sizes = pd.DataFrame(
            0.0,
            index=signals.index,
            columns=signals.columns
        )
        
        for symbol in signals.columns:
            if symbol not in returns.columns:
                continue
            
            # Calculate historical statistics when signal was active
            signal_returns = returns[symbol].copy()
            signal_returns[signals[symbol] == 0] = np.nan
            
            # Calculate win rate and average win/loss
            winning_returns = signal_returns[signal_returns > 0].dropna()
            losing_returns = signal_returns[signal_returns < 0].dropna()
            
            if len(winning_returns) > 10 and len(losing_returns) > 10:
                win_rate = len(winning_returns) / (len(winning_returns) + len(losing_returns))
                avg_win = winning_returns.mean()
                avg_loss = abs(losing_returns.mean())
                
                # Kelly formula: f = (p*b - q) / b
                # where p = win rate, q = loss rate, b = win/loss ratio
                if avg_loss > 0:
                    b = avg_win / avg_loss
                    kelly_pct = (win_rate * b - (1 - win_rate)) / b
                    kelly_pct = max(0, min(kelly_pct, 1))  # Bound between 0 and 1
                    
                    # Apply Kelly fraction for safety
                    kelly_sizes[symbol] = kelly_pct * kelly_fraction * signals[symbol]
                    
        return kelly_sizes
