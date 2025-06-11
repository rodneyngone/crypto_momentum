"""Enhanced signal generation with mean return options."""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import logging
from .technical_indicators import TechnicalIndicators
from .filters import SignalFilters


class SignalGenerator:
    """
    Generates trading signals based on momentum indicators and mean returns.
    """
    
    def __init__(
        self,
        # Existing parameters
        adx_period: int = 14,
        adx_threshold: float = 20.0,
        ewma_fast: int = 20,
        ewma_slow: int = 50,
        volume_filter_multiple: float = 1.5,
        # New mean return parameters
        use_mean_return_signal: bool = False,
        mean_return_window: int = 30,
        mean_return_type: str = 'ewm',  # 'simple' or 'ewm'
        mean_return_threshold: float = 0.0,  # threshold for signal generation
        mean_return_ewm_span: Optional[int] = None,  # If None, uses mean_return_window
        combine_signals: str = 'override',  # 'override', 'and', 'or'
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize signal generator.
        
        Args:
            adx_period: ADX calculation period
            adx_threshold: ADX threshold for trend strength
            ewma_fast: Fast EWMA period
            ewma_slow: Slow EWMA period
            volume_filter_multiple: Volume filter threshold
            use_mean_return_signal: Whether to use mean return signals
            mean_return_window: Lookback window for mean return calculation
            mean_return_type: 'simple' for arithmetic mean, 'ewm' for exponentially weighted
            mean_return_threshold: Threshold for mean return signals (e.g., 0.02 for 2%)
            mean_return_ewm_span: EWM span (if None, uses mean_return_window)
            combine_signals: How to combine with momentum signals ('override', 'and', 'or')
            logger: Logger instance
        """
        # Existing parameters
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.ewma_fast = ewma_fast
        self.ewma_slow = ewma_slow
        self.volume_filter_multiple = volume_filter_multiple
        
        # New mean return parameters
        self.use_mean_return_signal = use_mean_return_signal
        self.mean_return_window = mean_return_window
        self.mean_return_type = mean_return_type
        self.mean_return_threshold = mean_return_threshold
        self.mean_return_ewm_span = mean_return_ewm_span or mean_return_window
        self.combine_signals = combine_signals
        
        self.logger = logger or logging.getLogger(__name__)
        
        self.indicators = TechnicalIndicators()
        self.filters = SignalFilters()
        
    def calculate_mean_return_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate signals based on mean returns.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with mean return signals added
        """
        df = df.copy()
        
        # Calculate returns
        returns = df['close'].pct_change()
        
        # Calculate mean returns based on type
        if self.mean_return_type == 'simple':
            # Simple moving average of returns
            mean_returns = returns.rolling(window=self.mean_return_window).mean()
            df['mean_return'] = mean_returns
            
        elif self.mean_return_type == 'ewm':
            # Exponentially weighted mean of returns
            mean_returns = returns.ewm(span=self.mean_return_ewm_span, adjust=False).mean()
            df['mean_return'] = mean_returns
            
        else:
            raise ValueError(f"Unknown mean return type: {self.mean_return_type}")
        
        # Calculate rolling standard deviation for adaptive thresholds
        returns_std = returns.rolling(window=self.mean_return_window).std()
        df['returns_std'] = returns_std
        
        # Generate signals based on mean returns
        # Positive mean return above threshold = Long
        # Negative mean return below threshold = Short
        df['mean_return_long'] = (
            (df['mean_return'] > self.mean_return_threshold) & 
            (df['mean_return'] > 0)
        )
        
        df['mean_return_short'] = (
            (df['mean_return'] < -self.mean_return_threshold) & 
            (df['mean_return'] < 0)
        )
        
        # Optional: Use adaptive thresholds based on volatility
        # This helps adjust for different market regimes
        if hasattr(self, 'use_adaptive_threshold') and self.use_adaptive_threshold:
            adaptive_threshold = returns_std * 0.5  # Half standard deviation
            df['mean_return_long'] = (
                (df['mean_return'] > adaptive_threshold) & 
                (df['mean_return'] > 0)
            )
            df['mean_return_short'] = (
                (df['mean_return'] < -adaptive_threshold) & 
                (df['mean_return'] < 0)
            )
        
        # Add signal strength based on how far mean return is from threshold
        df['mean_return_strength'] = abs(df['mean_return']) / returns_std
        
        self.logger.info(
            f"Mean return signals - Long: {df['mean_return_long'].sum()}, "
            f"Short: {df['mean_return_short'].sum()}"
        )
        
        return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all required indicators.
        
        Args:
            df: OHLC DataFrame
            
        Returns:
            DataFrame with indicators added
        """
        df = df.copy()
        
        # Original momentum indicators
        df['adx'] = self.indicators.adx(df, self.adx_period)
        df['ewma_fast'] = self.indicators.ewma(df['close'], self.ewma_fast)
        df['ewma_slow'] = self.indicators.ewma(df['close'], self.ewma_slow)
        df['volume_ratio'] = self.indicators.volume_ratio(df['volume'], 20)
        df['atr'] = self.indicators.atr(df, self.adx_period)
        df['rsi'] = self.indicators.rsi(df['close'], 14)
        
        # Price momentum
        df['returns'] = df['close'].pct_change()
        df['momentum_5d'] = df['close'].pct_change(5)
        df['momentum_20d'] = df['close'].pct_change(20)
        
        # Add mean return signals if enabled
        if self.use_mean_return_signal:
            df = self.calculate_mean_return_signals(df)
        
        return df
    
    def generate_base_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate base trading signals.
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            DataFrame with signal columns
        """
        df = df.copy()
        
        # Original momentum signals
        trend_strong = df['adx'] > self.adx_threshold
        ewma_bullish = df['ewma_fast'] > df['ewma_slow']
        ewma_bearish = df['ewma_fast'] < df['ewma_slow']
        volume_confirmed = df['volume_ratio'] > self.volume_filter_multiple
        
        # Generate momentum signals
        momentum_long = trend_strong & ewma_bullish & volume_confirmed
        momentum_short = trend_strong & ewma_bearish & volume_confirmed
        
        # Determine final signals based on configuration
        if self.use_mean_return_signal:
            if self.combine_signals == 'override':
                # Mean return signals override momentum signals
                df['long_signal'] = df['mean_return_long']
                df['short_signal'] = df['mean_return_short']
                
            elif self.combine_signals == 'and':
                # Both momentum and mean return must agree
                df['long_signal'] = momentum_long & df['mean_return_long']
                df['short_signal'] = momentum_short & df['mean_return_short']
                
            elif self.combine_signals == 'or':
                # Either momentum or mean return can trigger
                df['long_signal'] = momentum_long | df['mean_return_long']
                df['short_signal'] = momentum_short | df['mean_return_short']
                
            else:
                raise ValueError(f"Unknown combine method: {self.combine_signals}")
        else:
            # Use only momentum signals
            df['long_signal'] = momentum_long
            df['short_signal'] = momentum_short
        
        # Ensure no simultaneous long/short signals
        df.loc[df['long_signal'] & df['short_signal'], ['long_signal', 'short_signal']] = False
        
        return df
    
    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply additional filters to improve signal quality.
        
        Args:
            df: DataFrame with base signals
            
        Returns:
            DataFrame with filtered signals
        """
        df = df.copy()
        
        # RSI filter - avoid overbought/oversold
        df['long_signal'] = df['long_signal'] & (df['rsi'] < 70)
        df['short_signal'] = df['short_signal'] & (df['rsi'] > 30)
        
        # Momentum consistency filter
        df['long_signal'] = df['long_signal'] & (df['momentum_5d'] > 0)
        df['short_signal'] = df['short_signal'] & (df['momentum_5d'] < 0)
        
        # Apply minimum holding period filter
        df = self.filters.apply_holding_period_filter(df, min_holding_days=3)
        
        # Add mean return strength filter if using mean return signals
        if self.use_mean_return_signal and 'mean_return_strength' in df.columns:
            # Only take signals with sufficient strength
            min_strength = 0.5  # Half standard deviation
            df['long_signal'] = df['long_signal'] & (df['mean_return_strength'] > min_strength)
            df['short_signal'] = df['short_signal'] & (df['mean_return_strength'] > min_strength)
        
        return df
    
    def calculate_signal_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate signal strength/confidence scores.
        
        Args:
            df: DataFrame with signals
            
        Returns:
            DataFrame with signal strength scores
        """
        df = df.copy()
        
        # Initialize scores
        df['long_strength'] = 0.0
        df['short_strength'] = 0.0
        
        # Original momentum-based scoring
        adx_score = np.clip((df['adx'] - 20) * 2, 0, 40)
        ewma_diff_pct = (df['ewma_fast'] - df['ewma_slow']) / df['ewma_slow'] * 100
        ewma_score = np.clip(abs(ewma_diff_pct) * 3, 0, 30)
        volume_score = np.clip((df['volume_ratio'] - 1) * 20, 0, 20)
        rsi_long_score = np.clip((60 - df['rsi']) / 4, 0, 10)
        rsi_short_score = np.clip((df['rsi'] - 40) / 4, 0, 10)
        
        # Add mean return contribution if enabled
        if self.use_mean_return_signal and 'mean_return_strength' in df.columns:
            mean_return_score = np.clip(df['mean_return_strength'] * 20, 0, 30)
            
            # Combine scores based on signal type
            if self.combine_signals == 'override':
                # Mean return is primary signal
                df.loc[df['long_signal'], 'long_strength'] = (
                    mean_return_score + volume_score + rsi_long_score
                ).loc[df['long_signal']]
                
                df.loc[df['short_signal'], 'short_strength'] = (
                    mean_return_score + volume_score + rsi_short_score
                ).loc[df['short_signal']]
                
            else:
                # Combine all scores
                df.loc[df['long_signal'], 'long_strength'] = (
                    adx_score + ewma_score + volume_score + rsi_long_score + mean_return_score
                ).loc[df['long_signal']]
                
                df.loc[df['short_signal'], 'short_strength'] = (
                    adx_score + ewma_score + volume_score + rsi_short_score + mean_return_score
                ).loc[df['short_signal']]
        else:
            # Original scoring
            df.loc[df['long_signal'], 'long_strength'] = (
                adx_score + ewma_score + volume_score + rsi_long_score
            ).loc[df['long_signal']]
            
            df.loc[df['short_signal'], 'short_strength'] = (
                adx_score + ewma_score + volume_score + rsi_short_score
            ).loc[df['short_signal']]
        
        # Normalize to 0-100
        df['long_strength'] = np.clip(df['long_strength'], 0, 100)
        df['short_strength'] = np.clip(df['short_strength'], 0, 100)
        
        return df
    
    def generate_signals(
        self,
        df: pd.DataFrame,
        apply_filters: bool = True,
        calculate_strength: bool = True
    ) -> pd.DataFrame:
        """
        Generate complete trading signals.
        
        Args:
            df: OHLC DataFrame
            apply_filters: Whether to apply additional filters
            calculate_strength: Whether to calculate signal strength
            
        Returns:
            DataFrame with all signals and indicators
        """
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Generate base signals
        df = self.generate_base_signals(df)
        
        # Apply filters if requested
        if apply_filters:
            df = self.apply_filters(df)
        
        # Calculate signal strength if requested
        if calculate_strength:
            df = self.calculate_signal_strength(df)
        
        # Add position column
        df['position'] = 0
        df.loc[df['long_signal'], 'position'] = 1
        df.loc[df['short_signal'], 'position'] = -1
        
        # Forward fill positions
        df['position'] = df['position'].replace(0, np.nan).ffill().fillna(0)
        
        signal_type = "mean return" if self.use_mean_return_signal else "momentum"
        self.logger.info(
            f"Generated {signal_type} signals - Long: {df['long_signal'].sum()}, "
            f"Short: {df['short_signal'].sum()}"
        )
        
        return df
    
    def get_signal_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get a summary of the generated signals.
        
        Args:
            df: DataFrame with signals
            
        Returns:
            Dictionary with signal statistics
        """
        summary = {
            'total_long_signals': df['long_signal'].sum(),
            'total_short_signals': df['short_signal'].sum(),
            'signal_type': 'mean_return' if self.use_mean_return_signal else 'momentum',
            'combine_method': self.combine_signals if self.use_mean_return_signal else 'N/A'
        }
        
        if self.use_mean_return_signal and 'mean_return' in df.columns:
            summary.update({
                'avg_mean_return': df['mean_return'].mean(),
                'mean_return_volatility': df['mean_return'].std(),
                'current_mean_return': df['mean_return'].iloc[-1] if len(df) > 0 else 0
            })
        
        return summary