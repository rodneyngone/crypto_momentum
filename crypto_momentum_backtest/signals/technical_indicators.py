# crypto_momentum_backtest/signals/technical_indicators.py
"""Technical indicators for signal generation."""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, Dict, List
import numba
from numba import jit


@jit(nopython=True)
def calculate_adx_components(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate ADX components using Numba for speed.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period
        
    Returns:
        Tuple of (ADX, +DI, -DI)
    """
    n = len(high)
    
    # Initialize arrays
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    # Calculate True Range and Directional Movement
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
        
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth the values
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    # Initial values
    atr[period] = np.mean(tr[1:period+1])
    plus_di[period] = np.mean(plus_dm[1:period+1])
    minus_di[period] = np.mean(minus_dm[1:period+1])
    
    # Calculate smoothed values
    for i in range(period + 1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        plus_di[i] = (plus_di[i-1] * (period - 1) + plus_dm[i]) / period
        minus_di[i] = (minus_di[i-1] * (period - 1) + minus_dm[i]) / period
    
    # Calculate DI values
    for i in range(period, n):
        if atr[i] != 0:
            plus_di[i] = 100 * plus_di[i] / atr[i]
            minus_di[i] = 100 * minus_di[i] / atr[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Calculate ADX
    adx[2*period-1] = np.mean(dx[period:2*period])
    
    for i in range(2*period, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx, plus_di, minus_di


class TechnicalIndicators:
    """
    Collection of technical indicators optimized for performance.
    """
    
    @staticmethod
    def adx(
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average Directional Index.
        
        Args:
            df: DataFrame with OHLC data
            period: ADX period
            
        Returns:
            ADX series
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        adx, _, _ = calculate_adx_components(high, low, close, period)
        
        return pd.Series(adx, index=df.index, name='adx')
    
    @staticmethod
    def ewma(
        series: pd.Series,
        span: int,
        min_periods: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate Exponential Weighted Moving Average.
        
        Args:
            series: Input series
            span: EMA span
            min_periods: Minimum periods required
            
        Returns:
            EWMA series
        """
        if min_periods is None:
            min_periods = span
            
        return series.ewm(span=span, min_periods=min_periods, adjust=False).mean()
    
    @staticmethod
    def atr(
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            df: DataFrame with OHLC data
            period: ATR period
            
        Returns:
            ATR series
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
        
        return atr.rename('atr')
    
    @staticmethod
    def volume_ratio(
        volume: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """
        Calculate volume ratio compared to moving average.
        
        Args:
            volume: Volume series
            period: MA period
            
        Returns:
            Volume ratio series
        """
        vol_ma = volume.rolling(window=period, min_periods=period).mean()
        return (volume / vol_ma).rename('volume_ratio')
    
    @staticmethod
    def volatility(
        returns: pd.Series,
        period: int = 20,
        annualize: bool = True
    ) -> pd.Series:
        """
        Calculate rolling volatility.
        
        Args:
            returns: Returns series
            period: Rolling window
            annualize: Whether to annualize
            
        Returns:
            Volatility series
        """
        vol = returns.rolling(window=period, min_periods=period).std()
        
        if annualize:
            vol = vol * np.sqrt(365)  # Assuming daily data
            
        return vol.rename('volatility')
    
    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        period: int = 20,
        num_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            series: Price series
            period: MA period
            num_std: Number of standard deviations
            
        Returns:
            DataFrame with upper, middle, lower bands
        """
        middle = series.rolling(window=period, min_periods=period).mean()
        std = series.rolling(window=period, min_periods=period).std()
        
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        
        return pd.DataFrame({
            'bb_upper': upper,
            'bb_middle': middle,
            'bb_lower': lower
        })
    
    @staticmethod
    def rsi(
        series: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            series: Price series
            period: RSI period
            
        Returns:
            RSI series
        """
        delta = series.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.rename('rsi')
    
    @staticmethod
    def macd(
        series: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> pd.DataFrame:
        """
        Calculate MACD indicator.
        
        Args:
            series: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            DataFrame with MACD, signal, and histogram
        """
        ema_fast = series.ewm(span=fast_period, adjust=False).mean()
        ema_slow = series.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        })
    
    @staticmethod
    def support_resistance(
        df: pd.DataFrame,
        window: int = 20,
        num_levels: int = 3
    ) -> Dict[str, List[float]]:
        """
        Identify support and resistance levels.
        
        Args:
            df: OHLC DataFrame
            window: Rolling window for finding levels
            num_levels: Number of levels to identify
            
        Returns:
            Dictionary with support and resistance levels
        """
        # Find local peaks and troughs
        highs = df['high'].rolling(window=window, center=True).max() == df['high']
        lows = df['low'].rolling(window=window, center=True).min() == df['low']
        
        # Get resistance levels from peaks
        resistance_levels = df.loc[highs, 'high'].nlargest(num_levels).tolist()
        
        # Get support levels from troughs
        support_levels = df.loc[lows, 'low'].nsmallest(num_levels).tolist()
        
        return {
            'resistance': sorted(resistance_levels, reverse=True),
            'support': sorted(support_levels)
        }
