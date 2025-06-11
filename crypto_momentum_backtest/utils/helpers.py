# crypto_momentum_backtest/utils/helpers.py
"""Helper utilities for the backtesting system."""
import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Optional
from datetime import datetime, timedelta
import numba


@numba.jit(nopython=True)
def fast_ewma(values: np.ndarray, span: int) -> np.ndarray:
    """
    Fast exponentially weighted moving average using Numba.
    
    Args:
        values: Input array
        span: EMA span
        
    Returns:
        EMA array
    """
    alpha = 2.0 / (span + 1.0)
    ewma = np.empty_like(values)
    ewma[0] = values[0]
    
    for i in range(1, len(values)):
        ewma[i] = alpha * values[i] + (1 - alpha) * ewma[i-1]
    
    return ewma


def resample_ohlcv(
    df: pd.DataFrame,
    freq: str,
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    Resample OHLCV data to different frequency.
    
    Args:
        df: OHLCV DataFrame
        freq: Target frequency
        price_col: Price column for volume weighting
        
    Returns:
        Resampled DataFrame
    """
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    return df.resample(freq).agg(agg_dict).dropna()


def calculate_returns(
    prices: Union[pd.Series, pd.DataFrame],
    method: str = 'simple'
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series or DataFrame
        method: 'simple' or 'log'
        
    Returns:
        Returns
    """
    if method == 'simple':
        return prices.pct_change()
    elif method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unknown method: {method}")


def chunk_date_range(
    start_date: datetime,
    end_date: datetime,
    chunk_size: str = 'month'
) -> List[Tuple[datetime, datetime]]:
    """
    Split date range into chunks.
    
    Args:
        start_date: Start date
        end_date: End date
        chunk_size: 'month' or 'week'
        
    Returns:
        List of (start, end) tuples
    """
    chunks = []
    current = start_date
    
    while current < end_date:
        if chunk_size == 'month':
            # Get last day of month
            if current.month == 12:
                chunk_end = datetime(current.year + 1, 1, 1) - timedelta(days=1)
            else:
                chunk_end = datetime(current.year, current.month + 1, 1) - timedelta(days=1)
        elif chunk_size == 'week':
            chunk_end = current + timedelta(days=7)
        else:
            raise ValueError(f"Unknown chunk size: {chunk_size}")
        
        chunk_end = min(chunk_end, end_date)
        chunks.append((current, chunk_end))
        
        current = chunk_end + timedelta(days=1)
    
    return chunks
