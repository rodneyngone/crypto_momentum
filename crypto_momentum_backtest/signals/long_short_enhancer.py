# File: crypto_momentum_backtest/signals/long_short_enhancer.py
"""
Enhanced signal generation for long/short portfolios.
This can be integrated into the existing signal generator or used as a wrapper.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


class LongShortSignalEnhancer:
    """Enhance signals for proper long/short portfolio generation."""
    
    def __init__(self, 
                 long_threshold: float = 0.2,
                 short_threshold: float = -0.2,
                 market_neutral: bool = False,
                 max_net_exposure: float = 0.5):
        """
        Initialize long/short signal enhancer.
        
        Args:
            long_threshold: Minimum score for long signals
            short_threshold: Maximum score for short signals (negative)
            market_neutral: If True, balance long and short exposure
            max_net_exposure: Maximum net exposure (long - short)
        """
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.market_neutral = market_neutral
        self.max_net_exposure = max_net_exposure
    
    def enhance_momentum_scores(self, scores: pd.Series) -> pd.Series:
        """
        Enhance momentum scores to ensure both long and short signals.
        
        Args:
            scores: Raw momentum scores
            
        Returns:
            Enhanced scores with clear long/short signals
        """
        enhanced = scores.copy()
        
        # Normalize scores to ensure good distribution
        score_mean = scores.mean()
        score_std = scores.std()
        
        if score_std > 0:
            # Z-score normalization
            enhanced = (scores - score_mean) / score_std
            
            # Scale to typical range [-1, 1]
            enhanced = np.tanh(enhanced * 0.5)
        
        return enhanced
    
    def generate_discrete_signals(self, scores: pd.Series) -> pd.Series:
        """
        Convert continuous scores to discrete long/short signals.
        
        Args:
            scores: Continuous momentum scores
            
        Returns:
            Discrete signals: 1 (long), -1 (short), 0 (neutral)
        """
        signals = pd.Series(0, index=scores.index)
        
        # Long signals
        signals[scores > self.long_threshold] = 1
        
        # Short signals
        signals[scores < self.short_threshold] = -1
        
        return signals
    
    def balance_signals(self, signals: pd.DataFrame, 
                       target_ratio: float = 1.0) -> pd.DataFrame:
        """
        Balance long and short signals to achieve target ratio.
        
        Args:
            signals: DataFrame of signals by asset
            target_ratio: Target long/short ratio (1.0 = equal)
            
        Returns:
            Balanced signals
        """
        balanced = signals.copy()
        
        for idx in signals.index:
            row_signals = signals.loc[idx]
            n_long = (row_signals == 1).sum()
            n_short = (row_signals == -1).sum()
            
            if n_long == 0 or n_short == 0:
                continue
            
            current_ratio = n_long / n_short if n_short > 0 else np.inf
            
            # Adjust if ratio is too imbalanced
            if current_ratio > target_ratio * 1.5:
                # Too many longs, convert some to neutral
                long_signals = row_signals[row_signals == 1]
                n_to_remove = int(n_long - n_short * target_ratio)
                if n_to_remove > 0:
                    # Remove weakest long signals
                    weakest = long_signals.nsmallest(n_to_remove).index
                    balanced.loc[idx, weakest] = 0
                    
            elif current_ratio < target_ratio * 0.67:
                # Too many shorts, convert some to neutral
                short_signals = row_signals[row_signals == -1]
                n_to_remove = int(n_short - n_long / target_ratio)
                if n_to_remove > 0:
                    # Remove weakest short signals
                    weakest = short_signals.nlargest(n_to_remove).index
                    balanced.loc[idx, weakest] = 0
        
        return balanced
    
    def calculate_signal_stats(self, signals: pd.DataFrame) -> Dict:
        """Calculate statistics about signal distribution."""
        stats = {}
        
        # Overall stats
        total_signals = signals.abs().sum().sum()
        long_signals = (signals == 1).sum().sum()
        short_signals = (signals == -1).sum().sum()
        
        stats['total_signals'] = total_signals
        stats['long_signals'] = long_signals
        stats['short_signals'] = short_signals
        stats['long_ratio'] = long_signals / total_signals if total_signals > 0 else 0
        stats['short_ratio'] = short_signals / total_signals if total_signals > 0 else 0
        
        # Time series stats
        stats['avg_long_per_period'] = (signals == 1).sum(axis=1).mean()
        stats['avg_short_per_period'] = (signals == -1).sum(axis=1).mean()
        stats['avg_net_exposure'] = (signals.sum(axis=1)).mean()
        
        return stats


def create_long_short_signals(data: pd.DataFrame, 
                             method: str = 'relative_momentum') -> pd.Series:
    """
    Create long/short signals using various methods.
    
    Args:
        data: OHLCV data
        method: Signal generation method
        
    Returns:
        Signal series with -1, 0, 1 values
    """
    if method == 'relative_momentum':
        # Calculate returns
        returns = data['close'].pct_change(20)  # 20-day returns
        
        # Rank assets by returns
        ranked = returns.rank(pct=True)
        
        # Top 30% long, bottom 30% short
        signals = pd.Series(0, index=data.index)
        signals[ranked > 0.7] = 1
        signals[ranked < 0.3] = -1
        
    elif method == 'mean_reversion':
        # RSI-based mean reversion
        from ..signals.technical_indicators import TechnicalIndicators
        
        rsi = TechnicalIndicators.rsi(data['close'])
        
        signals = pd.Series(0, index=data.index)
        signals[rsi < 30] = 1   # Oversold -> Long
        signals[rsi > 70] = -1  # Overbought -> Short
        
    elif method == 'trend_following':
        # Dual moving average with shorts
        fast_ma = data['close'].rolling(10).mean()
        slow_ma = data['close'].rolling(30).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[(fast_ma > slow_ma) & (data['close'] > slow_ma)] = 1
        signals[(fast_ma < slow_ma) & (data['close'] < slow_ma)] = -1
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return signals


# Integration function for existing codebase
def ensure_long_short_signals(signal_generator, data: pd.DataFrame, 
                            symbol: str = None) -> pd.Series:
    """
    Wrapper to ensure signal generator produces both long and short signals.
    """
    # Get base signals
    signals = signal_generator.generate_signals(data, signal_type='momentum_score', symbol=symbol)
    
    # Check if we have both long and short signals
    if isinstance(signals, pd.Series):
        n_long = (signals > 0).sum()
        n_short = (signals < 0).sum()
        
        print(f"Signal distribution for {symbol}:")
        print(f"  Long signals: {n_long}")
        print(f"  Short signals: {n_short}")
        
        # If too imbalanced, enhance
        if n_short == 0 or n_long / (n_short + 1) > 3:
            enhancer = LongShortSignalEnhancer()
            signals = enhancer.enhance_momentum_scores(signals)
            
            # Re-discretize if needed
            if signals.abs().max() <= 1:
                discrete = enhancer.generate_discrete_signals(signals)
                return discrete
    
    return signals