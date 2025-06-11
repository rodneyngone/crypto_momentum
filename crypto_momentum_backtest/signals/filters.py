# crypto_momentum_backtest/signals/filters.py
"""Signal filtering utilities."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from scipy.stats import zscore


class SignalFilters:
    """
    Collection of filters to improve signal quality.
    """
    
    @staticmethod
    def apply_holding_period_filter(
        df: pd.DataFrame,
        min_holding_days: int = 3,
        signal_cols: List[str] = ['long_signal', 'short_signal']
    ) -> pd.DataFrame:
        """
        Ensure minimum holding period between signals.
        
        Args:
            df: DataFrame with signals
            min_holding_days: Minimum days between signals
            signal_cols: Signal column names
            
        Returns:
            Filtered DataFrame
        """
        df = df.copy()
        
        for col in signal_cols:
            if col not in df.columns:
                continue
                
            # Find signal dates
            signal_dates = df.index[df[col]]
            
            if len(signal_dates) > 1:
                # Check time between signals
                filtered_dates = [signal_dates[0]]
                
                for date in signal_dates[1:]:
                    days_since_last = (date - filtered_dates[-1]).days
                    if days_since_last >= min_holding_days:
                        filtered_dates.append(date)
                
                # Update signals
                df[col] = False
                df.loc[filtered_dates, col] = True
        
        return df
    
    @staticmethod
    def apply_volatility_filter(
        df: pd.DataFrame,
        vol_col: str = 'volatility',
        max_vol_zscore: float = 2.0
    ) -> pd.DataFrame:
        """
        Filter signals during extreme volatility.
        
        Args:
            df: DataFrame with signals and volatility
            vol_col: Volatility column name
            max_vol_zscore: Maximum volatility z-score
            
        Returns:
            Filtered DataFrame
        """
        df = df.copy()
        
        if vol_col not in df.columns:
            return df
        
        # Calculate volatility z-score
        vol_zscore = zscore(df[vol_col].dropna())
        vol_zscore = pd.Series(vol_zscore, index=df[vol_col].dropna().index)
        
        # Filter signals
        high_vol_mask = vol_zscore.abs() > max_vol_zscore
        
        for col in ['long_signal', 'short_signal']:
            if col in df.columns:
                df.loc[high_vol_mask, col] = False
        
        return df
    
    @staticmethod
    def apply_correlation_filter(
        df: pd.DataFrame,
        returns_col: str = 'returns',
        lookback: int = 20,
        max_correlation: float = 0.8
    ) -> pd.DataFrame:
        """
        Filter signals based on recent correlation patterns.
        
        Args:
            df: DataFrame with returns
            returns_col: Returns column name
            lookback: Correlation lookback period
            max_correlation: Maximum allowed correlation
            
        Returns:
            Filtered DataFrame
        """
        # This is a placeholder for correlation-based filtering
        # In practice, this would compare against other assets
        return df
    
    @staticmethod
    def apply_regime_filter(
        df: pd.DataFrame,
        regime_col: str = 'market_regime',
        allowed_regimes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Filter signals based on market regime.
        
        Args:
            df: DataFrame with signals and regime
            regime_col: Regime column name
            allowed_regimes: List of regimes to trade in
            
        Returns:
            Filtered DataFrame
        """
        df = df.copy()
        
        if regime_col not in df.columns:
            return df
        
        if allowed_regimes is None:
            allowed_regimes = ['trending', 'normal']
        
        # Filter signals
        regime_mask = ~df[regime_col].isin(allowed_regimes)
        
        for col in ['long_signal', 'short_signal']:
            if col in df.columns:
                df.loc[regime_mask, col] = False
        
        return df
    
    @staticmethod
    def apply_drawdown_filter(
        df: pd.DataFrame,
        equity_col: str = 'equity',
        max_drawdown: float = 0.15
    ) -> pd.DataFrame:
        """
        Stop generating new signals during drawdowns.
        
        Args:
            df: DataFrame with equity curve
            equity_col: Equity column name
            max_drawdown: Maximum drawdown threshold
            
        Returns:
            Filtered DataFrame
        """
        df = df.copy()
        
        if equity_col not in df.columns:
            return df
        
        # Calculate running drawdown
        running_max = df[equity_col].expanding().max()
        drawdown = (df[equity_col] - running_max) / running_max
        
        # Filter signals during drawdown
        dd_mask = drawdown < -max_drawdown
        
        for col in ['long_signal', 'short_signal']:
            if col in df.columns:
                df.loc[dd_mask, col] = False
        
        return df
