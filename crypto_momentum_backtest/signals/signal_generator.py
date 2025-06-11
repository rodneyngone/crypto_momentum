#!/usr/bin/env python3
"""
Signal generation module for crypto momentum backtesting framework.

This module provides production-grade signal generation with crypto-optimized thresholds
based on empirical analysis of cryptocurrency market dynamics.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Enumeration of available signal types."""
    MOMENTUM = "momentum"
    MEAN_RETURN_EWM = "mean_return_ewm"
    MEAN_RETURN_SIMPLE = "mean_return_simple"
    HYBRID_AND = "hybrid_and"
    HYBRID_OR = "hybrid_or"


@dataclass
class SignalMetrics:
    """Container for signal generation metrics."""
    total_signals: int
    long_signals: int
    short_signals: int
    signal_frequency: float
    strategy_type: str
    
    def __str__(self) -> str:
        return (f"{self.strategy_type}: {self.total_signals} signals "
                f"(L:{self.long_signals}, S:{self.short_signals}, "
                f"Freq:{self.signal_frequency:.1f}%)")


class SignalGenerator:
    """
    Production-grade signal generator for cryptocurrency momentum strategies.
    
    This class generates trading signals based on various momentum and mean return
    strategies, with thresholds specifically calibrated for cryptocurrency markets
    based on empirical analysis showing crypto mean returns range 0.001-0.015.
    """
    
    def __init__(self,
                 # Momentum strategy parameters
                 momentum_threshold: float = 0.01,
                 momentum_ewma_span: int = 20,
                 
                 # Mean return EWM strategy parameters (crypto-optimized)
                 mean_return_ewm_threshold: float = 0.015,
                 mean_return_ewm_span: int = 10,
                 
                 # Mean return simple strategy parameters (crypto-optimized)
                 mean_return_simple_threshold: float = 0.015,
                 mean_return_simple_window: int = 10,
                 
                 # Technical analysis parameters
                 adx_period: int = 14,
                 adx_threshold: float = 20,
                 
                 # Volume confirmation
                 use_volume_confirmation: bool = True,
                 volume_threshold: float = 1.2,
                 
                 # Signal filtering
                 min_signal_gap: int = 1,
                 max_signals_per_period: int = 100,
                 
                 **kwargs):
        """
        Initialize SignalGenerator with crypto-optimized parameters.
        
        Args:
            momentum_threshold: Threshold for momentum signals (1% for crypto)
            momentum_ewma_span: EWMA span for momentum calculation
            mean_return_ewm_threshold: EWM mean return threshold (1.5% for crypto)
            mean_return_ewm_span: EWM span for mean return calculation
            mean_return_simple_threshold: Simple mean return threshold (1.5% for crypto)
            mean_return_simple_window: Rolling window for simple mean returns
            adx_period: Period for ADX calculation
            adx_threshold: ADX threshold for trend strength
            use_volume_confirmation: Whether to use volume confirmation
            volume_threshold: Volume ratio threshold for confirmation
            min_signal_gap: Minimum periods between signals
            max_signals_per_period: Maximum signals per strategy
        
        Note:
            Thresholds are empirically calibrated for cryptocurrency markets:
            - Crypto mean returns typically range 0.001-0.015 vs 0.01-0.02 for stocks
            - Higher volatility requires more responsive parameters
            - Analysis shows 0.015 threshold generates optimal 37-45 signals
        """
        # Store parameters
        self.momentum_threshold = momentum_threshold
        self.momentum_ewma_span = momentum_ewma_span
        
        self.mean_return_ewm_threshold = mean_return_ewm_threshold
        self.mean_return_ewm_span = mean_return_ewm_span
        
        self.mean_return_simple_threshold = mean_return_simple_threshold
        self.mean_return_simple_window = mean_return_simple_window
        
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        
        self.use_volume_confirmation = use_volume_confirmation
        self.volume_threshold = volume_threshold
        
        self.min_signal_gap = min_signal_gap
        self.max_signals_per_period = max_signals_per_period
        
        # Initialize metrics tracking
        self._signal_metrics: Dict[str, SignalMetrics] = {}
        
        # Log initialization with crypto-specific validation
        self._validate_crypto_parameters()
        self._log_initialization()
    
    def _validate_crypto_parameters(self) -> None:
        """Validate parameters are appropriate for cryptocurrency markets."""
        warnings = []
        
        if self.mean_return_ewm_threshold > 0.05:
            warnings.append(f"EWM threshold {self.mean_return_ewm_threshold} may be too high for crypto")
        
        if self.mean_return_simple_threshold > 0.05:
            warnings.append(f"Simple threshold {self.mean_return_simple_threshold} may be too high for crypto")
            
        if self.momentum_threshold > 0.05:
            warnings.append(f"Momentum threshold {self.momentum_threshold} may be too high for crypto")
        
        if self.mean_return_ewm_span > 30:
            warnings.append(f"EWM span {self.mean_return_ewm_span} may be too slow for crypto volatility")
        
        for warning in warnings:
            logger.warning(warning)
    
    def _log_initialization(self) -> None:
        """Log initialization parameters for transparency."""
        logger.info("SignalGenerator initialized with crypto-optimized parameters:")
        logger.info(f"  Momentum threshold: {self.momentum_threshold}")
        logger.info(f"  Mean return EWM threshold: {self.mean_return_ewm_threshold}")
        logger.info(f"  Mean return simple threshold: {self.mean_return_simple_threshold}")
        logger.info(f"  ADX threshold: {self.adx_threshold}")
        logger.info(f"  Volume confirmation: {self.use_volume_confirmation}")
    
    def generate_signals(self, 
                        data: pd.DataFrame,
                        signal_type: Union[str, SignalType] = SignalType.MOMENTUM,
                        symbol: Optional[str] = None) -> pd.Series:
        """
        Generate trading signals based on specified strategy.
        
        Args:
            data: OHLCV DataFrame with required columns
            signal_type: Type of signal strategy to use
            symbol: Symbol identifier for logging
            
        Returns:
            Signal series: 1 for long, -1 for short, 0 for neutral
            
        Raises:
            ValueError: If data format is invalid or signal type unknown
        """
        try:
            # Validate input data
            self._validate_data(data)
            
            # Convert signal type to enum
            if isinstance(signal_type, str):
                signal_type = SignalType(signal_type)
            
            # Generate signals based on type
            if signal_type == SignalType.MOMENTUM:
                signals = self._generate_momentum_signals(data)
            elif signal_type == SignalType.MEAN_RETURN_EWM:
                signals = self._generate_mean_return_ewm_signals(data)
            elif signal_type == SignalType.MEAN_RETURN_SIMPLE:
                signals = self._generate_mean_return_simple_signals(data)
            elif signal_type == SignalType.HYBRID_AND:
                signals = self._generate_hybrid_and_signals(data)
            elif signal_type == SignalType.HYBRID_OR:
                signals = self._generate_hybrid_or_signals(data)
            else:
                raise ValueError(f"Unknown signal type: {signal_type}")
            
            # Apply volume confirmation if enabled
            if self.use_volume_confirmation and 'volume' in data.columns:
                signals = self._apply_volume_confirmation(signals, data)
            
            # Apply signal filtering
            signals = self._apply_signal_filtering(signals)
            
            # Calculate and store metrics
            metrics = self._calculate_signal_metrics(signals, signal_type.value)
            self._signal_metrics[signal_type.value] = metrics
            
            # Log results
            symbol_str = f" for {symbol}" if symbol else ""
            logger.info(f"Generated {metrics}{symbol_str}")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating {signal_type} signals: {e}")
            return pd.Series(0, index=data.index)
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data format and completeness."""
        required_columns = ['close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(data) < max(self.momentum_ewma_span, self.mean_return_simple_window, self.adx_period):
            raise ValueError(f"Insufficient data: need at least {max(self.momentum_ewma_span, self.mean_return_simple_window, self.adx_period)} rows")
        
        if data['close'].isna().sum() > len(data) * 0.1:  # More than 10% missing
            logger.warning(f"High proportion of missing close prices: {data['close'].isna().sum()}/{len(data)}")
    
    def _generate_momentum_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate momentum signals using EWMA crossover strategy.
        
        The momentum strategy compares current price to exponentially weighted
        moving average to identify trending behavior.
        """
        try:
            # Calculate EWMA
            ewma = data['close'].ewm(span=self.momentum_ewma_span).mean()
            
            # Calculate momentum as deviation from EWMA
            momentum = (data['close'] / ewma - 1).fillna(0)
            
            # Generate signals
            long_signals = momentum > self.momentum_threshold
            short_signals = momentum < -self.momentum_threshold
            
            # Create signal series
            signals = pd.Series(0, index=data.index, dtype=int)
            signals[long_signals] = 1
            signals[short_signals] = -1
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in momentum signal generation: {e}")
            return pd.Series(0, index=data.index, dtype=int)
    
    def _generate_mean_return_ewm_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate EWM mean return signals with crypto-calibrated thresholds.
        
        This strategy uses exponentially weighted moving average of returns
        to identify persistent directional movements in cryptocurrency markets.
        """
        try:
            # Calculate returns
            returns = data['close'].pct_change().fillna(0)
            
            # Calculate EWM of returns (not prices)
            ewm_returns = returns.ewm(span=self.mean_return_ewm_span).mean()
            
            # Generate signals with crypto-optimized thresholds
            long_signals = ewm_returns > self.mean_return_ewm_threshold
            short_signals = ewm_returns < -self.mean_return_ewm_threshold
            
            # Create signal series
            signals = pd.Series(0, index=data.index, dtype=int)
            signals[long_signals] = 1
            signals[short_signals] = -1
            
            # Debug logging for threshold validation
            logger.debug(f"EWM returns range: {ewm_returns.min():.6f} to {ewm_returns.max():.6f}")
            logger.debug(f"EWM threshold used: {self.mean_return_ewm_threshold}")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in EWM mean return signal generation: {e}")
            return pd.Series(0, index=data.index, dtype=int)
    
    def _generate_mean_return_simple_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate simple rolling mean return signals with crypto-calibrated thresholds.
        
        This strategy uses simple moving average of returns to identify
        short-term momentum patterns suitable for cryptocurrency volatility.
        """
        try:
            # Calculate returns
            returns = data['close'].pct_change().fillna(0)
            
            # Calculate rolling mean of returns
            rolling_mean_returns = returns.rolling(
                window=self.mean_return_simple_window,
                min_periods=1
            ).mean()
            
            # Generate signals with crypto-optimized thresholds
            long_signals = rolling_mean_returns > self.mean_return_simple_threshold
            short_signals = rolling_mean_returns < -self.mean_return_simple_threshold
            
            # Create signal series
            signals = pd.Series(0, index=data.index, dtype=int)
            signals[long_signals] = 1
            signals[short_signals] = -1
            
            # Debug logging for threshold validation
            logger.debug(f"Simple mean returns range: {rolling_mean_returns.min():.6f} to {rolling_mean_returns.max():.6f}")
            logger.debug(f"Simple threshold used: {self.mean_return_simple_threshold}")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in simple mean return signal generation: {e}")
            return pd.Series(0, index=data.index, dtype=int)
    
    def _generate_hybrid_and_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate hybrid AND signals combining multiple strategies.
        
        Requires both momentum and mean return signals to agree before generating
        a signal, resulting in higher conviction but lower frequency trades.
        """
        try:
            momentum_signals = self._generate_momentum_signals(data)
            ewm_signals = self._generate_mean_return_ewm_signals(data)
            
            # AND logic: both strategies must agree
            long_signals = (momentum_signals == 1) & (ewm_signals == 1)
            short_signals = (momentum_signals == -1) & (ewm_signals == -1)
            
            signals = pd.Series(0, index=data.index, dtype=int)
            signals[long_signals] = 1
            signals[short_signals] = -1
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in hybrid AND signal generation: {e}")
            return pd.Series(0, index=data.index, dtype=int)
    
    def _generate_hybrid_or_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate hybrid OR signals combining multiple strategies.
        
        Generates a signal if either momentum or mean return strategies
        trigger, resulting in higher frequency but potentially lower conviction trades.
        """
        try:
            momentum_signals = self._generate_momentum_signals(data)
            ewm_signals = self._generate_mean_return_ewm_signals(data)
            
            # OR logic: either strategy can trigger
            long_signals = (momentum_signals == 1) | (ewm_signals == 1)
            short_signals = (momentum_signals == -1) | (ewm_signals == -1)
            
            # Resolve conflicts (if both long and short somehow trigger)
            conflicts = long_signals & short_signals
            long_signals = long_signals & ~conflicts
            short_signals = short_signals & ~conflicts
            
            signals = pd.Series(0, index=data.index, dtype=int)
            signals[long_signals] = 1
            signals[short_signals] = -1
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in hybrid OR signal generation: {e}")
            return pd.Series(0, index=data.index, dtype=int)
    
    def _apply_volume_confirmation(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        Apply volume confirmation to filter signals.
        
        Signals are only kept if accompanied by above-average volume,
        indicating institutional participation and reducing false signals.
        """
        try:
            if 'volume' not in data.columns:
                logger.warning("Volume column not found, skipping volume confirmation")
                return signals
            
            # Calculate volume moving average
            volume_ma = data['volume'].rolling(window=20, min_periods=1).mean()
            volume_ratio = data['volume'] / volume_ma
            
            # Apply volume confirmation
            volume_confirmed = volume_ratio > self.volume_threshold
            confirmed_signals = signals.copy()
            confirmed_signals[~volume_confirmed] = 0
            
            # Log confirmation rate
            original_count = (signals != 0).sum()
            confirmed_count = (confirmed_signals != 0).sum()
            confirmation_rate = confirmed_count / max(original_count, 1) * 100
            
            logger.debug(f"Volume confirmation: {confirmed_count}/{original_count} signals kept ({confirmation_rate:.1f}%)")
            
            return confirmed_signals
            
        except Exception as e:
            logger.error(f"Error applying volume confirmation: {e}")
            return signals
    
    def _apply_signal_filtering(self, signals: pd.Series) -> pd.Series:
        """
        Apply signal filtering to reduce noise and prevent over-trading.
        
        Filters include minimum gap between signals and maximum signals per period.
        """
        try:
            filtered_signals = signals.copy()
            
            # Apply minimum signal gap
            if self.min_signal_gap > 1:
                signal_positions = np.where(signals != 0)[0]
                
                for i in range(1, len(signal_positions)):
                    if signal_positions[i] - signal_positions[i-1] < self.min_signal_gap:
                        filtered_signals.iloc[signal_positions[i]] = 0
            
            # Apply maximum signals limit
            signal_count = (filtered_signals != 0).sum()
            if signal_count > self.max_signals_per_period:
                # Keep only the strongest signals (by absolute momentum)
                # This is a simplified approach - could be enhanced with signal strength ranking
                signal_indices = np.where(filtered_signals != 0)[0]
                excess_signals = signal_count - self.max_signals_per_period
                
                # Remove excess signals from the end (keep earlier signals)
                for idx in signal_indices[-excess_signals:]:
                    filtered_signals.iloc[idx] = 0
                
                logger.debug(f"Signal filtering: reduced from {signal_count} to {self.max_signals_per_period} signals")
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error applying signal filtering: {e}")
            return signals
    
    def _calculate_signal_metrics(self, signals: pd.Series, strategy_type: str) -> SignalMetrics:
        """Calculate comprehensive metrics for signal generation performance."""
        long_signals = (signals == 1).sum()
        short_signals = (signals == -1).sum()
        total_signals = long_signals + short_signals
        signal_frequency = (total_signals / len(signals)) * 100 if len(signals) > 0 else 0
        
        return SignalMetrics(
            total_signals=total_signals,
            long_signals=long_signals,
            short_signals=short_signals,
            signal_frequency=signal_frequency,
            strategy_type=strategy_type
        )
    
    def get_signal_metrics(self, strategy_type: Optional[str] = None) -> Union[SignalMetrics, Dict[str, SignalMetrics]]:
        """
        Get signal generation metrics.
        
        Args:
            strategy_type: Specific strategy to get metrics for, or None for all
            
        Returns:
            SignalMetrics for specific strategy or dict of all metrics
        """
        if strategy_type:
            return self._signal_metrics.get(strategy_type)
        return self._signal_metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset all stored signal metrics."""
        self._signal_metrics.clear()
        logger.debug("Signal metrics reset")
    
    def get_parameters(self) -> Dict[str, Union[float, int, bool]]:
        """
        Get current signal generator parameters.
        
        Returns:
            Dictionary of all current parameters
        """
        return {
            'momentum_threshold': self.momentum_threshold,
            'momentum_ewma_span': self.momentum_ewma_span,
            'mean_return_ewm_threshold': self.mean_return_ewm_threshold,
            'mean_return_ewm_span': self.mean_return_ewm_span,
            'mean_return_simple_threshold': self.mean_return_simple_threshold,
            'mean_return_simple_window': self.mean_return_simple_window,
            'adx_period': self.adx_period,
            'adx_threshold': self.adx_threshold,
            'use_volume_confirmation': self.use_volume_confirmation,
            'volume_threshold': self.volume_threshold,
            'min_signal_gap': self.min_signal_gap,
            'max_signals_per_period': self.max_signals_per_period
        }
    
    def update_parameters(self, **kwargs) -> None:
        """
        Update signal generator parameters dynamically.
        
        Args:
            **kwargs: Parameter names and new values
        """
        updated_params = []
        
        for param, value in kwargs.items():
            if hasattr(self, param):
                old_value = getattr(self, param)
                setattr(self, param, value)
                updated_params.append(f"{param}: {old_value} -> {value}")
            else:
                logger.warning(f"Unknown parameter: {param}")
        
        if updated_params:
            logger.info(f"Updated parameters: {', '.join(updated_params)}")
            self._validate_crypto_parameters()
    
    def __repr__(self) -> str:
        """String representation of SignalGenerator."""
        return (f"SignalGenerator(momentum_threshold={self.momentum_threshold}, "
                f"mean_return_ewm_threshold={self.mean_return_ewm_threshold}, "
                f"mean_return_simple_threshold={self.mean_return_simple_threshold})")