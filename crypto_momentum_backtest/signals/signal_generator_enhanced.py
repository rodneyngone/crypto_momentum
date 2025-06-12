#!/usr/bin/env python3
"""
Enhanced Signal Generator with momentum scoring, multi-timeframe analysis,
and adaptive parameters for improved crypto trading performance.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Enumeration of available signal types."""
    MOMENTUM = "momentum"
    MOMENTUM_SCORE = "momentum_score"
    DUAL_MOMENTUM = "dual_momentum"
    MULTI_TIMEFRAME = "multi_timeframe"
    MEAN_RETURN_EWM = "mean_return_ewm"
    MEAN_RETURN_SIMPLE = "mean_return_simple"
    ENSEMBLE = "ensemble"


@dataclass
class SignalMetrics:
    """Container for signal generation metrics."""
    total_signals: int
    long_signals: int
    short_signals: int
    signal_frequency: float
    avg_score: float
    strategy_type: str


class EnhancedSignalGenerator:
    """
    Enhanced signal generator with momentum scoring and multi-timeframe analysis.
    """
    
    def __init__(self,
                 # Multi-timeframe ADX parameters
                 adx_periods: List[int] = [7, 14, 21],
                 adx_threshold: float = 15.0,  # Lowered from 25
                 
                 # Adaptive EWMA parameters
                 base_ewma_span: int = 20,
                 adaptive_ewma: bool = True,
                 
                 # Momentum scoring weights
                 momentum_weights: Dict[str, float] = None,
                 
                 # Dual momentum parameters
                 absolute_momentum_threshold: float = 0.05,
                 relative_momentum_threshold: float = 0.0,
                 momentum_lookback: int = 30,
                 
                 # Volume confirmation
                 use_volume_confirmation: bool = True,
                 volume_threshold: float = 1.2,
                 
                 # Technical indicators for scoring
                 rsi_period: int = 14,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 
                 # Risk parameters
                 max_correlation: float = 0.7,
                 min_score_threshold: float = 0.3,
                 
                 **kwargs):
        """Initialize enhanced signal generator with improved parameters."""
        
        # Multi-timeframe ADX
        self.adx_periods = adx_periods
        self.adx_threshold = adx_threshold
        
        # Adaptive EWMA
        self.base_ewma_span = base_ewma_span
        self.adaptive_ewma = adaptive_ewma
        
        # Momentum scoring weights
        self.momentum_weights = momentum_weights or {
            'price': 0.4,
            'volume': 0.2,
            'rsi': 0.2,
            'macd': 0.2
        }
        
        # Dual momentum
        self.absolute_momentum_threshold = absolute_momentum_threshold
        self.relative_momentum_threshold = relative_momentum_threshold
        self.momentum_lookback = momentum_lookback
        
        # Volume confirmation
        self.use_volume_confirmation = use_volume_confirmation
        self.volume_threshold = volume_threshold
        
        # Technical indicators
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        
        # Risk parameters
        self.max_correlation = max_correlation
        self.min_score_threshold = min_score_threshold
        
        # Store any additional parameters
        self.__dict__.update(kwargs)
        
        # Metrics tracking
        self._signal_metrics: Dict[str, SignalMetrics] = {}
        
        logger.info("Enhanced Signal Generator initialized with crypto-optimized parameters")
    
    def generate_signals(self, 
                        data: pd.DataFrame,
                        signal_type: Union[str, SignalType] = SignalType.MOMENTUM_SCORE,
                        symbol: Optional[str] = None,
                        universe_data: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generate trading signals based on specified strategy.
        
        Args:
            data: OHLCV DataFrame with required columns
            signal_type: Type of signal strategy to use
            symbol: Symbol identifier for logging
            universe_data: Full universe data for relative momentum
            
        Returns:
            Signal series: values between -1 and 1 for momentum scoring,
                          or -1, 0, 1 for discrete signals
        """
        try:
            # Convert signal type to enum
            if isinstance(signal_type, str):
                signal_type = SignalType(signal_type)
            
            # Calculate adaptive parameters if enabled
            if self.adaptive_ewma:
                volatility = self._calculate_volatility(data)
                self.current_ewma_span = self._adaptive_ewma_span(volatility)
            else:
                self.current_ewma_span = self.base_ewma_span
            
            # Generate signals based on type
            if signal_type == SignalType.MOMENTUM_SCORE:
                signals = self._generate_momentum_score_signals(data)
            elif signal_type == SignalType.MULTI_TIMEFRAME:
                signals = self._generate_multi_timeframe_signals(data)
            elif signal_type == SignalType.DUAL_MOMENTUM:
                signals = self._generate_dual_momentum_signals(data, universe_data)
            elif signal_type == SignalType.ENSEMBLE:
                signals = self._generate_ensemble_signals(data, universe_data)
            elif signal_type == SignalType.MOMENTUM:
                signals = self._generate_momentum_signals(data)
            elif signal_type == SignalType.MEAN_RETURN_EWM:
                signals = self._generate_mean_return_ewm_signals(data)
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
            logger.info(f"Generated {signal_type.value} signals{symbol_str}: {metrics.total_signals} signals")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating {signal_type} signals: {e}")
            return pd.Series(0, index=data.index)
    
    def _calculate_volatility(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calculate recent volatility for adaptive parameters."""
        returns = data['close'].pct_change()
        return returns.rolling(period).std().iloc[-1] * np.sqrt(365)
    
    def _adaptive_ewma_span(self, volatility: float) -> int:
        """
        Calculate adaptive EWMA span based on volatility.
        Higher volatility = longer span for stability.
        """
        vol_multiplier = min(2.0, max(0.5, volatility / 0.6))  # Normalize to 60% annual vol
        return int(self.base_ewma_span * vol_multiplier)
    
    def _generate_momentum_score_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate composite momentum score signals.
        Returns continuous scores between -1 and 1.
        """
        scores = {}
        
        # Price momentum (40% weight)
        price_momentum = (data['close'] / data['close'].shift(self.momentum_lookback) - 1)
        scores['price'] = np.clip(price_momentum / 0.2, -1, 1)  # Normalize to [-1, 1]
        
        # Volume momentum (20% weight)
        volume_ma = data['volume'].rolling(20).mean()
        volume_ratio = data['volume'] / volume_ma
        scores['volume'] = np.clip((volume_ratio - 1) / 0.5, -1, 1)
        
        # RSI momentum (20% weight)
        rsi = self._calculate_rsi(data['close'], self.rsi_period)
        scores['rsi'] = (rsi - 50) / 50  # Normalize to [-1, 1]
        
        # MACD momentum (20% weight)
        macd_line, signal_line, histogram = self._calculate_macd(data['close'])
        macd_momentum = histogram / data['close'] * 1000  # Normalize
        scores['macd'] = np.clip(macd_momentum, -1, 1)
        
        # Calculate weighted composite score
        composite_score = pd.Series(0.0, index=data.index)
        for component, weight in self.momentum_weights.items():
            if component in scores:
                composite_score += scores[component].fillna(0) * weight
        
        # Convert to signals based on thresholds
        signals = pd.Series(0.0, index=data.index)
        signals[composite_score > self.min_score_threshold] = composite_score[composite_score > self.min_score_threshold]
        signals[composite_score < -self.min_score_threshold] = composite_score[composite_score < -self.min_score_threshold]
        
        return signals
    
    def _generate_multi_timeframe_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals using multiple ADX timeframes with majority voting.
        """
        from .technical_indicators import TechnicalIndicators
        
        signals_list = []
        
        for period in self.adx_periods:
            # Calculate ADX for this period
            adx = TechnicalIndicators.adx(data, period)
            
            # Calculate directional movement
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            # Price direction signal
            ewma_fast = data['close'].ewm(span=period).mean()
            ewma_slow = data['close'].ewm(span=period * 2).mean()
            
            # Generate signal for this timeframe
            timeframe_signal = pd.Series(0, index=data.index)
            
            # Long signal: ADX > threshold and price above slow EMA
            long_condition = (adx > self.adx_threshold) & (ewma_fast > ewma_slow) & (data['close'] > ewma_slow)
            timeframe_signal[long_condition] = 1
            
            # Short signal: ADX > threshold and price below slow EMA
            short_condition = (adx > self.adx_threshold) & (ewma_fast < ewma_slow) & (data['close'] < ewma_slow)
            timeframe_signal[short_condition] = -1
            
            signals_list.append(timeframe_signal)
        
        # Majority voting across timeframes
        signals_df = pd.DataFrame(signals_list).T
        majority_signals = signals_df.mean(axis=1)
        
        # Convert to discrete signals
        final_signals = pd.Series(0, index=data.index)
        final_signals[majority_signals > 0.5] = 1
        final_signals[majority_signals < -0.5] = -1
        
        return final_signals
    
    def _generate_dual_momentum_signals(self, data: pd.DataFrame, 
                                       universe_data: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generate dual momentum signals combining absolute and relative momentum.
        """
        # Absolute momentum: asset vs its own history
        abs_momentum = (data['close'] / data['close'].shift(self.momentum_lookback) - 1)
        
        # Relative momentum: asset vs universe median (if universe data provided)
        if universe_data is not None and len(universe_data) > 0:
            # Calculate universe median returns
            universe_returns = universe_data.pct_change()
            universe_median = universe_returns.median(axis=1)
            
            # Asset returns
            asset_returns = data['close'].pct_change()
            
            # Relative momentum
            rel_momentum = asset_returns.rolling(self.momentum_lookback).mean() - \
                          universe_median.rolling(self.momentum_lookback).mean()
        else:
            # If no universe data, use zero threshold
            rel_momentum = pd.Series(0, index=data.index)
        
        # Generate signals: both momentums must be positive for long, both negative for short
        signals = pd.Series(0, index=data.index)
        
        long_condition = (abs_momentum > self.absolute_momentum_threshold) & \
                        (rel_momentum > self.relative_momentum_threshold)
        short_condition = (abs_momentum < -self.absolute_momentum_threshold) & \
                         (rel_momentum < -self.relative_momentum_threshold)
        
        signals[long_condition] = 1
        signals[short_condition] = -1
        
        return signals
    
    def _generate_ensemble_signals(self, data: pd.DataFrame, 
                                  universe_data: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generate ensemble signals combining multiple strategies.
        """
        strategies = {
            'momentum_score': self._generate_momentum_score_signals(data),
            'multi_timeframe': self._generate_multi_timeframe_signals(data),
            'dual_momentum': self._generate_dual_momentum_signals(data, universe_data)
        }
        
        # Calculate recent performance for each strategy (mock implementation)
        # In production, this would use actual backtested returns
        strategy_weights = {
            'momentum_score': 0.4,
            'multi_timeframe': 0.3,
            'dual_momentum': 0.3
        }
        
        # Weighted average of strategies
        ensemble_signal = pd.Series(0.0, index=data.index)
        for strategy, signal in strategies.items():
            weight = strategy_weights.get(strategy, 0.33)
            ensemble_signal += signal * weight
        
        # Discretize if needed
        final_signals = pd.Series(0, index=data.index)
        final_signals[ensemble_signal > 0.3] = 1
        final_signals[ensemble_signal < -0.3] = -1
        
        return final_signals
    
    def _generate_momentum_signals(self, data: pd.DataFrame) -> pd.Series:
        """Classic momentum signals (backward compatibility)."""
        ewma = data['close'].ewm(span=self.current_ewma_span).mean()
        momentum = (data['close'] / ewma - 1)
        
        signals = pd.Series(0, index=data.index, dtype=int)
        signals[momentum > 0.01] = 1
        signals[momentum < -0.01] = -1
        
        return signals
    
    def _generate_mean_return_ewm_signals(self, data: pd.DataFrame) -> pd.Series:
        """EWM mean return signals (backward compatibility)."""
        returns = data['close'].pct_change().fillna(0)
        ewm_returns = returns.ewm(span=10).mean()
        
        signals = pd.Series(0, index=data.index, dtype=int)
        signals[ewm_returns > 0.015] = 1
        signals[ewm_returns < -0.015] = -1
        
        return signals
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.macd_slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _apply_volume_confirmation(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Apply volume confirmation to filter signals."""
        volume_ma = data['volume'].rolling(window=20, min_periods=1).mean()
        volume_ratio = data['volume'] / volume_ma
        
        # Only keep signals with above-average volume
        volume_confirmed = volume_ratio > self.volume_threshold
        confirmed_signals = signals.copy()
        confirmed_signals[~volume_confirmed] = 0
        
        return confirmed_signals
    
    def _apply_signal_filtering(self, signals: pd.Series) -> pd.Series:
        """Apply additional signal filtering rules."""
        filtered_signals = signals.copy()
        
        # Remove signals that are too close together (within 2 days)
        signal_positions = np.where(signals != 0)[0]
        
        for i in range(1, len(signal_positions)):
            if signal_positions[i] - signal_positions[i-1] < 2:
                filtered_signals.iloc[signal_positions[i]] = 0
        
        return filtered_signals
    
    def _calculate_signal_metrics(self, signals: pd.Series, strategy_type: str) -> SignalMetrics:
        """Calculate comprehensive metrics for signal generation performance."""
        # Handle both discrete and continuous signals
        if signals.dtype == float and ((signals > 1) | (signals < -1)).any():
            # Continuous scores
            long_signals = (signals > 0).sum()
            short_signals = (signals < 0).sum()
            avg_score = signals[signals != 0].mean() if (signals != 0).any() else 0
        else:
            # Discrete signals
            long_signals = (signals == 1).sum()
            short_signals = (signals == -1).sum()
            avg_score = 0
        
        total_signals = long_signals + short_signals
        signal_frequency = (total_signals / len(signals)) * 100 if len(signals) > 0 else 0
        
        return SignalMetrics(
            total_signals=total_signals,
            long_signals=long_signals,
            short_signals=short_signals,
            signal_frequency=signal_frequency,
            avg_score=avg_score,
            strategy_type=strategy_type
        )
    
    def get_signal_strength(self, data: pd.DataFrame, signal_type: SignalType = SignalType.MOMENTUM_SCORE) -> pd.Series:
        """
        Get signal strength scores for position sizing.
        Returns values between 0 and 1 indicating signal strength.
        """
        signals = self.generate_signals(data, signal_type)
        
        # For continuous signals, use absolute value
        if signals.dtype == float:
            return signals.abs()
        else:
            # For discrete signals, return 1 for any signal
            return (signals != 0).astype(float)
    
    def calculate_correlation_filter(self, returns: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Filter signals based on correlation to avoid concentration risk.
        """
        # Calculate rolling correlation
        corr_matrix = returns.rolling(60).corr()
        
        # For each timestamp, check if any assets are too correlated
        filtered_signals = signals.copy()
        
        for idx in signals.index:
            if idx in corr_matrix.index:
                active_signals = signals.loc[idx][signals.loc[idx] != 0]
                
                if len(active_signals) > 1:
                    # Check correlations between active positions
                    for i, asset1 in enumerate(active_signals.index):
                        for asset2 in active_signals.index[i+1:]:
                            if abs(corr_matrix.loc[idx, asset1, asset2]) > self.max_correlation:
                                # Keep the one with stronger signal
                                if abs(active_signals[asset1]) < abs(active_signals[asset2]):
                                    filtered_signals.loc[idx, asset1] = 0
                                else:
                                    filtered_signals.loc[idx, asset2] = 0
        
        return filtered_signals