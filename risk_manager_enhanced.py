# crypto_momentum_backtest/risk/risk_manager_enhanced.py
"""Enhanced risk management with regime detection and adaptive limits."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class MarketRegime:
    """Market regime information."""
    name: str  # 'trending', 'volatile', 'ranging', 'crisis'
    volatility: float
    trend_strength: float
    correlation_level: float
    risk_multiplier: float


class EnhancedRiskManager:
    """
    Enhanced risk management system with regime detection,
    adaptive limits, and trailing stops.
    """
    
    def __init__(
        self,
        max_drawdown: float = 0.20,  # Reduced from 0.25 for better protection
        max_position_size: float = 0.25,  # Increased from 0.20
        max_correlation: float = 0.7,  # Reduced from 0.8
        max_exchange_exposure: float = 0.40,
        volatility_scale_threshold: float = 1.5,
        use_trailing_stops: bool = True,
        trailing_stop_atr_multiplier: float = 2.5,
        regime_lookback: int = 60,
        crisis_vol_threshold: float = 1.0,  # 100% annual vol = crisis
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize enhanced risk manager.
        
        Args:
            max_drawdown: Maximum allowed drawdown
            max_position_size: Maximum position size
            max_correlation: Maximum correlation between positions
            max_exchange_exposure: Maximum exposure per exchange
            volatility_scale_threshold: Volatility regime threshold
            use_trailing_stops: Whether to use trailing stops
            trailing_stop_atr_multiplier: ATR multiplier for trailing stops
            regime_lookback: Lookback period for regime detection
            crisis_vol_threshold: Volatility threshold for crisis detection
            logger: Logger instance
        """
        self.max_drawdown = max_drawdown
        self.max_position_size = max_position_size
        self.max_correlation = max_correlation
        self.max_exchange_exposure = max_exchange_exposure
        self.volatility_scale_threshold = volatility_scale_threshold
        self.use_trailing_stops = use_trailing_stops
        self.trailing_stop_atr_multiplier = trailing_stop_atr_multiplier
        self.regime_lookback = regime_lookback
        self.crisis_vol_threshold = crisis_vol_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        # Risk state tracking
        self.risk_state = {
            'current_drawdown': 0.0,
            'peak_value': 0.0,
            'risk_on': True,
            'volatility_regime': 'normal',
            'exposure_level': 1.0,
            'market_regime': MarketRegime('normal', 0.5, 0.0, 0.5, 1.0),
            'trailing_stops': {}
        }
        
        # Regime-specific parameters
        self.regime_params = {
            'trending': {
                'exposure_multiplier': 1.5,  # Increase exposure in trends
                'stop_loss_multiplier': 1.2,  # Wider stops in trends
                'correlation_threshold': 0.8   # Allow higher correlation
            },
            'volatile': {
                'exposure_multiplier': 0.5,   # Reduce exposure
                'stop_loss_multiplier': 0.8,  # Tighter stops
                'correlation_threshold': 0.6   # Lower correlation tolerance
            },
            'ranging': {
                'exposure_multiplier': 0.7,   # Moderate exposure
                'stop_loss_multiplier': 1.0,  # Normal stops
                'correlation_threshold': 0.7   # Normal correlation
            },
            'crisis': {
                'exposure_multiplier': 0.3,   # Minimal exposure
                'stop_loss_multiplier': 0.6,  # Very tight stops
                'correlation_threshold': 0.5   # Low correlation tolerance
            }
        }
    
    def detect_market_regime(
        self,
        returns: pd.Series,
        prices: Optional[pd.DataFrame] = None,
        correlations: Optional[pd.DataFrame] = None
    ) -> MarketRegime:
        """
        Detect current market regime using multiple indicators.
        
        Args:
            returns: Portfolio returns
            prices: Asset prices for trend detection
            correlations: Asset correlation matrix
            
        Returns:
            Current market regime
        """
        # Calculate key metrics
        recent_returns = returns.iloc[-self.regime_lookback:]
        volatility = recent_returns.std() * np.sqrt(252)
        
        # Trend strength (returns vs volatility)
        trend_strength = recent_returns.mean() / recent_returns.std() if recent_returns.std() > 0 else 0
        trend_strength = trend_strength * np.sqrt(252)  # Annualized
        
        # Average correlation
        if correlations is not None and len(correlations) > 0:
            # Get upper triangle of correlation matrix
            mask = np.triu(np.ones_like(correlations), k=1).astype(bool)
            avg_correlation = correlations.where(mask).stack().mean()
        else:
            avg_correlation = 0.5  # Default
        
        # Determine regime
        if volatility > self.crisis_vol_threshold:
            regime_name = 'crisis'
            risk_multiplier = self.regime_params['crisis']['exposure_multiplier']
        elif abs(trend_strength) > 1.0 and volatility < 0.8:
            regime_name = 'trending'
            risk_multiplier = self.regime_params['trending']['exposure_multiplier']
        elif volatility > 0.8:
            regime_name = 'volatile'
            risk_multiplier = self.regime_params['volatile']['exposure_multiplier']
        else:
            regime_name = 'ranging'
            risk_multiplier = self.regime_params['ranging']['exposure_multiplier']
        
        regime = MarketRegime(
            name=regime_name,
            volatility=volatility,
            trend_strength=trend_strength,
            correlation_level=avg_correlation,
            risk_multiplier=risk_multiplier
        )
        
        self.logger.info(
            f"Market regime: {regime_name} "
            f"(vol={volatility:.1%}, trend={trend_strength:.2f}, "
            f"corr={avg_correlation:.2f})"
        )
        
        return regime
    
    def update_risk_state(
        self,
        portfolio_value: float,
        returns: pd.Series,
        positions: pd.DataFrame,
        prices: Optional[pd.DataFrame] = None,
        correlations: Optional[pd.DataFrame] = None
    ):
        """
        Update current risk state with regime detection.
        
        Args:
            portfolio_value: Current portfolio value
            returns: Recent returns
            positions: Current positions
            prices: Asset prices
            correlations: Asset correlations
        """
        # Update drawdown
        if portfolio_value > self.risk_state['peak_value']:
            self.risk_state['peak_value'] = portfolio_value
        
        self.risk_state['current_drawdown'] = (
            portfolio_value - self.risk_state['peak_value']
        ) / self.risk_state['peak_value']
        
        # Detect market regime
        regime = self.detect_market_regime(returns, prices, correlations)
        self.risk_state['market_regime'] = regime
        
        # Update exposure based on regime and drawdown
        base_exposure = regime.risk_multiplier
        
        # Reduce exposure during drawdowns
        if abs(self.risk_state['current_drawdown']) > 0.1:
            drawdown_multiplier = 1 - (abs(self.risk_state['current_drawdown']) / self.max_drawdown)
            drawdown_multiplier = max(0.3, drawdown_multiplier)  # Minimum 30% exposure
            base_exposure *= drawdown_multiplier
        
        self.risk_state['exposure_level'] = base_exposure
        
        # Update trailing stops if enabled
        if self.use_trailing_stops and prices is not None:
            self._update_trailing_stops(positions, prices)
        
        # Risk-off conditions
        if abs(self.risk_state['current_drawdown']) > self.max_drawdown:
            self.risk_state['risk_on'] = False
            self.logger.warning(
                f"Risk-off triggered: drawdown {self.risk_state['current_drawdown']:.2%}"
            )
        elif abs(self.risk_state['current_drawdown']) < self.max_drawdown * 0.5:
            self.risk_state['risk_on'] = True
    
    def _update_trailing_stops(
        self,
        positions: pd.DataFrame,
        prices: pd.DataFrame
    ):
        """Update trailing stop levels for active positions."""
        current_positions = positions.iloc[-1]
        current_prices = prices.iloc[-1] if isinstance(prices, pd.DataFrame) else prices
        
        # Import technical indicators for ATR
        from ..signals.technical_indicators import TechnicalIndicators
        
        for symbol in current_positions[current_positions != 0].index:
            position = current_positions[symbol]
            
            if symbol in prices.columns:
                # Calculate ATR
                atr = TechnicalIndicators.atr(prices[symbol].to_frame(), period=14).iloc[-1]
                
                # Calculate stop distance based on regime
                regime_multiplier = self.regime_params[
                    self.risk_state['market_regime'].name
                ]['stop_loss_multiplier']
                
                stop_distance = atr * self.trailing_stop_atr_multiplier * regime_multiplier
                
                # Update trailing stop
                current_price = current_prices[symbol]
                
                if position > 0:  # Long position
                    stop_price = current_price - stop_distance
                    
                    # Only update if higher than previous stop
                    if symbol not in self.risk_state['trailing_stops'] or \
                       stop_price > self.risk_state['trailing_stops'][symbol]:
                        self.risk_state['trailing_stops'][symbol] = stop_price
                
                elif position < 0:  # Short position
                    stop_price = current_price + stop_distance
                    
                    # Only update if lower than previous stop
                    if symbol not in self.risk_state['trailing_stops'] or \
                       stop_price < self.risk_state['trailing_stops'][symbol]:
                        self.risk_state['trailing_stops'][symbol] = stop_price
    
    def check_stop_losses(
        self,
        positions: pd.DataFrame,
        prices: pd.DataFrame
    ) -> Dict[str, bool]:
        """
        Check if any positions hit their stop losses.
        
        Args:
            positions: Current positions
            prices: Current prices
            
        Returns:
            Dictionary of symbol -> should_exit
        """
        exits = {}
        current_positions = positions.iloc[-1]
        current_prices = prices.iloc[-1] if isinstance(prices, pd.DataFrame) else prices
        
        for symbol, position in current_positions[current_positions != 0].items():
            if symbol in self.risk_state['trailing_stops'] and symbol in current_prices:
                stop_price = self.risk_state['trailing_stops'][symbol]
                current_price = current_prices[symbol]
                
                if position > 0 and current_price <= stop_price:
                    exits[symbol] = True
                    self.logger.info(f"Stop loss triggered for {symbol} (long): "
                                   f"price {current_price:.2f} <= stop {stop_price:.2f}")
                elif position < 0 and current_price >= stop_price:
                    exits[symbol] = True
                    self.logger.info(f"Stop loss triggered for {symbol} (short): "
                                   f"price {current_price:.2f} >= stop {stop_price:.2f}")
                else:
                    exits[symbol] = False
            else:
                exits[symbol] = False
        
        return exits
    
    def check_position_limits(
        self,
        proposed_positions: pd.Series,
        current_prices: pd.Series,
        portfolio_value: float
    ) -> pd.Series:
        """
        Check and enforce position limits with regime adjustments.
        
        Args:
            proposed_positions: Proposed position sizes
            current_prices: Current asset prices
            portfolio_value: Total portfolio value
            
        Returns:
            Adjusted positions
        """
        adjusted_positions = proposed_positions.copy()
        
        # Get regime-adjusted max position size
        regime = self.risk_state['market_regime']
        regime_params = self.regime_params.get(regime.name, {})
        
        # Adjust max position size based on regime
        if regime.name == 'trending':
            max_size = min(self.max_position_size * 1.2, 0.30)  # Allow up to 30% in trends
        elif regime.name in ['volatile', 'crisis']:
            max_size = self.max_position_size * 0.7  # Reduce in volatile/crisis
        else:
            max_size = self.max_position_size
        
        # Calculate position values
        position_values = adjusted_positions * current_prices
        position_weights = position_values / portfolio_value
        
        # Enforce maximum position size
        oversized = position_weights.abs() > max_size
        if oversized.any():
            for symbol in position_weights[oversized].index:
                scale = max_size / abs(position_weights[symbol])
                adjusted_positions[symbol] *= scale
                
            self.logger.info(
                f"Scaled down {oversized.sum()} oversized positions "
                f"(max size: {max_size:.1%} in {regime.name} regime)"
            )
        
        return adjusted_positions
    
    def check_correlation_risk(
        self,
        positions: pd.DataFrame,
        returns: pd.DataFrame,
        lookback: int = 60
    ) -> Dict[str, List[str]]:
        """
        Check for correlation risk with regime-based thresholds.
        
        Args:
            positions: Current positions
            returns: Historical returns
            lookback: Correlation lookback period
            
        Returns:
            Dictionary of high correlation pairs
        """
        # Get active positions
        active_positions = positions.columns[positions.iloc[-1] != 0]
        
        if len(active_positions) < 2:
            return {}
        
        # Calculate correlation matrix
        corr_matrix = returns[active_positions].iloc[-lookback:].corr()
        
        # Get regime-based correlation threshold
        regime = self.risk_state['market_regime']
        corr_threshold = self.regime_params.get(
            regime.name, {}
        ).get('correlation_threshold', self.max_correlation)
        
        # Find high correlations
        high_corr_pairs = {}
        
        for i, sym1 in enumerate(active_positions):
            for j, sym2 in enumerate(active_positions):
                if i < j and abs(corr_matrix.loc[sym1, sym2]) > corr_threshold:
                    pair_key = f"{sym1}-{sym2}"
                    high_corr_pairs[pair_key] = [sym1, sym2]
                    
                    self.logger.warning(
                        f"High correlation in {regime.name} regime: "
                        f"{sym1}-{sym2} = {corr_matrix.loc[sym1, sym2]:.2f} "
                        f"(threshold: {corr_threshold:.2f})"
                    )
        
        return high_corr_pairs
    
    def calculate_regime_adjusted_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate regime-adjusted Value at Risk.
        
        Args:
            returns: Portfolio returns
            confidence_level: VaR confidence level
            
        Returns:
            Regime-adjusted VaR
        """
        # Base VaR calculation
        historical_returns = returns.iloc[-252:]  # 1 year
        var_percentile = (1 - confidence_level) * 100
        base_var = np.percentile(historical_returns, var_percentile)
        
        # Adjust for current regime
        regime = self.risk_state['market_regime']
        
        if regime.name == 'crisis':
            # Assume 2x worse in crisis
            adjusted_var = base_var * 2.0
        elif regime.name == 'volatile':
            # 1.5x worse in volatile regime
            adjusted_var = base_var * 1.5
        elif regime.name == 'trending':
            # Slightly better in trending regime
            adjusted_var = base_var * 0.8
        else:
            adjusted_var = base_var
        
        return adjusted_var
    
    def generate_risk_report(
        self,
        portfolio_value: float,
        returns: pd.Series,
        positions: pd.DataFrame
    ) -> Dict:
        """
        Generate comprehensive risk report with regime information.
        
        Args:
            portfolio_value: Current portfolio value
            returns: Portfolio returns
            positions: Current positions
            
        Returns:
            Risk metrics dictionary
        """
        report = {
            'timestamp': datetime.now(),
            'portfolio_value': portfolio_value,
            'risk_state': self.risk_state.copy()
        }
        
        # Market regime information
        regime = self.risk_state['market_regime']
        report['market_regime'] = {
            'name': regime.name,
            'volatility': regime.volatility,
            'trend_strength': regime.trend_strength,
            'correlation_level': regime.correlation_level,
            'risk_multiplier': regime.risk_multiplier,
            'exposure_adjustment': self.risk_state['exposure_level']
        }
        
        # Calculate risk metrics
        report['metrics'] = {
            'current_drawdown': self.risk_state['current_drawdown'],
            'var_95': self.calculate_var(returns, 0.95),
            'regime_adjusted_var_95': self.calculate_regime_adjusted_var(returns, 0.95),
            'expected_shortfall_95': self.calculate_expected_shortfall(returns, 0.95),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'max_drawdown_30d': self.calculate_max_drawdown(returns.iloc[-30:])
        }
        
        # Position concentration
        position_values = positions.iloc[-1].abs()
        total_exposure = position_values.sum()
        
        report['concentration'] = {
            'num_positions': (position_values > 0).sum(),
            'total_exposure': total_exposure / portfolio_value,
            'largest_position': position_values.max() / portfolio_value,
            'herfindahl_index': (position_values / total_exposure).pow(2).sum() if total_exposure > 0 else 0
        }
        
        # Trailing stops status
        if self.use_trailing_stops:
            report['trailing_stops'] = {
                'active_stops': len(self.risk_state['trailing_stops']),
                'stops': self.risk_state['trailing_stops'].copy()
            }
        
        return report
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk."""
        var_percentile = (1 - confidence_level) * 100
        return np.percentile(returns.dropna(), var_percentile)
    
    @staticmethod
    def calculate_expected_shortfall(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (CVaR)."""
        var = RiskManager.calculate_var(returns, confidence_level)
        tail_returns = returns[returns <= var]
        return tail_returns.mean() if len(tail_returns) > 0 else var
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        
        if downside_std > 0:
            return np.sqrt(252) * excess_returns.mean() / downside_std
        else:
            return np.inf if excess_returns.mean() > 0 else 0
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()