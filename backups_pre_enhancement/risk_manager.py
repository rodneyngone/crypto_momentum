# crypto_momentum_backtest/risk/risk_manager.py
"""Risk management system for portfolio protection."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta


class RiskManager:
    """
    Comprehensive risk management system.
    """
    
    def __init__(
        self,
        max_drawdown: float = 0.15,
        max_position_size: float = 0.20,
        max_correlation: float = 0.8,
        max_exchange_exposure: float = 0.40,
        volatility_scale_threshold: float = 1.5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize risk manager.
        
        Args:
            max_drawdown: Maximum allowed drawdown
            max_position_size: Maximum position size
            max_correlation: Maximum correlation between positions
            max_exchange_exposure: Maximum exposure per exchange
            volatility_scale_threshold: Volatility regime threshold
            logger: Logger instance
        """
        self.max_drawdown = max_drawdown
        self.max_position_size = max_position_size
        self.max_correlation = max_correlation
        self.max_exchange_exposure = max_exchange_exposure
        self.volatility_scale_threshold = volatility_scale_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        # Risk state tracking
        self.risk_state = {
            'current_drawdown': 0.0,
            'peak_value': 0.0,
            'risk_on': True,
            'volatility_regime': 'normal',
            'exposure_level': 1.0
        }
        
    def update_risk_state(
        self,
        portfolio_value: float,
        returns: pd.Series,
        positions: pd.DataFrame
    ):
        """
        Update current risk state.
        
        Args:
            portfolio_value: Current portfolio value
            returns: Recent returns
            positions: Current positions
        """
        # Update drawdown
        if portfolio_value > self.risk_state['peak_value']:
            self.risk_state['peak_value'] = portfolio_value
        
        self.risk_state['current_drawdown'] = (
            portfolio_value - self.risk_state['peak_value']
        ) / self.risk_state['peak_value']
        
        # Check volatility regime
        current_vol = returns.iloc[-30:].std() * np.sqrt(252)
        long_term_vol = returns.iloc[-90:].std() * np.sqrt(252)
        
        if current_vol > long_term_vol * self.volatility_scale_threshold:
            self.risk_state['volatility_regime'] = 'high'
            self.risk_state['exposure_level'] = 0.5
        else:
            self.risk_state['volatility_regime'] = 'normal'
            self.risk_state['exposure_level'] = 1.0
        
        # Check if risk-off
        if abs(self.risk_state['current_drawdown']) > self.max_drawdown:
            self.risk_state['risk_on'] = False
            self.logger.warning(
                f"Risk-off triggered: drawdown {self.risk_state['current_drawdown']:.2%}"
            )
        elif abs(self.risk_state['current_drawdown']) < self.max_drawdown * 0.5:
            self.risk_state['risk_on'] = True
    
    def check_position_limits(
        self,
        proposed_positions: pd.Series,
        current_prices: pd.Series,
        portfolio_value: float
    ) -> pd.Series:
        """
        Check and enforce position limits.
        
        Args:
            proposed_positions: Proposed position sizes
            current_prices: Current asset prices
            portfolio_value: Total portfolio value
            
        Returns:
            Adjusted positions
        """
        adjusted_positions = proposed_positions.copy()
        
        # Calculate position values
        position_values = adjusted_positions * current_prices
        position_weights = position_values / portfolio_value
        
        # Enforce maximum position size
        oversized = position_weights.abs() > self.max_position_size
        if oversized.any():
            for symbol in position_weights[oversized].index:
                scale = self.max_position_size / abs(position_weights[symbol])
                adjusted_positions[symbol] *= scale
                
            self.logger.info(
                f"Scaled down {oversized.sum()} oversized positions"
            )
        
        return adjusted_positions
    
    def check_correlation_risk(
        self,
        positions: pd.DataFrame,
        returns: pd.DataFrame,
        lookback: int = 60
    ) -> Dict[str, List[str]]:
        """
        Check for high correlation between positions.
        
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
        
        # Find high correlations
        high_corr_pairs = {}
        
        for i, sym1 in enumerate(active_positions):
            for j, sym2 in enumerate(active_positions):
                if i < j and abs(corr_matrix.loc[sym1, sym2]) > self.max_correlation:
                    pair_key = f"{sym1}-{sym2}"
                    high_corr_pairs[pair_key] = [sym1, sym2]
                    
                    self.logger.warning(
                        f"High correlation detected: {sym1}-{sym2} = "
                        f"{corr_matrix.loc[sym1, sym2]:.2f}"
                    )
        
        return high_corr_pairs
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        lookback: int = 252
    ) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            returns: Portfolio returns
            confidence_level: VaR confidence level
            lookback: Historical lookback period
            
        Returns:
            VaR estimate
        """
        historical_returns = returns.iloc[-lookback:]
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(historical_returns, var_percentile)
        
        return var
    
    def calculate_expected_shortfall(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        lookback: int = 252
    ) -> float:
        """
        Calculate Expected Shortfall (CVaR).
        
        Args:
            returns: Portfolio returns
            confidence_level: Confidence level
            lookback: Historical lookback period
            
        Returns:
            Expected shortfall
        """
        var = self.calculate_var(returns, confidence_level, lookback)
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) > 0:
            return tail_returns.mean()
        else:
            return var
    
    def check_exchange_concentration(
        self,
        positions: pd.DataFrame,
        exchange_mapping: Dict[str, str],
        portfolio_value: float
    ) -> Dict[str, float]:
        """
        Check concentration by exchange.
        
        Args:
            positions: Current positions
            exchange_mapping: Symbol to exchange mapping
            portfolio_value: Total portfolio value
            
        Returns:
            Exchange exposure percentages
        """
        exchange_exposure = {}
        
        for exchange in set(exchange_mapping.values()):
            exchange_symbols = [
                sym for sym, exch in exchange_mapping.items()
                if exch == exchange and sym in positions.columns
            ]
            
            if exchange_symbols:
                exchange_value = positions[exchange_symbols].iloc[-1].abs().sum()
                exchange_pct = exchange_value / portfolio_value
                exchange_exposure[exchange] = exchange_pct
                
                if exchange_pct > self.max_exchange_exposure:
                    self.logger.warning(
                        f"High exchange concentration: {exchange} = {exchange_pct:.2%}"
                    )
        
        return exchange_exposure
    
    def generate_risk_report(
        self,
        portfolio_value: float,
        returns: pd.Series,
        positions: pd.DataFrame
    ) -> Dict:
        """
        Generate comprehensive risk report.
        
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
        
        # Calculate risk metrics
        report['metrics'] = {
            'current_drawdown': self.risk_state['current_drawdown'],
            'var_95': self.calculate_var(returns, 0.95),
            'var_99': self.calculate_var(returns, 0.99),
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
            'herfindahl_index': (position_values / total_exposure).pow(2).sum()
        }
        
        return report
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        
        if downside_std > 0:
            return np.sqrt(252) * excess_returns.mean() / downside_std
        else:
            return np.inf
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()
