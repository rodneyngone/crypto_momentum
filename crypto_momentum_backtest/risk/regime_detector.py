# crypto_momentum_backtest/risk/regime_detector.py
"""Market regime detection for adaptive risk management."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import logging


class RegimeDetector:
    """
    Detects market regimes using various statistical methods.
    """
    
    def __init__(
        self,
        lookback_period: int = 252,
        n_regimes: int = 3,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize regime detector.
        
        Args:
            lookback_period: Historical lookback for regime detection
            n_regimes: Number of regimes to detect
            logger: Logger instance
        """
        self.lookback_period = lookback_period
        self.n_regimes = n_regimes
        self.logger = logger or logging.getLogger(__name__)
        
        self.regime_names = {
            0: 'low_volatility',
            1: 'normal',
            2: 'high_volatility'
        }
        
    def detect_volatility_regime(
        self,
        returns: pd.Series,
        vol_thresholds: Optional[List[float]] = None
    ) -> str:
        """
        Simple volatility-based regime detection.
        
        Args:
            returns: Returns series
            vol_thresholds: Volatility thresholds
            
        Returns:
            Current regime name
        """
        if vol_thresholds is None:
            vol_thresholds = [0.10, 0.20]  # 10% and 20% annualized
        
        # Calculate recent volatility
        recent_vol = returns.iloc[-20:].std() * np.sqrt(252)
        
        if recent_vol < vol_thresholds[0]:
            return 'low_volatility'
        elif recent_vol < vol_thresholds[1]:
            return 'normal'
        else:
            return 'high_volatility'
    
    def detect_trend_regime(
        self,
        prices: pd.Series,
        short_window: int = 50,
        long_window: int = 200
    ) -> str:
        """
        Trend-based regime detection.
        
        Args:
            prices: Price series
            short_window: Short MA window
            long_window: Long MA window
            
        Returns:
            Trend regime
        """
        short_ma = prices.rolling(short_window).mean()
        long_ma = prices.rolling(long_window).mean()
        
        current_price = prices.iloc[-1]
        current_short_ma = short_ma.iloc[-1]
        current_long_ma = long_ma.iloc[-1]
        
        if current_price > current_short_ma > current_long_ma:
            return 'strong_uptrend'
        elif current_price > current_long_ma:
            return 'uptrend'
        elif current_price < current_short_ma < current_long_ma:
            return 'strong_downtrend'
        elif current_price < current_long_ma:
            return 'downtrend'
        else:
            return 'sideways'
    
    def detect_gmm_regime(
        self,
        returns: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> pd.Series:
        """
        Gaussian Mixture Model regime detection.
        
        Args:
            returns: Returns DataFrame
            features: Features to use for regime detection
            
        Returns:
            Regime labels
        """
        if features is None:
            # Default features
            features_df = pd.DataFrame({
                'returns': returns.mean(axis=1),
                'volatility': returns.std(axis=1).rolling(20).mean(),
                'skew': returns.rolling(60).skew().mean(axis=1),
                'kurt': returns.rolling(60).kurt().mean(axis=1)
            })
        else:
            features_df = returns[features]
        
        # Remove NaN values
        features_df = features_df.dropna()
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)
        
        # Fit GMM
        gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=42
        )
        
        regimes = gmm.fit_predict(features_scaled)
        
        # Create regime series
        regime_series = pd.Series(
            regimes,
            index=features_df.index,
            name='regime'
        )
        
        # Map to regime names based on average volatility
        regime_vols = {}
        for regime in range(self.n_regimes):
            mask = regime_series == regime
            avg_vol = features_df.loc[mask, 'volatility'].mean()
            regime_vols[regime] = avg_vol
        
        # Sort regimes by volatility
        sorted_regimes = sorted(regime_vols.items(), key=lambda x: x[1])
        regime_mapping = {
            sorted_regimes[i][0]: i for i in range(self.n_regimes)
        }
        
        # Apply mapping
        regime_series = regime_series.map(regime_mapping)
        
        return regime_series
    
    def calculate_regime_statistics(
        self,
        returns: pd.DataFrame,
        regimes: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate statistics for each regime.
        
        Args:
            returns: Returns DataFrame
            regimes: Regime labels
            
        Returns:
            Statistics by regime
        """
        stats = []
        
        for regime in sorted(regimes.unique()):
            mask = regimes == regime
            regime_returns = returns[mask]
            
            stats.append({
                'regime': self.regime_names.get(regime, f'regime_{regime}'),
                'count': mask.sum(),
                'frequency': mask.sum() / len(regimes),
                'avg_return': regime_returns.mean().mean() * 252,
                'volatility': regime_returns.std().mean() * np.sqrt(252),
                'sharpe': (regime_returns.mean().mean() * 252) / 
                         (regime_returns.std().mean() * np.sqrt(252)),
                'max_drawdown': self._calculate_max_drawdown(
                    regime_returns.mean(axis=1)
                )
            })
        
        return pd.DataFrame(stats)
    
    def calculate_transition_matrix(
        self,
        regimes: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate regime transition probability matrix.
        
        Args:
            regimes: Regime labels
            
        Returns:
            Transition matrix
        """
        n_regimes = len(regimes.unique())
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        # Count transitions
        for i in range(len(regimes) - 1):
            from_regime = regimes.iloc[i]
            to_regime = regimes.iloc[i + 1]
            transition_matrix[from_regime, to_regime] += 1
        
        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = transition_matrix / row_sums
        
        # Create DataFrame
        regime_labels = [
            self.regime_names.get(i, f'regime_{i}')
            for i in range(n_regimes)
        ]
        
        transition_df = pd.DataFrame(
            transition_matrix,
            index=regime_labels,
            columns=regime_labels
        )
        
        return transition_df
    
    @staticmethod
    def _calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()
