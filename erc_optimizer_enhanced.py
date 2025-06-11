# crypto_momentum_backtest/portfolio/erc_optimizer_enhanced.py
"""Enhanced ERC portfolio optimizer with momentum tilting and concentration modes."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
import logging


class EnhancedERCOptimizer:
    """
    Enhanced Equal Risk Contribution optimizer with momentum tilting,
    concentration modes, and adaptive allocation.
    """
    
    def __init__(
        self,
        max_position_size: float = 0.25,  # Increased from 0.20
        min_position_size: float = 0.02,
        allow_short: bool = True,
        concentration_mode: bool = True,
        top_n_assets: int = 5,
        concentration_weight: float = 0.6,
        momentum_tilt_strength: float = 0.5,
        use_momentum_weighting: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize enhanced ERC optimizer.
        
        Args:
            max_position_size: Maximum weight per asset
            min_position_size: Minimum weight per asset
            allow_short: Whether to allow short positions
            concentration_mode: Whether to use concentrated portfolio mode
            top_n_assets: Number of top assets to focus on in concentration mode
            concentration_weight: Weight allocated to top assets in concentration mode
            momentum_tilt_strength: Strength of momentum tilt (0-1)
            use_momentum_weighting: Whether to tilt weights by momentum
            logger: Logger instance
        """
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.allow_short = allow_short
        self.concentration_mode = concentration_mode
        self.top_n_assets = top_n_assets
        self.concentration_weight = concentration_weight
        self.momentum_tilt_strength = momentum_tilt_strength
        self.use_momentum_weighting = use_momentum_weighting
        self.logger = logger or logging.getLogger(__name__)
    
    def optimize(
        self,
        returns: pd.DataFrame,
        signals: pd.DataFrame,
        momentum_scores: Optional[pd.DataFrame] = None,
        constraints: Optional[Dict] = None,
        market_regime: str = 'normal'
    ) -> pd.Series:
        """
        Optimize portfolio weights with enhanced features.
        
        Args:
            returns: Asset returns DataFrame
            signals: Trading signals DataFrame
            momentum_scores: Optional momentum scores for tilting
            constraints: Additional constraints
            market_regime: Current market regime ('trending', 'volatile', 'normal')
            
        Returns:
            Optimal weights
        """
        # Get active assets based on signals
        active_long = signals.columns[signals.iloc[-1] == 1].tolist()
        active_short = signals.columns[signals.iloc[-1] == -1].tolist() if self.allow_short else []
        
        active_assets = active_long + active_short
        
        if not active_assets:
            return pd.Series(0, index=returns.columns)
        
        # In concentration mode, focus on top performers
        if self.concentration_mode and momentum_scores is not None:
            active_assets = self._select_concentrated_assets(
                active_assets, momentum_scores, signals.iloc[-1]
            )
        
        # Adjust allocation based on market regime
        if market_regime == 'trending' and self.concentration_mode:
            # Even more concentrated in trending markets
            self.top_n_assets = min(3, len(active_assets))
            self.concentration_weight = 0.75
        elif market_regime == 'volatile':
            # More diversified in volatile markets
            self.concentration_mode = False
        
        # Calculate base weights
        if self.use_momentum_weighting and momentum_scores is not None:
            weights = self._optimize_with_momentum_tilt(
                returns, active_assets, momentum_scores, active_long, active_short
            )
        else:
            weights = self._optimize_erc(
                returns, active_assets, active_long, active_short
            )
        
        # Apply concentration if enabled
        if self.concentration_mode:
            weights = self._apply_concentration(weights, momentum_scores)
        
        # Create full weight series
        weight_series = pd.Series(0.0, index=returns.columns)
        for asset, weight in weights.items():
            weight_series[asset] = weight
        
        # Log optimization results
        self._log_optimization_results(weight_series, active_assets, market_regime)
        
        return weight_series
    
    def _select_concentrated_assets(
        self,
        active_assets: List[str],
        momentum_scores: pd.DataFrame,
        current_signals: pd.Series
    ) -> List[str]:
        """Select top assets for concentrated portfolio."""
        # Get momentum scores for active assets
        asset_scores = {}
        
        for asset in active_assets:
            if asset in momentum_scores.columns:
                # Use recent average momentum score
                score = momentum_scores[asset].iloc[-20:].mean()
                # Adjust for signal direction
                if current_signals[asset] < 0:
                    score = -score
                asset_scores[asset] = score
        
        # Sort by score and select top N
        sorted_assets = sorted(asset_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        selected = [asset for asset, _ in sorted_assets[:self.top_n_assets]]
        
        self.logger.info(f"Concentrated portfolio: selected {len(selected)} assets from {len(active_assets)}")
        
        return selected
    
    def _optimize_with_momentum_tilt(
        self,
        returns: pd.DataFrame,
        active_assets: List[str],
        momentum_scores: pd.DataFrame,
        active_long: List[str],
        active_short: List[str]
    ) -> Dict[str, float]:
        """Optimize with momentum-based tilting."""
        # First get base ERC weights
        base_weights = self._optimize_erc(returns, active_assets, active_long, active_short)
        
        # Calculate momentum multipliers
        momentum_multipliers = {}
        
        for asset in active_assets:
            if asset in momentum_scores.columns:
                # Use recent momentum score
                score = momentum_scores[asset].iloc[-20:].mean()
                
                # Convert to multiplier (1 + tilt_strength * normalized_score)
                # Normalize score to [-1, 1] range
                normalized_score = np.clip(score, -1, 1)
                
                # Apply tilt strength
                multiplier = 1 + (self.momentum_tilt_strength * normalized_score)
                momentum_multipliers[asset] = max(0.5, min(1.5, multiplier))  # Cap between 0.5x and 1.5x
            else:
                momentum_multipliers[asset] = 1.0
        
        # Apply multipliers to base weights
        tilted_weights = {}
        for asset, base_weight in base_weights.items():
            tilted_weights[asset] = base_weight * momentum_multipliers.get(asset, 1.0)
        
        # Renormalize
        total_weight = sum(abs(w) for w in tilted_weights.values())
        if total_weight > 0:
            for asset in tilted_weights:
                tilted_weights[asset] /= total_weight
        
        # Apply position size constraints
        for asset in tilted_weights:
            if asset in active_long:
                tilted_weights[asset] = np.clip(tilted_weights[asset], self.min_position_size, self.max_position_size)
            else:  # short position
                tilted_weights[asset] = np.clip(tilted_weights[asset], -self.max_position_size, -self.min_position_size)
        
        # Final renormalization
        total_weight = sum(abs(w) for w in tilted_weights.values())
        if total_weight > 0:
            for asset in tilted_weights:
                tilted_weights[asset] /= total_weight
        
        return tilted_weights
    
    def _optimize_erc(
        self,
        returns: pd.DataFrame,
        active_assets: List[str],
        active_long: List[str],
        active_short: List[str]
    ) -> Dict[str, float]:
        """Standard ERC optimization."""
        # Calculate covariance matrix for active assets
        active_returns = returns[active_assets]
        cov_matrix = active_returns.cov().values
        
        # Handle singular matrix
        if np.linalg.cond(cov_matrix) > 1e10:
            cov_matrix += np.eye(len(cov_matrix)) * 1e-6
        
        n_assets = len(active_assets)
        
        # Initial weights
        x0 = np.ones(n_assets) / n_assets
        
        # Bounds based on position direction
        bounds = []
        for i, asset in enumerate(active_assets):
            if asset in active_long:
                bounds.append((self.min_position_size, self.max_position_size))
            else:  # short position
                bounds.append((-self.max_position_size, -self.min_position_size))
        
        # Constraints
        constraints_list = []
        
        # Budget constraint
        if self.allow_short:
            constraints_list.append({
                'type': 'ineq',
                'fun': lambda x: 1.0 - np.sum(np.abs(x))
            })
        else:
            constraints_list.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x) - 1.0
            })
        
        # Optimize
        try:
            result = minimize(
                self._erc_objective,
                x0,
                args=(cov_matrix,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'disp': False, 'maxiter': 1000}
            )
            
            if not result.success:
                self.logger.warning(f"Optimization failed: {result.message}")
                weights = x0
            else:
                weights = result.x
                
        except Exception as e:
            self.logger.error(f"Optimization error: {e}")
            weights = x0
        
        # Create weight dictionary
        weight_dict = {asset: weight for asset, weight in zip(active_assets, weights)}
        
        return weight_dict
    
    def _apply_concentration(
        self,
        weights: Dict[str, float],
        momentum_scores: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Apply concentration to top assets."""
        if len(weights) <= self.top_n_assets:
            return weights
        
        # Sort by absolute weight (or momentum if available)
        if momentum_scores is not None:
            # Sort by momentum score
            asset_scores = {}
            for asset in weights:
                if asset in momentum_scores.columns:
                    asset_scores[asset] = abs(momentum_scores[asset].iloc[-20:].mean())
                else:
                    asset_scores[asset] = abs(weights[asset])
            
            sorted_assets = sorted(asset_scores.items(), key=lambda x: x[1], reverse=True)
        else:
            # Sort by weight
            sorted_assets = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Allocate concentration_weight to top N assets
        concentrated_weights = {}
        top_assets = [asset for asset, _ in sorted_assets[:self.top_n_assets]]
        other_assets = [asset for asset, _ in sorted_assets[self.top_n_assets:]]
        
        # Calculate weights for top assets
        top_weight_sum = sum(abs(weights[asset]) for asset in top_assets)
        if top_weight_sum > 0:
            for asset in top_assets:
                concentrated_weights[asset] = (
                    weights[asset] / top_weight_sum * self.concentration_weight
                )
        
        # Remaining weight for other assets
        remaining_weight = 1.0 - self.concentration_weight
        other_weight_sum = sum(abs(weights[asset]) for asset in other_assets)
        
        if other_weight_sum > 0 and remaining_weight > 0:
            for asset in other_assets:
                concentrated_weights[asset] = (
                    weights[asset] / other_weight_sum * remaining_weight
                )
        else:
            # Set other assets to zero
            for asset in other_assets:
                concentrated_weights[asset] = 0.0
        
        return concentrated_weights
    
    def _erc_objective(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """ERC objective function to minimize."""
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        marginal_contrib = cov_matrix @ weights
        contrib = weights * marginal_contrib / portfolio_vol
        
        # Target equal contribution
        target_contrib = np.ones(len(weights)) / len(weights)
        
        # Normalize contributions
        contrib_sum = contrib.sum()
        if contrib_sum > 0:
            contrib = contrib / contrib_sum
        
        # Calculate squared deviations
        deviations = contrib - target_contrib
        
        return np.sum(deviations ** 2)
    
    def _log_optimization_results(
        self,
        weights: pd.Series,
        active_assets: List[str],
        market_regime: str
    ):
        """Log optimization results for monitoring."""
        non_zero_weights = weights[weights != 0]
        
        self.logger.info(
            f"Optimized portfolio - Regime: {market_regime}, "
            f"Active: {len(active_assets)}, Allocated: {len(non_zero_weights)}, "
            f"Max weight: {non_zero_weights.abs().max():.2%}, "
            f"Concentration: {non_zero_weights.abs().iloc[:self.top_n_assets].sum():.2%}"
        )
    
    def calculate_risk_contributions(
        self,
        weights: pd.Series,
        returns: pd.DataFrame
    ) -> pd.Series:
        """Calculate risk contribution of each asset."""
        # Filter to non-zero weights
        active_weights = weights[weights != 0]
        active_returns = returns[active_weights.index]
        
        if len(active_weights) == 0:
            return pd.Series()
        
        # Calculate covariance matrix
        cov_matrix = active_returns.cov()
        
        # Portfolio volatility
        portfolio_vol = np.sqrt(active_weights @ cov_matrix @ active_weights)
        
        # Marginal contributions
        marginal_contrib = cov_matrix @ active_weights
        
        # Risk contributions
        risk_contrib = active_weights * marginal_contrib / portfolio_vol
        
        return risk_contrib
    
    def optimize_with_turnover_penalty(
        self,
        returns: pd.DataFrame,
        signals: pd.DataFrame,
        current_weights: pd.Series,
        momentum_scores: Optional[pd.DataFrame] = None,
        turnover_penalty: float = 0.01,
        market_regime: str = 'normal'
    ) -> pd.Series:
        """
        Optimize with turnover penalty to reduce excessive trading.
        
        Args:
            returns: Asset returns
            signals: Trading signals
            current_weights: Current portfolio weights
            momentum_scores: Optional momentum scores
            turnover_penalty: Penalty per unit of turnover
            market_regime: Current market regime
            
        Returns:
            Optimal weights with turnover consideration
        """
        # Get target weights without turnover consideration
        target_weights = self.optimize(returns, signals, momentum_scores, market_regime=market_regime)
        
        # Calculate expected turnover
        turnover = np.sum(np.abs(target_weights - current_weights))
        
        # Adjust based on market regime
        regime_thresholds = {
            'trending': 0.15,  # Allow more turnover in trending markets
            'volatile': 0.05,  # Minimize turnover in volatile markets
            'normal': 0.10
        }
        
        threshold = regime_thresholds.get(market_regime, 0.10)
        
        # Apply penalty if turnover is high
        if turnover > threshold:
            # Blend with current weights
            alpha = 1 - (turnover - threshold) * turnover_penalty
            alpha = max(0.3, min(1.0, alpha))  # Keep alpha between 30% and 100%
            
            new_weights = alpha * target_weights + (1 - alpha) * current_weights
            
            # Renormalize
            if self.allow_short:
                # Maintain leverage
                scale = np.sum(np.abs(target_weights)) / np.sum(np.abs(new_weights))
                new_weights *= scale
            else:
                new_weights /= new_weights.sum()
            
            self.logger.info(
                f"Applied turnover penalty: {turnover:.2%} turnover, "
                f"alpha={alpha:.2%}, regime={market_regime}"
            )
            
            return new_weights
        
        return target_weights