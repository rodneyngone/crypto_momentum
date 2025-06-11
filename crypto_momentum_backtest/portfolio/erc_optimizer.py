# crypto_momentum_backtest/portfolio/erc_optimizer.py
"""Equal Risk Contribution portfolio optimizer."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
import logging


class ERCOptimizer:
    """
    Equal Risk Contribution (Risk Parity) portfolio optimizer.
    """
    
    def __init__(
        self,
        max_position_size: float = 0.20,
        min_position_size: float = 0.02,
        allow_short: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize ERC optimizer.
        
        Args:
            max_position_size: Maximum weight per asset
            min_position_size: Minimum weight per asset
            allow_short: Whether to allow short positions
            logger: Logger instance
        """
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.allow_short = allow_short
        self.logger = logger or logging.getLogger(__name__)
        
    def calculate_risk_contributions(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate risk contributions for each asset.
        
        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix
            
        Returns:
            Risk contributions
        """
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        marginal_contrib = cov_matrix @ weights
        contrib = weights * marginal_contrib / portfolio_vol
        
        return contrib
    
    def erc_objective(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        target_contrib: Optional[np.ndarray] = None
    ) -> float:
        """
        ERC objective function to minimize.
        
        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix
            target_contrib: Target risk contributions
            
        Returns:
            Objective value
        """
        n_assets = len(weights)
        
        if target_contrib is None:
            target_contrib = np.ones(n_assets) / n_assets
        
        # Calculate actual risk contributions
        actual_contrib = self.calculate_risk_contributions(weights, cov_matrix)
        
        # Normalize contributions
        actual_contrib = actual_contrib / actual_contrib.sum()
        
        # Calculate squared deviations from target
        deviations = actual_contrib - target_contrib
        
        return np.sum(deviations ** 2)
    
    def optimize(
        self,
        returns: pd.DataFrame,
        signals: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> pd.Series:
        """
        Optimize portfolio weights using ERC.
        
        Args:
            returns: Asset returns DataFrame
            signals: Trading signals DataFrame
            constraints: Additional constraints
            
        Returns:
            Optimal weights
        """
        # Get assets with active signals
        active_long = signals.columns[signals.iloc[-1] == 1].tolist()
        active_short = signals.columns[signals.iloc[-1] == -1].tolist() if self.allow_short else []
        
        active_assets = active_long + active_short
        
        if not active_assets:
            return pd.Series(0, index=returns.columns)
        
        # Calculate covariance matrix for active assets
        active_returns = returns[active_assets]
        cov_matrix = active_returns.cov().values
        
        # Handle singular matrix
        if np.linalg.cond(cov_matrix) > 1e10:
            # Add small regularization
            cov_matrix += np.eye(len(cov_matrix)) * 1e-6
        
        n_assets = len(active_assets)
        
        # Initial weights
        x0 = np.ones(n_assets) / n_assets
        
        # Bounds
        if self.allow_short:
            bounds = [
                (-self.max_position_size, self.max_position_size)
                for _ in range(n_assets)
            ]
            # Set bounds based on signal direction
            for i, asset in enumerate(active_assets):
                if asset in active_long:
                    bounds[i] = (self.min_position_size, self.max_position_size)
                else:  # short
                    bounds[i] = (-self.max_position_size, -self.min_position_size)
        else:
            bounds = [
                (self.min_position_size, self.max_position_size)
                for _ in range(n_assets)
            ]
        
        # Constraints
        constraints_list = []
        
        # Budget constraint (sum of absolute weights <= 1)
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
        
        # Add custom constraints if provided
        if constraints:
            # Sector constraints, etc.
            pass
        
        # Optimize
        try:
            result = minimize(
                self.erc_objective,
                x0,
                args=(cov_matrix,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'disp': False, 'maxiter': 1000}
            )
            
            if not result.success:
                self.logger.warning(f"Optimization failed: {result.message}")
                # Fall back to equal weight
                weights = x0
            else:
                weights = result.x
                
        except Exception as e:
            self.logger.error(f"Optimization error: {e}")
            weights = x0
        
        # Create weight series
        weight_series = pd.Series(0.0, index=returns.columns)
        weight_series[active_assets] = weights
        
        # Log results
        risk_contrib = self.calculate_risk_contributions(weights, cov_matrix)
        self.logger.info(
            f"Optimized {len(active_assets)} assets - "
            f"Max weight: {np.max(np.abs(weights)):.2%}, "
            f"Risk contrib std: {np.std(risk_contrib):.4f}"
        )
        
        return weight_series
    
    def optimize_with_turnover_penalty(
        self,
        returns: pd.DataFrame,
        signals: pd.DataFrame,
        current_weights: pd.Series,
        turnover_penalty: float = 0.01
    ) -> pd.Series:
        """
        Optimize with turnover penalty to reduce trading.
        
        Args:
            returns: Asset returns
            signals: Trading signals
            current_weights: Current portfolio weights
            turnover_penalty: Penalty per unit of turnover
            
        Returns:
            Optimal weights
        """
        # Get base optimization
        target_weights = self.optimize(returns, signals)
        
        # Calculate turnover
        turnover = np.sum(np.abs(target_weights - current_weights))
        
        # Apply penalty if turnover is high
        if turnover > 0.10:  # More than 10% turnover
            # Blend with current weights
            alpha = 1 - turnover_penalty
            new_weights = alpha * target_weights + (1 - alpha) * current_weights
            
            # Renormalize
            if self.allow_short:
                # Maintain leverage
                scale = np.sum(np.abs(target_weights)) / np.sum(np.abs(new_weights))
                new_weights *= scale
            else:
                new_weights /= new_weights.sum()
            
            return new_weights
        
        return target_weights
