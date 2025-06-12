# crypto_momentum_backtest/portfolio/agnostic_optimizer.py
"""
Agnostic Risk Parity (ARP) portfolio optimizer implementation.

Based on "Agnostic Risk Parity: Taming Known and Unknown-Unknowns" by Benichou et al. (2016)
arXiv:1610.08818

The key insight of ARP is to achieve equal risk contribution across all principal components
of the covariance matrix when signal predictors are assumed uncorrelated with equal variance.
This addresses the rotational invariance problem in portfolio construction.
"""

import numpy as np
import pandas as pd
from scipy import linalg
from typing import Dict, Optional, Union, Tuple
import logging
from functools import lru_cache


class AgnosticRiskParityOptimizer:
    """
    Agnostic Risk Parity (ARP) portfolio optimizer.
    
    ARP computes portfolio weights as:
        π = ω * C^(-1/2) * p
    
    Where:
        - C: Cleaned covariance matrix of asset returns
        - p: Vector of standardized signal scores/predictors
        - ω: Scalar for volatility scaling
        
    The key assumption is that signal covariance Q = I (uncorrelated, equal variance),
    which makes the portfolio construction rotationally invariant.
    
    This differs from Markowitz optimization (π = C^(-1) * p) by using the square root
    of the inverse covariance, which equalizes risk across all eigenvectors of C.
    """
    
    def __init__(
        self,
        target_volatility: Optional[float] = None,
        budget_constraint: str = 'full_investment',
        min_eigenvalue: float = 1e-8,
        cache_matrix_ops: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Agnostic Risk Parity optimizer.
        
        Args:
            target_volatility: Target portfolio volatility (annualized). If None, uses budget constraint.
            budget_constraint: 'full_investment' (sum(abs(w)) = 1) or 'long_only' (sum(w) = 1)
            min_eigenvalue: Minimum eigenvalue for numerical stability
            cache_matrix_ops: Whether to cache expensive matrix operations
            logger: Logger instance
        """
        self.target_volatility = target_volatility
        self.budget_constraint = budget_constraint
        self.min_eigenvalue = min_eigenvalue
        self.cache_matrix_ops = cache_matrix_ops
        self.logger = logger or logging.getLogger(__name__)
        
        # Cache for matrix operations
        self._matrix_cache = {}
        
    def optimize(
        self,
        covariance_matrix: Union[np.ndarray, pd.DataFrame],
        signals: Union[np.ndarray, pd.Series],
        signal_covariance: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        shrinkage_factor: Optional[float] = None
    ) -> Union[np.ndarray, pd.Series]:
        """
        Compute ARP portfolio weights.
        
        Args:
            covariance_matrix: Cleaned covariance matrix of returns (NxN)
            signals: Vector of signal scores/predictors (N)
            signal_covariance: Optional signal covariance matrix. If None, assumes Q = I
            shrinkage_factor: Optional shrinkage between ARP (Q=I) and Markowitz (Q=C).
                             0 = pure ARP, 1 = pure Markowitz
                             
        Returns:
            Portfolio weights normalized according to constraints
        """
        # Convert inputs to numpy arrays
        C = self._to_numpy(covariance_matrix)
        p = self._to_numpy(signals).flatten()
        
        # Validate inputs
        self._validate_inputs(C, p)
        
        # Compute C^(-1/2)
        C_inv_sqrt = self._compute_matrix_inverse_sqrt(C)
        
        # Handle signal covariance if provided
        if signal_covariance is not None:
            Q = self._to_numpy(signal_covariance)
            Q_inv_sqrt = self._compute_matrix_inverse_sqrt(Q)
            # General formula: π = ω * C^(-1/2) * Q^(-1/2) * p
            raw_weights = C_inv_sqrt @ Q_inv_sqrt @ p
        elif shrinkage_factor is not None:
            # Shrinkage between Q = I (ARP) and Q = C (Markowitz)
            # Q = φ*C + (1-φ)*I
            if not 0 <= shrinkage_factor <= 1:
                raise ValueError("Shrinkage factor must be between 0 and 1")
            
            Q_shrunk = shrinkage_factor * C + (1 - shrinkage_factor) * np.eye(len(C))
            Q_inv_sqrt = self._compute_matrix_inverse_sqrt(Q_shrunk)
            raw_weights = C_inv_sqrt @ Q_inv_sqrt @ p
        else:
            # Pure ARP: Q = I, so Q^(-1/2) = I
            raw_weights = C_inv_sqrt @ p
        
        # Scale weights according to target volatility or budget constraint
        weights = self._scale_weights(raw_weights, C)
        
        # Log portfolio statistics
        self._log_portfolio_stats(weights, C, p)
        
        # Return in same format as input
        if isinstance(signals, pd.Series):
            return pd.Series(weights, index=signals.index)
        else:
            return weights
    
    def _compute_matrix_inverse_sqrt(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute matrix inverse square root using eigendecomposition.
        
        For a positive definite matrix A:
            A^(-1/2) = U * Λ^(-1/2) * U^T
            
        Where U are eigenvectors and Λ are eigenvalues.
        """
        # Check cache if enabled
        cache_key = f"inv_sqrt_{matrix.tobytes()}"
        if self.cache_matrix_ops and cache_key in self._matrix_cache:
            return self._matrix_cache[cache_key]
        
        # Eigendecomposition
        eigenvalues, eigenvectors = linalg.eigh(matrix)
        
        # Handle numerical issues with small eigenvalues
        eigenvalues = np.maximum(eigenvalues, self.min_eigenvalue)
        
        # Compute Λ^(-1/2)
        inv_sqrt_eigenvalues = 1.0 / np.sqrt(eigenvalues)
        
        # Reconstruct A^(-1/2) = U * Λ^(-1/2) * U^T
        inv_sqrt_matrix = eigenvectors @ np.diag(inv_sqrt_eigenvalues) @ eigenvectors.T
        
        # Cache result
        if self.cache_matrix_ops:
            self._matrix_cache[cache_key] = inv_sqrt_matrix
        
        return inv_sqrt_matrix
    
    def _scale_weights(self, raw_weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        """
        Scale weights according to target volatility or budget constraint.
        """
        if self.target_volatility is not None:
            # Scale to target volatility
            current_vol = np.sqrt(raw_weights @ covariance @ raw_weights)
            # Assume annualized volatility with 252 trading days
            annualized_factor = np.sqrt(252)
            scale_factor = self.target_volatility / (current_vol * annualized_factor)
            return raw_weights * scale_factor
        else:
            # Apply budget constraint
            if self.budget_constraint == 'full_investment':
                # Sum of absolute weights = 1
                return raw_weights / np.sum(np.abs(raw_weights))
            elif self.budget_constraint == 'long_only':
                # Long only: set negative weights to 0 and normalize
                positive_weights = np.maximum(raw_weights, 0)
                weight_sum = np.sum(positive_weights)
                if weight_sum > 0:
                    return positive_weights / weight_sum
                else:
                    # Fall back to equal weights if all signals are negative
                    return np.ones_like(raw_weights) / len(raw_weights)
            else:
                raise ValueError(f"Unknown budget constraint: {self.budget_constraint}")
    
    def calculate_risk_contributions(
        self,
        weights: Union[np.ndarray, pd.Series],
        covariance: Union[np.ndarray, pd.DataFrame]
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Calculate risk contributions for each asset and principal component.
        
        Returns:
            Dictionary containing:
                - 'asset_contributions': Risk contribution of each asset
                - 'pc_contributions': Risk contribution of each principal component
                - 'concentration_ratio': Herfindahl index of risk contributions
        """
        w = self._to_numpy(weights)
        C = self._to_numpy(covariance)
        
        # Portfolio volatility
        portfolio_vol = np.sqrt(w @ C @ w)
        
        # Asset risk contributions
        marginal_contributions = C @ w
        asset_contributions = w * marginal_contributions / portfolio_vol
        
        # Principal component risk contributions
        eigenvalues, eigenvectors = linalg.eigh(C)
        pc_weights = eigenvectors.T @ w  # Project weights onto PCs
        pc_variances = eigenvalues * pc_weights**2
        pc_contributions = pc_variances / np.sum(pc_variances)
        
        # Concentration ratio (should be close to 1/N for true risk parity)
        concentration_ratio = np.sum(asset_contributions**2)
        
        return {
            'asset_contributions': asset_contributions,
            'pc_contributions': pc_contributions,
            'concentration_ratio': concentration_ratio,
            'equal_risk_score': 1.0 / (len(weights) * concentration_ratio)  # 1 = perfect parity
        }
    
    def _validate_inputs(self, covariance: np.ndarray, signals: np.ndarray):
        """Validate input matrices and vectors."""
        # Check dimensions
        n_assets = len(signals)
        if covariance.shape != (n_assets, n_assets):
            raise ValueError(f"Covariance matrix shape {covariance.shape} doesn't match "
                           f"signal length {n_assets}")
        
        # Check symmetry
        if not np.allclose(covariance, covariance.T):
            raise ValueError("Covariance matrix must be symmetric")
        
        # Check positive definiteness (all eigenvalues > 0)
        eigenvalues = linalg.eigvalsh(covariance)
        if np.min(eigenvalues) < -self.min_eigenvalue:
            raise ValueError("Covariance matrix must be positive semi-definite")
        
        # Check for NaN or inf
        if np.any(np.isnan(covariance)) or np.any(np.isinf(covariance)):
            raise ValueError("Covariance matrix contains NaN or inf values")
        
        if np.any(np.isnan(signals)) or np.any(np.isinf(signals)):
            raise ValueError("Signal vector contains NaN or inf values")
    
    def _to_numpy(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values
        return np.asarray(data)
    
    def _log_portfolio_stats(self, weights: np.ndarray, covariance: np.ndarray, signals: np.ndarray):
        """Log portfolio statistics for monitoring."""
        portfolio_vol = np.sqrt(weights @ covariance @ weights) * np.sqrt(252)
        n_long = np.sum(weights > 0)
        n_short = np.sum(weights < 0)
        leverage = np.sum(np.abs(weights))
        
        # Calculate risk contributions
        risk_metrics = self.calculate_risk_contributions(weights, covariance)
        
        self.logger.info(
            f"ARP Portfolio - Vol: {portfolio_vol:.2%}, "
            f"Long: {n_long}, Short: {n_short}, "
            f"Leverage: {leverage:.2f}, "
            f"Risk Parity Score: {risk_metrics['equal_risk_score']:.3f}"
        )
    
    def compare_with_markowitz(
        self,
        covariance: Union[np.ndarray, pd.DataFrame],
        signals: Union[np.ndarray, pd.Series]
    ) -> Dict[str, np.ndarray]:
        """
        Compare ARP weights with traditional Markowitz weights.
        
        Returns:
            Dictionary with 'arp' and 'markowitz' weights
        """
        # ARP weights
        arp_weights = self.optimize(covariance, signals)
        
        # Markowitz weights: π = C^(-1) * p
        C = self._to_numpy(covariance)
        p = self._to_numpy(signals).flatten()
        
        # Use pseudo-inverse for numerical stability
        C_inv = linalg.pinv(C, rcond=self.min_eigenvalue)
        markowitz_raw = C_inv @ p
        
        # Scale Markowitz weights using same method
        markowitz_weights = self._scale_weights(markowitz_raw, C)
        
        return {
            'arp': arp_weights,
            'markowitz': markowitz_weights
        }