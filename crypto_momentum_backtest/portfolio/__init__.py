# crypto_momentum_backtest/portfolio/__init__.py
"""Portfolio module with enhanced components and compatibility wrappers."""

from .erc_optimizer_enhanced import EnhancedERCOptimizer
from .rebalancer_enhanced import EnhancedRebalancer as _EnhancedRebalancer
from .agnostic_optimizer import AgnosticRiskParityOptimizer
import logging
from typing import Optional


class Rebalancer(_EnhancedRebalancer):
    """Compatibility wrapper for Rebalancer."""
    
    def __init__(self, rebalance_frequency: Optional[str] = None,
                 base_rebalance_frequency: Optional[str] = None, logger: Optional[logging.Logger] = None, **kwargs):
        # Translate parameter name
        if rebalance_frequency is not None and base_rebalance_frequency is None:
            base_rebalance_frequency = rebalance_frequency
            if logger:
                logger.debug(f"Rebalancer: Translating rebalance_frequency={rebalance_frequency} to base_rebalance_frequency={base_rebalance_frequency}")
            
        super().__init__(base_rebalance_frequency=base_rebalance_frequency or 'weekly', logger=logger, **kwargs)


# Make enhanced versions available with compatibility
ERCOptimizer = EnhancedERCOptimizer
EnhancedRebalancer = _EnhancedRebalancer
ARPOptimizer = AgnosticRiskParityOptimizer

__all__ = [
    'ERCOptimizer', 
    'EnhancedERCOptimizer', 
    'Rebalancer', 
    'EnhancedRebalancer',
    'ARPOptimizer',
    'AgnosticRiskParityOptimizer'
]