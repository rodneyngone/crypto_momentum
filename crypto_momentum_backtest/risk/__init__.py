"""Enhanced risk module."""
from .risk_manager_enhanced import EnhancedRiskManager, MarketRegime

# Alias for backward compatibility
RiskManager = EnhancedRiskManager

__all__ = ['EnhancedRiskManager', 'RiskManager', 'MarketRegime']
