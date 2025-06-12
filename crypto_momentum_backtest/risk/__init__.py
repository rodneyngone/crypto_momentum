# crypto_momentum_backtest/risk/__init__.py
"""Risk module with enhanced components."""

from .risk_manager_enhanced import EnhancedRiskManager, MarketRegime

# Make enhanced version the default
RiskManager = EnhancedRiskManager

__all__ = ['RiskManager', 'EnhancedRiskManager', 'MarketRegime']