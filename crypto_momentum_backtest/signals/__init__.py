# crypto_momentum_backtest/signals/__init__.py
"""Signals module with enhanced components."""

from .signal_generator_enhanced import EnhancedSignalGenerator, SignalType, SignalMetrics

# Make enhanced version the default
SignalGenerator = EnhancedSignalGenerator

__all__ = ['SignalGenerator', 'EnhancedSignalGenerator', 'SignalType', 'SignalMetrics']