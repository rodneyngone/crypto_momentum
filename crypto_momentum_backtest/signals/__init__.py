"""Enhanced signals module."""
from .signal_generator_enhanced import EnhancedSignalGenerator
from .signal_generator_enhanced import SignalType, SignalMetrics

# Alias for backward compatibility
SignalGenerator = EnhancedSignalGenerator

__all__ = ['EnhancedSignalGenerator', 'SignalGenerator', 'SignalType', 'SignalMetrics']
