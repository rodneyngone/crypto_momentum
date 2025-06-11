"""Tests for signal generation."""
import pytest
import pandas as pd
import numpy as np
from crypto_momentum_backtest.signals.signal_generator import SignalGenerator


def test_signal_generator_initialization():
    """Test SignalGenerator initialization."""
    generator = SignalGenerator(
        adx_period=14,
        adx_threshold=20,
        ewma_fast=20,
        ewma_slow=50
    )

    assert generator.adx_period == 14
    assert generator.adx_threshold == 20
    assert generator.ewma_fast == 20
    assert generator.ewma_slow == 50


def test_signal_generation():
    """Test basic signal generation."""
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.rand(100) * 1000000
    }, index=dates)

    generator = SignalGenerator()
    signals = generator.generate_signals(df)

    assert 'long_signal' in signals.columns
    assert 'short_signal' in signals.columns
    assert 'position' in signals.columns
    assert len(signals) == len(df)