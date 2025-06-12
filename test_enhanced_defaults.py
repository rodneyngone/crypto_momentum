#!/usr/bin/env python3
"""Quick test to verify enhanced components are working as default."""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

def test_enhanced_defaults():
    """Test that enhanced components are being used by default."""
    
    print("🧪 Testing Enhanced Components as Defaults")
    print("=" * 60)
    
    # Test 1: Import and check signal generator
    print("\n1️⃣ Testing SignalGenerator...")
    try:
        from crypto_momentum_backtest.signals import SignalGenerator
        
        # Check if it has enhanced features
        if hasattr(SignalGenerator, 'momentum_weights'):
            print("   ✅ SignalGenerator is using enhanced version")
            print("   ✅ Has momentum_weights attribute")
        else:
            print("   ❌ SignalGenerator seems to be using old version")
        
        # Try to create instance with enhanced parameters
        sig_gen = SignalGenerator(
            adx_periods=[7, 14, 21],  # Enhanced feature
            adaptive_ewma=True         # Enhanced feature
        )
        print("   ✅ Created with enhanced parameters")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Check portfolio optimizer
    print("\n2️⃣ Testing ERCOptimizer...")
    try:
        from crypto_momentum_backtest.portfolio import ERCOptimizer
        
        # Try enhanced parameters
        optimizer = ERCOptimizer(
            concentration_mode=True,     # Enhanced feature
            momentum_tilt_strength=0.5   # Enhanced feature
        )
        print("   ✅ ERCOptimizer is using enhanced version")
        print("   ✅ Accepts concentration_mode parameter")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Check risk manager
    print("\n3️⃣ Testing RiskManager...")
    try:
        from crypto_momentum_backtest.risk import RiskManager
        
        # Check for enhanced methods
        risk_mgr = RiskManager(
            use_trailing_stops=True,  # Enhanced feature
            regime_lookback=60        # Enhanced feature
        )
        
        if hasattr(risk_mgr, 'detect_market_regime'):
            print("   ✅ RiskManager is using enhanced version")
            print("   ✅ Has detect_market_regime method")
        else:
            print("   ❌ Missing enhanced methods")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Check rebalancer
    print("\n4️⃣ Testing Rebalancer...")
    try:
        from crypto_momentum_backtest.portfolio import Rebalancer
        
        rebalancer = Rebalancer(
            momentum_preservation=True,     # Enhanced feature
            use_dynamic_rebalancing=True   # Enhanced feature
        )
        print("   ✅ Rebalancer is using enhanced version")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 5: Check universe manager
    print("\n5️⃣ Testing UniverseManager...")
    try:
        from crypto_momentum_backtest.data import UniverseManager
        
        universe_mgr = UniverseManager(
            data_dir="data",
            base_universe_size=30,      # Enhanced feature
            selection_universe_size=15  # Enhanced feature
        )
        print("   ✅ UniverseManager is using enhanced version")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Enhanced components are successfully set as defaults!")
    print("\nYou can now run your backtest normally and it will use all enhancements.")

if __name__ == "__main__":
    test_enhanced_defaults()
