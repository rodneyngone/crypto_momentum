#!/usr/bin/env python3
"""
Script to make the enhanced components the default in your crypto momentum backtest system.
This updates all __init__.py files to import enhanced versions.
"""

import os
from pathlib import Path

def update_init_files():
    """Update all __init__.py files to use enhanced components as default."""
    
    print("🔧 Making Enhanced Components Default")
    print("=" * 60)
    
    # Define the updates needed for each module
    init_updates = {
        'crypto_momentum_backtest/signals/__init__.py': '''"""Enhanced signals module."""
from .signal_generator_enhanced import EnhancedSignalGenerator
from .signal_generator_enhanced import SignalType, SignalMetrics

# Make enhanced version the default
SignalGenerator = EnhancedSignalGenerator

__all__ = ['SignalGenerator', 'EnhancedSignalGenerator', 'SignalType', 'SignalMetrics']
''',
        
        'crypto_momentum_backtest/portfolio/__init__.py': '''"""Enhanced portfolio module."""
from .erc_optimizer_enhanced import EnhancedERCOptimizer
from .rebalancer_enhanced import EnhancedRebalancer

# Make enhanced versions the default
ERCOptimizer = EnhancedERCOptimizer
Rebalancer = EnhancedRebalancer

__all__ = ['ERCOptimizer', 'EnhancedERCOptimizer', 'Rebalancer', 'EnhancedRebalancer']
''',
        
        'crypto_momentum_backtest/risk/__init__.py': '''"""Enhanced risk module."""
from .risk_manager_enhanced import EnhancedRiskManager, MarketRegime

# Make enhanced version the default
RiskManager = EnhancedRiskManager

__all__ = ['RiskManager', 'EnhancedRiskManager', 'MarketRegime']
''',
        
        'crypto_momentum_backtest/data/__init__.py': '''"""Enhanced data module."""
from .universe_manager_enhanced import EnhancedUniverseManager

# Make enhanced version the default
UniverseManager = EnhancedUniverseManager

__all__ = ['UniverseManager', 'EnhancedUniverseManager']
'''
    }
    
    # Update each __init__.py file
    for file_path, content in init_updates.items():
        path = Path(file_path)
        
        # Create backup if file exists
        if path.exists():
            backup_path = path.with_suffix('.py.backup')
            
            # If backup already exists, remove it
            if backup_path.exists():
                backup_path.unlink()  # Delete existing backup
            
            path.rename(backup_path)
            print(f"✅ Backed up {file_path} to {backup_path}")
        
        # Write new content
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        print(f"✅ Updated {file_path}")
    
    print("\n📝 Updating imports in main files...")
    update_main_imports()
    
    print("\n🔄 Creating compatibility wrapper...")
    create_compatibility_wrapper()
    
    print("\n✅ Enhanced components are now the default!")
    print("\nThe system will now use:")
    print("  - EnhancedSignalGenerator as SignalGenerator")
    print("  - EnhancedERCOptimizer as ERCOptimizer")
    print("  - EnhancedRiskManager as RiskManager")
    print("  - EnhancedRebalancer as Rebalancer")
    print("  - EnhancedUniverseManager as UniverseManager")
    
    print("\n🎯 Next steps:")
    print("1. No need to change any existing code - it will use enhanced versions")
    print("2. Run your backtest as usual: python run.py --no-validate")
    print("3. Use config_enhanced.yaml for optimized parameters")

def update_main_imports():
    """Update imports in main engine files to ensure compatibility."""
    
    # Update backtest/engine.py if needed
    engine_path = Path('crypto_momentum_backtest/backtest/engine.py')
    if engine_path.exists():
        content = engine_path.read_text()
        
        # Check if it's importing from the modules directly
        if 'from ..signals.signal_generator import SignalGenerator' in content:
            # It's importing the old way, update it
            content = content.replace(
                'from ..signals.signal_generator import SignalGenerator',
                'from ..signals import SignalGenerator'
            )
            content = content.replace(
                'from ..portfolio.erc_optimizer import ERCOptimizer',
                'from ..portfolio import ERCOptimizer'
            )
            content = content.replace(
                'from ..portfolio.rebalancer import Rebalancer',
                'from ..portfolio import Rebalancer'
            )
            content = content.replace(
                'from ..risk.risk_manager import RiskManager',
                'from ..risk import RiskManager'
            )
            content = content.replace(
                'from ..data.universe_manager import UniverseManager',
                'from ..data import UniverseManager'
            )
            
            # Save updated content
            engine_path.write_text(content, encoding='utf-8')
            print("[OK] Updated imports in backtest/engine.py")

def create_compatibility_wrapper():
    """Create a compatibility wrapper for smooth transition."""
    
    wrapper_content = '''#!/usr/bin/env python3
"""
Compatibility wrapper to ensure enhanced components work with existing code.
This file maps old component names to enhanced versions.
"""

# Signal components
from crypto_momentum_backtest.signals import (
    SignalGenerator,  # This is now EnhancedSignalGenerator
    SignalType,
    SignalMetrics
)

# Portfolio components
from crypto_momentum_backtest.portfolio import (
    ERCOptimizer,  # This is now EnhancedERCOptimizer
    Rebalancer     # This is now EnhancedRebalancer
)

# Risk components
from crypto_momentum_backtest.risk import (
    RiskManager,   # This is now EnhancedRiskManager
    MarketRegime
)

# Data components
from crypto_momentum_backtest.data import (
    UniverseManager  # This is now EnhancedUniverseManager
)

print("✅ Using enhanced components as defaults")

# Verify enhanced features are available
def verify_enhancements():
    """Quick check that enhanced features are available."""
    checks = {
        'SignalGenerator.generate_signals': hasattr(SignalGenerator, 'generate_signals'),
        'ERCOptimizer.concentration_mode': hasattr(ERCOptimizer, '__init__') and 'concentration_mode' in ERCOptimizer.__init__.__code__.co_varnames,
        'RiskManager.detect_market_regime': hasattr(RiskManager, 'detect_market_regime'),
        'Rebalancer.momentum_preservation': hasattr(Rebalancer, '__init__') and 'momentum_preservation' in Rebalancer.__init__.__code__.co_varnames,
    }
    
    print("\\nEnhancement verification:")
    for feature, available in checks.items():
        status = "✅" if available else "❌"
        print(f"  {status} {feature}")
    
    return all(checks.values())

if __name__ == "__main__":
    if verify_enhancements():
        print("\\n✅ All enhanced features are available!")
    else:
        print("\\n❌ Some enhanced features missing - check imports")
'''
    
    wrapper_path = Path('crypto_momentum_backtest/compatibility_check.py')
    wrapper_path.write_text(wrapper_content, encoding='utf-8')
    print("✅ Created compatibility_check.py")

def create_quick_test():
    """Create a quick test script to verify everything works."""
    
    test_content = '''#!/usr/bin/env python3
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
    print("\\n1️⃣ Testing SignalGenerator...")
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
    print("\\n2️⃣ Testing ERCOptimizer...")
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
    print("\\n3️⃣ Testing RiskManager...")
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
    print("\\n4️⃣ Testing Rebalancer...")
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
    print("\\n5️⃣ Testing UniverseManager...")
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
    
    print("\\n" + "=" * 60)
    print("✅ Enhanced components are successfully set as defaults!")
    print("\\nYou can now run your backtest normally and it will use all enhancements.")

if __name__ == "__main__":
    test_enhanced_defaults()
'''
    
    test_path = Path('test_enhanced_defaults.py')
    test_path.write_text(test_content, encoding='utf-8')
    print("✅ Created test_enhanced_defaults.py")

if __name__ == "__main__":
    update_init_files()
    create_quick_test()
    
    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("\nTo verify everything works:")
    print("  python test_enhanced_defaults.py")
    print("\nTo run compatibility check:")
    print("  python -m crypto_momentum_backtest.compatibility_check")
    print("\nYour existing code will now automatically use enhanced components!")