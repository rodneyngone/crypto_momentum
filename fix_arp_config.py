#!/usr/bin/env python3
"""
Script to fix ARP configuration issues in config.py
"""

import re
from pathlib import Path
import shutil
from datetime import datetime

def fix_arp_configuration():
    """Fix both OptimizationMethod enum and PortfolioConfig to support ARP."""
    
    config_path = Path("crypto_momentum_backtest/utils/config.py")
    
    # Create backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = config_path.with_name(f"{config_path.stem}_backup_{timestamp}{config_path.suffix}")
    shutil.copy2(config_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    # Read the file
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Fix 1: Add AGNOSTIC_RISK_PARITY to OptimizationMethod enum
    if 'AGNOSTIC_RISK_PARITY' not in content:
        print("Adding AGNOSTIC_RISK_PARITY to OptimizationMethod enum...")
        
        # Find the OptimizationMethod enum and add ARP
        pattern = r'(class OptimizationMethod\(Enum\):.*?ENHANCED_RISK_PARITY = "enhanced_risk_parity")'
        replacement = r'\1\n    AGNOSTIC_RISK_PARITY = "agnostic_risk_parity"'
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Fix 2: Add arp_shrinkage_factor to PortfolioConfig
    if 'arp_shrinkage_factor' not in content:
        print("Adding arp_shrinkage_factor to PortfolioConfig...")
        
        # Find where to insert (after target_volatility)
        pattern = r'(target_volatility: float = 0\.15.*?\n)'
        replacement = r'\1    \n    # ARP-specific parameters\n    arp_shrinkage_factor: Optional[float] = None  # Shrinkage between ARP and Markowitz\n'
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        # Also need to import Optional if not already imported
        if 'from typing import' in content and 'Optional' not in content:
            # Add Optional to existing import
            pattern = r'(from typing import[^\n]+)'
            replacement = lambda m: m.group(1) + ', Optional' if ', Optional' not in m.group(1) else m.group(1)
            content = re.sub(pattern, replacement, content)
    
    # Write the fixed content back
    with open(config_path, 'w') as f:
        f.write(content)
    
    print("Configuration fixed successfully!")
    print("\nChanges made:")
    print("1. Added AGNOSTIC_RISK_PARITY to OptimizationMethod enum")
    print("2. Added arp_shrinkage_factor parameter to PortfolioConfig")
    print("\nYou can now run the migration script again.")

if __name__ == "__main__":
    fix_arp_configuration()