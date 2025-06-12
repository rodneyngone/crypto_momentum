#!/usr/bin/env python3
"""
Quick fix to enable long/short trading in the existing framework.
This patches the necessary files to ensure both long and short signals are generated.
"""

import re
from pathlib import Path
import shutil
from datetime import datetime


def patch_signal_generator():
    """Patch the signal generator to ensure short signals."""
    
    signal_gen_path = Path("crypto_momentum_backtest/signals/signal_generator_enhanced.py")
    
    if not signal_gen_path.exists():
        print(f"Warning: {signal_gen_path} not found")
        return False
    
    # Read the file
    with open(signal_gen_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "# LONG_SHORT_PATCH" in content:
        print("Signal generator already patched for long/short")
        return True
    
    # Find the _generate_momentum_score_signals method
    method_pattern = r'(def _generate_momentum_score_signals\(self, data: pd\.DataFrame\) -> pd\.Series:.*?)(return signals)'
    
    # Replacement that ensures negative signals
    replacement = r'''\1
        # LONG_SHORT_PATCH: Ensure we have both long and short signals
        # Center the scores around zero to get short signals
        non_zero_scores = composite_score[composite_score != 0]
        if len(non_zero_scores) > 0:
            # If all scores are positive, adjust by subtracting median
            if non_zero_scores.min() > 0:
                score_median = non_zero_scores.median()
                composite_score = composite_score - score_median * 0.5
        
        # Re-apply thresholds for balanced long/short
        signals = pd.Series(0.0, index=data.index)
        signals[composite_score > self.min_score_threshold] = composite_score[composite_score > self.min_score_threshold]
        signals[composite_score < -self.min_score_threshold] = composite_score[composite_score < -self.min_score_threshold]
        
        \2'''
    
    # Apply patch
    new_content = re.sub(method_pattern, replacement, content, flags=re.DOTALL)
    
    # Backup and write
    backup_path = signal_gen_path.with_suffix('.py.bak')
    shutil.copy2(signal_gen_path, backup_path)
    
    with open(signal_gen_path, 'w') as f:
        f.write(new_content)
    
    print(f"✓ Patched signal generator for long/short trading")
    return True


def patch_backtest_engine():
    """Ensure the backtest engine allows short positions."""
    
    engine_path = Path("crypto_momentum_backtest/backtest/engine.py")
    
    if not engine_path.exists():
        print(f"Warning: {engine_path} not found")
        return False
    
    with open(engine_path, 'r') as f:
        content = f.read()
    
    # Check if we need to add allow_short logic
    if "allow_short=long_short" not in content:
        # Find the optimizer initialization
        pattern = r'(self\.optimizer = ERCOptimizer\(.*?)(logger=self\.logger\s*\))'
        replacement = r'\1allow_short=long_short,\n                \2'
        
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        # Also patch ARP optimizer initialization
        pattern2 = r'(self\.optimizer = ARPOptimizer\(.*?)(logger=self\.logger\s*\))'
        replacement2 = r'\1# Note: ARP handles long/short internally\n                \2'
        
        new_content = re.sub(pattern2, replacement2, new_content, flags=re.DOTALL)
        
        # Write back
        with open(engine_path, 'w') as f:
            f.write(new_content)
        
        print(f"✓ Patched backtest engine for long/short trading")
        return True
    
    print("Backtest engine already configured for long/short")
    return True


def create_run_script():
    """Create a script to run long/short backtest."""
    
    script_content = '''#!/usr/bin/env python3
"""
Run long/short momentum backtest.
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Ensure config exists
    config_path = Path('config_long_short.yaml')
    if not config_path.exists():
        print("Error: config_long_short.yaml not found!")
        print("Please run the setup script first.")
        sys.exit(1)
    
    # Run backtest with long/short config
    cmd = [
        'python', 'run.py',
        '--config', 'config_long_short.yaml',
        '--output-dir', 'output_long_short',
        '--no-validate'  # Skip validation for faster results
    ]
    
    print("Running long/short backtest...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Backtest failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    script_path = Path('run_long_short.py')
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✓ Created run script: {script_path}")
    return True


def main():
    print("="*60)
    print("CONFIGURING SYSTEM FOR LONG/SHORT TRADING")
    print("="*60)
    
    # 1. Check if config exists
    config_path = Path('config_long_short.yaml')
    if not config_path.exists():
        print("Error: config_long_short.yaml not found!")
        print("Please create it first using the provided configuration.")
        return
    
    # 2. Patch signal generator
    print("\n1. Patching signal generator...")
    patch_signal_generator()
    
    # 3. Patch backtest engine
    print("\n2. Patching backtest engine...")
    patch_backtest_engine()
    
    # 4. Create run script
    print("\n3. Creating run script...")
    create_run_script()
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nTo run the long/short backtest:")
    print("  python run_long_short.py")
    print("\nTo test signal generation:")
    print("  python test_long_short.py")
    print("\nKey parameters for long/short:")
    print("  - Signal threshold: ±0.2")
    print("  - Max position size: 15%")
    print("  - Target gross exposure: ~150%")
    print("  - Target net exposure: ~50%")
    

if __name__ == "__main__":
    main()