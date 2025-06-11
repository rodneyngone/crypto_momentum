#!/usr/bin/env python3
"""
Fix the validator index error.
"""

from pathlib import Path

def fix_validator():
    """Fix the index error in validator.py"""
    file_path = Path("crypto_momentum_backtest/backtest/validator.py")
    
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix the index error
        old_code = """for i in range(1, len(positions) - 1):
                future_return = returns.iloc[i+1]"""
        
        new_code = """for i in range(1, min(len(positions) - 1, len(returns) - 1)):
                if i + 1 < len(returns):
                    future_return = returns.iloc[i+1]
                else:
                    continue"""
        
        content = content.replace(old_code, new_code)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print("✅ Fixed validator.py")

def fix_metrics():
    """Fix the deprecated frequency strings in metrics.py"""
    file_path = Path("crypto_momentum_backtest/backtest/metrics.py")
    
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix deprecated frequency strings
        content = content.replace(".resample('M')", ".resample('ME')")
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print("✅ Fixed metrics.py frequency strings")

def main():
    print("🔧 Fixing remaining issues...")
    print("=" * 50)
    
    fix_validator()
    fix_metrics()
    
    print("\n✅ Fixes applied!")
    print("\nNow you can run:")
    print("python run.py")

if __name__ == "__main__":
    main()