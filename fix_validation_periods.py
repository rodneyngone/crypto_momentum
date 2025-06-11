#!/usr/bin/env python3
"""
Fix the walk-forward validation to use appropriate period lengths.
"""

from pathlib import Path

def fix_validator():
    """Fix the validator to use longer periods for walk-forward analysis."""
    file_path = Path("crypto_momentum_backtest/backtest/validator.py")
    
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix the test_size calculation to ensure minimum period
        old_code = """test_days = int(total_days * self.test_size_ratio)"""
        new_code = """test_days = max(30, int(total_days * self.test_size_ratio))  # Minimum 30 days"""
        
        content = content.replace(old_code, new_code)
        
        # Also fix the train days
        old_train = """train_days = total_days - test_days"""
        new_train = """train_days = max(60, total_days - test_days)  # Minimum 60 days for training"""
        
        content = content.replace(old_train, new_train)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print("✅ Fixed validator.py for minimum period lengths")

def check_current_results():
    """Check if we already have good results from the main backtest."""
    from pathlib import Path
    import pandas as pd
    
    try:
        # Check portfolio values
        portfolio_df = pd.read_csv('output/portfolio_values.csv', index_col=0)
        print("\n📊 Current Results Available:")
        print(f"Portfolio data points: {len(portfolio_df)}")
        print(f"Date range: {portfolio_df.index[0]} to {portfolio_df.index[-1]}")
        
        # Check if it's from the new config
        if len(portfolio_df) > 700:  # Full 2-year backtest
            print("\n✅ Full backtest results are available!")
            print("You can analyze them using: python quick_results_check.py")
        
    except:
        print("\n⚠️ No results found yet")

def main():
    print("🔧 Fixing validation issues...")
    print("=" * 50)
    
    fix_validator()
    check_current_results()
    
    print("\n📋 Next Steps:")
    print("1. If you have results, run: python quick_results_check.py")
    print("2. To run without validation: python run.py --no-validate")
    print("3. To run with fixed validation: python run.py")

if __name__ == "__main__":
    main()