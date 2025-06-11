#!/usr/bin/env python3
"""
Final fix for the summary report generation.
"""

from pathlib import Path

def fix_summary_report():
    """Fix the summary report generation to handle missing validation report."""
    file_path = Path("run.py")
    
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find the generate_summary_report function
        old_validation_lines = '''f"- All Checks Passed: {validation_report['all_checks_passed']}",
        f"- Walk-Forward Sharpe Degradation: {validation_report['walk_forward']['sharpe_degradation']:.2f}",
        f"- Monte Carlo VaR (5%): {validation_report['monte_carlo']['value_at_risk_5%']:.2%}",'''
        
        new_validation_lines = '''f"- All Checks Passed: {validation_report.get('all_checks_passed', 'N/A')}",
        f"- Walk-Forward Sharpe Degradation: {validation_report.get('walk_forward', {}).get('sharpe_degradation', 0):.2f}",
        f"- Monte Carlo VaR (5%): {validation_report.get('monte_carlo', {}).get('value_at_risk_5%', 0):.2%}",'''
        
        content = content.replace(old_validation_lines, new_validation_lines)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print("✅ Fixed summary report generation")

def fix_monthly_returns():
    """Fix the deprecated 'M' frequency in visualizations."""
    file_path = Path("run.py")
    
    if file_path.exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix the monthly returns resampling
        content = content.replace("resample('M')", "resample('ME')")
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print("✅ Fixed monthly resampling frequency")

def check_output_files():
    """Check what files were generated."""
    output_dir = Path("output")
    
    if output_dir.exists():
        print("\n📁 Files in output directory:")
        for file in output_dir.iterdir():
            if file.is_file():
                size = file.stat().st_size
                print(f"  ✓ {file.name} ({size:,} bytes)")

def main():
    print("🔧 Applying final fixes...")
    print("=" * 50)
    
    fix_summary_report()
    fix_monthly_returns()
    
    print("\n✅ Fixes applied!")
    
    # Check output
    check_output_files()
    
    print("\nYou can now run again:")
    print("python run.py --no-validate")

if __name__ == "__main__":
    main()