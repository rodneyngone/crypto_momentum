#!/usr/bin/env python3
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
