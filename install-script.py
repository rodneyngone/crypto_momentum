#!/usr/bin/env python3
"""
Installation helper script for crypto momentum backtest system.
Handles Python 3.13 compatibility issues.
"""

import subprocess
import sys
import os
from pathlib import Path


def check_python_version():
    """Check Python version and warn if using 3.12+."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 12:
        print("\n⚠️  WARNING: Python 3.12+ detected!")
        print("Some packages (empyrical) are not compatible with Python 3.12+")
        print("Using alternative packages instead.")
        return True
    return False


def install_requirements():
    """Install requirements with proper handling."""
    print("\n📦 Installing requirements...")
    
    # Core packages that should work
    core_packages = [
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "aiohttp>=3.8.0",
        "requests>=2.31.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "pyyaml>=6.0",
        "numba>=0.57.0",
        "tqdm>=4.65.0",
        "pytest>=7.4.0",
        "black>=23.7.0",
        "flake8>=6.1.0",
        "mypy>=1.5.0"
    ]
    
    # Install core packages
    for package in core_packages:
        print(f"\nInstalling {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
    
    # Try to install vectorbt
    print("\n📊 Installing vectorbt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "vectorbt>=0.24.0"])
    except subprocess.CalledProcessError:
        print("❌ Failed to install vectorbt")
        print("You may need to install it manually or use an alternative")
    
    # For metrics, use quantstats instead of empyrical
    print("\n📈 Installing metrics libraries...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "quantstats>=0.0.62"])
        print("✅ Installed quantstats as metrics library")
    except subprocess.CalledProcessError:
        print("❌ Failed to install quantstats")


def create_alternative_requirements():
    """Create alternative requirements file for Python 3.12+."""
    content = """# requirements_py312.txt - For Python 3.12+ compatibility

# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.24.0
quantstats>=0.0.62  # Alternative to empyrical

# Data fetching
aiohttp>=3.8.0
requests>=2.31.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Utils
pyyaml>=6.0
numba>=0.57.0
tqdm>=4.65.0

# Development
pytest>=7.4.0
black>=23.7.0
flake8>=6.1.0
mypy>=1.5.0
"""
    
    with open("requirements_py312.txt", "w") as f:
        f.write(content)
    
    print("\n✅ Created requirements_py312.txt for Python 3.12+ compatibility")


def update_metrics_import():
    """Update the backtest/metrics.py file to remove empyrical import."""
    metrics_file = Path("crypto_momentum_backtest/backtest/metrics.py")
    
    if metrics_file.exists():
        print("\n🔧 Updating metrics.py to remove empyrical dependency...")
        # The metrics.py content is already updated in the artifact above
        print("✅ Please copy the updated metrics.py from the artifact above")
    else:
        print("\n⚠️  metrics.py not found. Make sure to use the updated version without empyrical")


def main():
    """Main installation process."""
    print("🚀 Crypto Momentum Backtest Installation Helper")
    print("=" * 50)
    
    # Check Python version
    is_py312_plus = check_python_version()
    
    if is_py312_plus:
        # Create alternative requirements
        create_alternative_requirements()
        
        # Install packages
        install_requirements()
        
        # Update imports
        update_metrics_import()
        
        print("\n" + "=" * 50)
        print("✅ Installation complete!")
        print("\n⚠️  IMPORTANT NOTES:")
        print("1. Replace crypto_momentum_backtest/backtest/metrics.py with the updated version")
        print("2. Use requirements_py312.txt instead of requirements.txt")
        print("3. Some features may require manual adjustments")
    else:
        # Standard installation
        print("\n✅ Python version is compatible with all packages")
        print("Run: pip install -r requirements.txt")
    
    print("\n📋 Next steps:")
    print("1. Copy all module files from the main artifact")
    print("2. Run: python -m crypto_momentum_backtest.main")
    print("3. Check output/ directory for results")


if __name__ == "__main__":
    main()
