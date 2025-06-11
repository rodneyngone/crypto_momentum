from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="crypto-momentum-backtest",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-grade crypto momentum backtesting system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/crypto-momentum-backtest",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.1.0",
        "vectorbt>=0.24.0",
        "empyrical>=0.5.5",
        "aiohttp>=3.8.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "numba>=0.56.0",
    ],
)