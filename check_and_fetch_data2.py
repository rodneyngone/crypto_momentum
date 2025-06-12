#!/usr/bin/env python3
"""
Check what data is available and fetch if needed.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_momentum_backtest.data.binance_fetcher import BinanceFetcher
from crypto_momentum_backtest.utils.logger import setup_logger
import asyncio


def check_data_directory():
    """Check what data files are available."""
    data_dir = Path("data")
    market_data_dir = data_dir / "market_data"
    
    print(f"Checking data directory: {data_dir.absolute()}")
    print("=" * 60)
    
    if not data_dir.exists():
        print("[WARNING] Data directory doesn't exist!")
        data_dir.mkdir(exist_ok=True)
        print("[OK] Created data directory")
        return False
    
    if not market_data_dir.exists():
        print("[WARNING] No market_data subdirectory found")
        market_data_dir.mkdir(exist_ok=True)
        print("[OK] Created market_data directory")
        return False
    
    # List all symbols
    symbols_found = []
    for symbol_dir in market_data_dir.iterdir():
        if symbol_dir.is_dir():
            files = list(symbol_dir.glob("*.json.gz"))
            if files:
                symbols_found.append(symbol_dir.name)
                print(f"[OK] {symbol_dir.name}: {len(files)} files")
                # Show first few files
                for f in sorted(files)[:3]:
                    print(f"     - {f.name}")
    
    if not symbols_found:
        print("[WARNING] No data files found")
        return False
    
    print(f"\n[OK] Found data for {len(symbols_found)} symbols")
    return True


async def fetch_sample_data():
    """Fetch sample data for testing."""
    logger = setup_logger("data_fetcher", "INFO")
    data_dir = Path("data")
    
    # Create fetcher
    fetcher = BinanceFetcher(data_dir, logger=logger)
    
    # List of symbols to fetch (from your universe)
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
        'AVAXUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT', 'ATOMUSDT',
        'LTCUSDT', 'ETCUSDT', 'XLMUSDT', 'TRXUSDT', 'ALGOUSDT',
        'AAVEUSDT', 'UNIUSDT', 'AXSUSDT', 'SANDUSDT', 'MANAUSDT',
        'FTMUSDT', 'NEARUSDT', 'FLOWUSDT', 'THETAUSDT', 'ICPUSDT',
        'VETUSDT', 'FILUSDT', 'ARBUSDT', 'OPUSDT', 'SHIBUSDT'
    ]
    
    # Date range
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2025, 6, 11)
    
    print(f"\nFetching data for {len(symbols)} symbols from {start_date} to {end_date}")
    print("This may take a few minutes...")
    
    # Fetch data
    await fetcher.update_universe_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        max_concurrent=5
    )
    
    print("\n[OK] Data fetching complete!")


async def main():
    """Main function."""
    print("Crypto Momentum Backtest - Data Check")
    print("=" * 60)
    
    # Check existing data
    has_data = check_data_directory()
    
    if not has_data:
        print("\nNo data found. Would you like to fetch historical data?")
        print("This will download 2 years of daily data for 30 cryptocurrencies.")
        print("Data source: Binance API (free)")
        
        choice = input("\nFetch data? (y/n): ").strip().lower()
        
        if choice == 'y':
            await fetch_sample_data()
            
            # Check again
            print("\nRechecking data directory...")
            check_data_directory()
        else:
            print("\nTo run the backtest, you need historical data.")
            print("You can fetch data by running:")
            print("  python run.py --fetch-data --no-validate")
    else:
        print("\n[OK] Data is available. You can run the backtest with:")
        print("  python run.py --no-validate")


if __name__ == "__main__":
    asyncio.run(main())