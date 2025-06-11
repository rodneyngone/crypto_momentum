#!/usr/bin/env python3
"""
Script to fetch historical crypto data for backtesting.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_momentum_backtest.data.binance_fetcher import BinanceFetcher
from crypto_momentum_backtest.data.universe_manager import UniverseManager
from crypto_momentum_backtest.utils.logger import setup_logger
from crypto_momentum_backtest.utils.config import Config


async def fetch_all_data():
    """Fetch historical data for all symbols."""
    # Setup
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    logger = setup_logger("data_fetcher", log_dir=Path("logs"))
    
    # Load config
    config = Config.from_yaml(Path("config.yaml"))
    
    # Initialize components
    universe_manager = UniverseManager(
        data_dir,
        universe_size=config.strategy.universe_size,
        logger=logger
    )
    
    fetcher = BinanceFetcher(data_dir, logger=logger)
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)  # 2 years of data
    
    # Get universe
    symbols = universe_manager.get_universe(start_date)
    
    print(f"Fetching data for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")
    print(f"Symbols: {', '.join(symbols[:5])}... and {len(symbols)-5} more")
    
    # Fetch data
    await fetcher.update_universe_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        max_concurrent=3  # Limit concurrent requests
    )
    
    print("\n✅ Data fetching complete!")
    
    # Verify data
    print("\nVerifying downloaded data:")
    for symbol in symbols[:5]:  # Check first 5
        symbol_dir = data_dir / "market_data" / symbol
        if symbol_dir.exists():
            files = list(symbol_dir.glob("*.json.gz"))
            print(f"  {symbol}: {len(files)} files")


def main():
    """Main function."""
    print("🚀 Crypto Data Fetcher")
    print("=" * 50)
    
    try:
        asyncio.run(fetch_all_data())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Binance API might be temporarily unavailable")
        print("3. Try running with fewer symbols or shorter date range")


if __name__ == "__main__":
    main()