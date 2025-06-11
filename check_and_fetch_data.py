#!/usr/bin/env python3
"""
Check what data is available and fetch some if needed.
"""

from pathlib import Path
import json
import gzip
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import aiohttp


def check_data_directory():
    """Check what data files are available."""
    
    data_dir = Path("data")
    print(f"Checking data directory: {data_dir.absolute()}")
    print("=" * 60)
    
    if not data_dir.exists():
        print("[WARNING] Data directory doesn't exist!")
        data_dir.mkdir(exist_ok=True)
        print("[OK] Created data directory")
        return False
    
    # Check for market_data subdirectory
    market_data_dir = data_dir / "market_data"
    if not market_data_dir.exists():
        print("[WARNING] No market_data subdirectory found")
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


def create_sample_data():
    """Create sample data files for testing."""
    
    print("\nCreating sample data for testing...")
    
    # Create directory structure
    data_dir = Path("data")
    market_data_dir = data_dir / "market_data"
    market_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Symbols to create
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    # Generate 2 years of data
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    for symbol in symbols:
        symbol_dir = market_data_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        
        # Initial prices
        initial_prices = {
            'BTCUSDT': 40000,
            'ETHUSDT': 3000,
            'BNBUSDT': 400
        }
        
        current_date = start_date
        
        while current_date <= end_date:
            # Create monthly file
            year_month = current_date.strftime('%Y-%m')
            
            # Generate daily data for the month
            month_data = []
            days_in_month = 30  # Simplified
            
            for day in range(days_in_month):
                date = current_date + timedelta(days=day)
                if date > end_date:
                    break
                
                # Generate realistic price movement
                base_price = initial_prices[symbol]
                # Random walk with trend
                trend = 0.0002 * day  # Slight upward trend
                volatility = 0.02  # 2% daily volatility
                
                close_price = base_price * (1 + trend + np.random.normal(0, volatility))
                
                daily_data = {
                    'timestamp': date.isoformat(),
                    'open': float(close_price * np.random.uniform(0.99, 1.01)),
                    'high': float(close_price * np.random.uniform(1.0, 1.02)),
                    'low': float(close_price * np.random.uniform(0.98, 1.0)),
                    'close': float(close_price),
                    'volume': float(np.random.uniform(1e8, 1e10))
                }
                
                month_data.append(daily_data)
            
            # Save compressed JSON
            file_data = {
                'symbol': symbol,
                'period': year_month,
                'data': month_data
            }
            
            file_path = symbol_dir / f"{year_month}.json.gz"
            with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                json.dump(file_data, f)
            
            print(f"[OK] Created {symbol} {year_month}")
            
            # Move to next month
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
    
    print("\n[OK] Sample data created successfully!")


async def fetch_real_binance_data(symbol='BTCUSDT', days=30):
    """Fetch recent data from Binance API."""
    
    print(f"\nFetching {days} days of {symbol} data from Binance...")
    
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': '1d',
        'limit': min(days, 1000)
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    klines = await response.json()
                    
                    # Convert to our format
                    data_dir = Path("data") / "market_data" / symbol
                    data_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Group by month
                    monthly_data = {}
                    
                    for kline in klines:
                        timestamp = datetime.fromtimestamp(kline[0] / 1000)
                        year_month = timestamp.strftime('%Y-%m')
                        
                        if year_month not in monthly_data:
                            monthly_data[year_month] = []
                        
                        daily_data = {
                            'timestamp': timestamp.isoformat(),
                            'open': float(kline[1]),
                            'high': float(kline[2]),
                            'low': float(kline[3]),
                            'close': float(kline[4]),
                            'volume': float(kline[5])
                        }
                        
                        monthly_data[year_month].append(daily_data)
                    
                    # Save each month
                    for year_month, data in monthly_data.items():
                        file_data = {
                            'symbol': symbol,
                            'period': year_month,
                            'data': data
                        }
                        
                        file_path = data_dir / f"{year_month}.json.gz"
                        with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                            json.dump(file_data, f)
                        
                        print(f"[OK] Saved {symbol} {year_month}: {len(data)} days")
                    
                    return True
                else:
                    print(f"[ERROR] Binance API returned status {response.status}")
                    return False
                    
        except Exception as e:
            print(f"[ERROR] Failed to fetch data: {e}")
            return False


def test_data_loading():
    """Test if data can be loaded properly."""
    
    print("\nTesting data loading...")
    
    from crypto_momentum_backtest.data.json_storage import JsonStorage
    
    storage = JsonStorage(Path('data'))
    
    # Try to load BTCUSDT data
    df = storage.load_range(
        symbol='BTCUSDT',
        start_date=pd.Timestamp('2022-01-01'),
        end_date=pd.Timestamp('2022-12-31')
    )
    
    if not df.empty:
        print(f"[OK] Successfully loaded {len(df)} days of BTCUSDT data")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"Columns: {list(df.columns)}")
        print(f"Sample data:")
        print(df.head())
        return True
    else:
        print("[ERROR] Failed to load data")
        return False


async def main():
    """Main function."""
    print("Data Check and Setup")
    print("=" * 60)
    
    # Check existing data
    has_data = check_data_directory()
    
    if not has_data:
        print("\nNo data found. Choose an option:")
        print("1. Create sample data for testing")
        print("2. Fetch real data from Binance")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            create_sample_data()
        elif choice == '2':
            # Fetch real data
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            for symbol in symbols:
                await fetch_real_binance_data(symbol, days=60)
        else:
            print("Exiting...")
            return
    
    # Test data loading
    test_data_loading()
    
    print("\n[OK] Data setup complete!")
    print("\nYou can now run:")
    print("  python test_signal_generation.py")
    print("  python run.py --no-validate")


if __name__ == "__main__":
    asyncio.run(main())