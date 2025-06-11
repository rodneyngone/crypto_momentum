#!/usr/bin/env python3
"""
Simple script to fetch crypto data from Binance.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import gzip
import time


async def fetch_binance_klines(symbol, interval='1d', limit=1000):
    """Fetch klines from Binance API."""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Error fetching {symbol}: HTTP {response.status}")
                    return None
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None


def process_klines(klines):
    """Convert klines to DataFrame."""
    if not klines:
        return pd.DataFrame()
    
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    # Keep only OHLCV
    return df[['open', 'high', 'low', 'close', 'volume']]


def save_monthly_data(df, symbol, data_dir):
    """Save data by month in compressed JSON format."""
    if df.empty:
        return
    
    # Group by month
    monthly_groups = df.groupby(pd.Grouper(freq='M'))
    
    for month_end, month_data in monthly_groups:
        if month_data.empty:
            continue
            
        # Create directory
        symbol_dir = data_dir / "market_data" / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        year_month = month_end.strftime('%Y-%m')
        
        # Convert to JSON-serializable format
        records = []
        for idx, row in month_data.iterrows():
            records.append({
                'timestamp': idx.isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })
        
        file_data = {
            'symbol': symbol,
            'period': year_month,
            'data': records
        }
        
        # Save compressed
        file_path = symbol_dir / f"{year_month}.json.gz"
        with gzip.open(file_path, 'wt', encoding='utf-8') as f:
            json.dump(file_data, f)
        
        print(f"  Saved {symbol} {year_month}: {len(month_data)} days")


async def fetch_and_save_symbol(symbol, data_dir):
    """Fetch and save data for a single symbol."""
    print(f"Fetching {symbol}...")
    
    # Fetch latest 1000 days
    klines = await fetch_binance_klines(symbol)
    
    if klines:
        df = process_klines(klines)
        save_monthly_data(df, symbol, data_dir)
        return True
    return False


async def main():
    """Main function."""
    print("🚀 Simple Crypto Data Fetcher")
    print("=" * 50)
    
    # Setup
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Top symbols to fetch
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
        'XRPUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'SHIBUSDT',
        'MATICUSDT', 'LTCUSDT', 'UNIUSDT', 'LINKUSDT', 'ATOMUSDT',
        'XLMUSDT', 'ETCUSDT', 'NEARUSDT', 'ALGOUSDT', 'VETUSDT'
    ]
    
    print(f"Fetching data for {len(symbols)} symbols")
    print("This will get the latest ~1000 days of data for each symbol\n")
    
    # Fetch data for each symbol with delay
    success_count = 0
    for symbol in symbols:
        success = await fetch_and_save_symbol(symbol, data_dir)
        if success:
            success_count += 1
        
        # Rate limit - wait 0.5 seconds between requests
        await asyncio.sleep(0.5)
    
    print(f"\n✅ Fetching complete! Successfully fetched {success_count}/{len(symbols)} symbols")
    
    # Verify
    print("\nVerifying downloaded data:")
    for symbol in symbols[:5]:
        symbol_dir = data_dir / "market_data" / symbol
        if symbol_dir.exists():
            files = list(symbol_dir.glob("*.json.gz"))
            print(f"  {symbol}: {len(files)} monthly files")


if __name__ == "__main__":
    asyncio.run(main())