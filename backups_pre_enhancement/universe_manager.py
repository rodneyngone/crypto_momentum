# crypto_momentum_backtest/data/universe_manager.py
"""Universe selection and management for crypto assets - Fixed version."""
import pandas as pd
import numpy as np
from typing import List, Dict, Set, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from dataclasses import dataclass, asdict


@dataclass
class UniverseSnapshot:
    """Snapshot of universe at a point in time."""
    date: datetime
    symbols: List[str]
    market_caps: Dict[str, float]
    
    def to_dict(self):
        """Convert to dictionary with JSON-serializable types."""
        return {
            'date': self.date.isoformat(),
            'symbols': self.symbols,
            'market_caps': self.market_caps
        }
    
    
class UniverseManager:
    """
    Manages dynamic universe selection based on market cap rankings.
    Includes delisted assets to avoid survivorship bias.
    """
    
    def __init__(
        self,
        data_dir: Path,
        universe_size: int = 20,
        selection_pool_size: int = 50,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize universe manager.
        
        Args:
            data_dir: Directory containing universe data
            universe_size: Number of assets to select
            selection_pool_size: Size of the selection pool
            logger: Logger instance
        """
        self.data_dir = Path(data_dir)
        self.universe_size = universe_size
        self.selection_pool_size = selection_pool_size
        self.logger = logger or logging.getLogger(__name__)
        
        # Delisted assets that must be included historically
        self.delisted_assets = {
            'LUNAUSDT': datetime(2022, 5, 13),  # Terra Luna crash
            'FTTUSDT': datetime(2022, 11, 11),  # FTX collapse
            'CELSIUSUSDT': datetime(2022, 6, 13),  # Celsius bankruptcy
        }
        
        # Cache for universe snapshots
        self._universe_cache: Dict[str, UniverseSnapshot] = {}
        
    def load_market_cap_data(self, date: datetime) -> pd.DataFrame:
        """
        Load market cap data for a specific date.
        
        Args:
            date: Date to load data for
            
        Returns:
            DataFrame with symbol and market_cap columns
        """
        # In production, this would load from a real data source
        # For now, we'll use a synthetic approach
        file_path = self.data_dir / f"market_caps/{date.strftime('%Y-%m')}.json"
        
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        else:
            # Generate synthetic market cap data for demo
            return self._generate_synthetic_market_caps(date)
    
    def _generate_synthetic_market_caps(self, date: datetime) -> pd.DataFrame:
        """Generate synthetic market cap data for testing."""
        # Base symbols that are always in top 50
        base_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
            'XRPUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'SHIBUSDT',
            'MATICUSDT', 'LTCUSDT', 'UNIUSDT', 'LINKUSDT', 'ATOMUSDT',
            'XLMUSDT', 'ETCUSDT', 'NEARUSDT', 'ALGOUSDT', 'VETUSDT',
            'ICPUSDT', 'FILUSDT', 'TRXUSDT', 'AAVEUSDT', 'EOSUSDT',
            'MANAUSDT', 'SANDUSDT', 'THETAUSDT', 'XTZUSDT', 'AXSUSDT'
        ]
        
        # Add more symbols to reach pool size
        additional_symbols = [
            'HBARUSDT', 'EGLDUSDT', 'KLAYUSDT', 'FTMUSDT', 'MIRUSDT',
            'RUNEUSDT', 'NEOUSDT', 'MKRUSDT', 'ZILUSDT', 'BATUSDT',
            'ENJUSDT', 'HOTUSDT', 'ZECUSDT', 'XEMUSDT', 'DCRUSDT',
            'QTUMUSDT', 'ICXUSDT', 'OMGUSDT', 'WAVESUSDT', 'BTGUSDT'
        ]
        
        all_symbols = base_symbols + additional_symbols
        
        # Add delisted assets if before their delisting date
        for symbol, delist_date in self.delisted_assets.items():
            if date < delist_date and symbol not in all_symbols:
                all_symbols.append(symbol)
        
        # Generate market caps with some randomness
        np.random.seed(int(date.timestamp()) % 2**32)
        market_caps = np.random.lognormal(20, 2, len(all_symbols))
        market_caps = np.sort(market_caps)[::-1]  # Sort descending
        
        # Special handling for top coins
        if 'BTCUSDT' in all_symbols:
            idx = all_symbols.index('BTCUSDT')
            market_caps[idx] = market_caps[0] * 2  # BTC is always largest
            
        if 'ETHUSDT' in all_symbols:
            idx = all_symbols.index('ETHUSDT')
            market_caps[idx] = market_caps[0] * 0.4  # ETH is second
        
        df = pd.DataFrame({
            'symbol': all_symbols[:self.selection_pool_size],
            'market_cap': market_caps[:self.selection_pool_size]
        })
        
        return df.sort_values('market_cap', ascending=False).reset_index(drop=True)
    
    def get_universe(self, date: datetime) -> List[str]:
        """
        Get universe of symbols for a specific date.
        
        Args:
            date: Date to get universe for
            
        Returns:
            List of symbols in the universe
        """
        date_key = date.strftime('%Y-%m-%d')
        
        # Check cache
        if date_key in self._universe_cache:
            return self._universe_cache[date_key].symbols
        
        # Load market cap data
        market_cap_df = self.load_market_cap_data(date)
        
        # Select top N by market cap
        universe_df = market_cap_df.nlargest(self.universe_size, 'market_cap')
        symbols = universe_df['symbol'].tolist()
        
        # Cache the snapshot
        self._universe_cache[date_key] = UniverseSnapshot(
            date=date,
            symbols=symbols,
            market_caps=dict(zip(universe_df['symbol'], universe_df['market_cap']))
        )
        
        self.logger.info(f"Universe for {date_key}: {len(symbols)} symbols selected")
        
        return symbols
    
    def get_universe_changes(
        self,
        start_date: datetime,
        end_date: datetime,
        frequency: str = 'M'
    ) -> pd.DataFrame:
        """
        Get universe changes over time.
        
        Args:
            start_date: Start date
            end_date: End date
            frequency: Rebalance frequency
            
        Returns:
            DataFrame with universe changes
        """
        dates = pd.date_range(start_date, end_date, freq=frequency)
        changes = []
        
        prev_universe = set()
        
        for date in dates:
            current_universe = set(self.get_universe(date))
            
            added = current_universe - prev_universe
            removed = prev_universe - current_universe
            
            changes.append({
                'date': date,
                'added': list(added),
                'removed': list(removed),
                'total': len(current_universe)
            })
            
            prev_universe = current_universe
        
        return pd.DataFrame(changes)
    
    def save_universe_history(self, output_path: Path):
        """Save universe history to file."""
        history = {}
        
        for date_key, snapshot in self._universe_cache.items():
            history[date_key] = snapshot.to_dict()
        
        with open(output_path, 'w') as f:
            json.dump(history, f, indent=2)
            
        self.logger.info(f"Saved universe history to {output_path}")