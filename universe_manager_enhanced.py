# crypto_momentum_backtest/data/universe_manager_enhanced.py
"""Enhanced universe selection with momentum filtering and category diversification."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
import logging


class EnhancedUniverseManager:
    """
    Enhanced universe manager with dynamic selection based on momentum,
    category diversification, and quality filters.
    """
    
    def __init__(
        self,
        data_dir: Path,
        base_universe_size: int = 30,  # Increased from 10
        selection_universe_size: int = 15,  # Final selected universe
        use_momentum_filter: bool = True,
        momentum_lookback: int = 30,
        category_limits: Optional[Dict[str, float]] = None,
        min_market_cap: float = 100_000_000,  # $100M minimum
        min_volume_avg: float = 10_000_000,    # $10M daily volume
        exclude_stablecoins: bool = True,
        exclude_wrapped: bool = True,
        quality_score_threshold: float = 0.5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize enhanced universe manager.
        
        Args:
            data_dir: Data directory path
            base_universe_size: Initial universe size before filtering
            selection_universe_size: Final universe size after momentum filter
            use_momentum_filter: Whether to filter by momentum
            momentum_lookback: Days to look back for momentum calculation
            category_limits: Maximum allocation per category
            min_market_cap: Minimum market cap requirement
            min_volume_avg: Minimum average daily volume
            exclude_stablecoins: Whether to exclude stablecoins
            exclude_wrapped: Whether to exclude wrapped tokens
            quality_score_threshold: Minimum quality score
            logger: Logger instance
        """
        self.data_dir = Path(data_dir)
        self.base_universe_size = base_universe_size
        self.selection_universe_size = selection_universe_size
        self.use_momentum_filter = use_momentum_filter
        self.momentum_lookback = momentum_lookback
        self.min_market_cap = min_market_cap
        self.min_volume_avg = min_volume_avg
        self.exclude_stablecoins = exclude_stablecoins
        self.exclude_wrapped = exclude_wrapped
        self.quality_score_threshold = quality_score_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        # Default category limits
        self.category_limits = category_limits or {
            'defi': 0.40,        # Max 40% DeFi
            'layer1': 0.50,      # Max 50% Layer 1
            'layer2': 0.30,      # Max 30% Layer 2
            'meme': 0.20,        # Max 20% meme coins
            'exchange': 0.30,    # Max 30% exchange tokens
            'gaming': 0.20,      # Max 20% gaming
            'ai': 0.20,          # Max 20% AI tokens
            'other': 0.30        # Max 30% other
        }
        
        # Asset categorization (expand as needed)
        self.asset_categories = {
            'defi': ['UNI', 'AAVE', 'SUSHI', 'COMP', 'MKR', 'SNX', 'CRV', 'YFI', '1INCH'],
            'layer1': ['BTC', 'ETH', 'SOL', 'ADA', 'AVAX', 'DOT', 'ATOM', 'NEAR', 'ALGO'],
            'layer2': ['MATIC', 'ARB', 'OP', 'IMX'],
            'meme': ['DOGE', 'SHIB', 'FLOKI', 'PEPE'],
            'exchange': ['BNB', 'CRO', 'KCS', 'OKB', 'FTT'],
            'gaming': ['AXS', 'SAND', 'MANA', 'ENJ', 'GALA'],
            'ai': ['FET', 'OCEAN', 'RNDR', 'AGIX']
        }
        
        # Stablecoins to exclude
        self.stablecoins = {
            'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'GUSD', 'FRAX', 'LUSD'
        }
        
        # Wrapped tokens to exclude
        self.wrapped_tokens = {
            'WBTC', 'WETH', 'WBNB', 'WMATIC', 'WFTM', 'WAVAX'
        }
        
        # Cache for universe selection
        self._universe_cache = {}
    
    def get_universe(
        self,
        date: datetime,
        force_refresh: bool = False
    ) -> List[str]:
        """
        Get the universe of tradeable assets for a given date with momentum filtering.
        
        Args:
            date: Date for universe selection
            force_refresh: Force recalculation even if cached
            
        Returns:
            List of symbols in the universe
        """
        # Check cache
        cache_key = date.strftime('%Y-%m-%d')
        if not force_refresh and cache_key in self._universe_cache:
            return self._universe_cache[cache_key]
        
        # Get base universe by market cap
        base_universe = self._get_market_cap_universe(date)
        
        # Apply quality filters
        quality_universe = self._apply_quality_filters(base_universe, date)
        
        # Apply momentum filter if enabled
        if self.use_momentum_filter and len(quality_universe) > self.selection_universe_size:
            universe = self._apply_momentum_filter(quality_universe, date)
        else:
            universe = quality_universe[:self.selection_universe_size]
        
        # Apply category diversification
        universe = self._apply_category_diversification(universe)
        
        # Cache result
        self._universe_cache[cache_key] = universe
        
        self.logger.info(
            f"Universe for {date.date()}: {len(universe)} symbols selected "
            f"from {len(base_universe)} candidates"
        )
        
        return universe
    
    def _get_market_cap_universe(self, date: datetime) -> List[str]:
        """Get top assets by market cap."""
        # In production, this would query real market cap data
        # For now, return a predefined list based on typical rankings
        
        default_universe = [
            'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'DOT', 'MATIC', 'AVAX',
            'SHIB', 'TRX', 'LINK', 'UNI', 'ATOM', 'LTC', 'ETC', 'FIL', 'NEAR', 'APT',
            'ARB', 'VET', 'OP', 'ALGO', 'GRT', 'FTM', 'SAND', 'AXS', 'MANA', 'AAVE',
            'SNX', 'CRV', 'MKR', 'COMP', 'LDO', 'RNDR', 'IMX', 'HBAR', 'EGLD', 'THETA'
        ]
        
        # Filter out excluded categories
        filtered = []
        for symbol in default_universe:
            # Remove 'USDT' suffix if present
            clean_symbol = symbol.replace('USDT', '')
            
            if self.exclude_stablecoins and clean_symbol in self.stablecoins:
                continue
            if self.exclude_wrapped and clean_symbol in self.wrapped_tokens:
                continue
                
            filtered.append(symbol)
        
        return filtered[:self.base_universe_size]
    
    def _apply_quality_filters(
        self,
        symbols: List[str],
        date: datetime
    ) -> List[str]:
        """Apply quality filters based on volume, volatility, and other metrics."""
        quality_scores = {}
        
        for symbol in symbols:
            # Calculate quality score (mock implementation)
            # In production, this would use real data
            score = self._calculate_quality_score(symbol, date)
            
            if score >= self.quality_score_threshold:
                quality_scores[symbol] = score
        
        # Sort by quality score
        sorted_symbols = sorted(
            quality_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [symbol for symbol, _ in sorted_symbols]
    
    def _calculate_quality_score(self, symbol: str, date: datetime) -> float:
        """
        Calculate quality score for an asset.
        
        Factors:
        - Volume consistency
        - Price stability
        - Listing duration
        - Exchange coverage
        """
        # Mock implementation - in production would use real metrics
        base_scores = {
            'BTC': 0.95, 'ETH': 0.93, 'BNB': 0.85, 'SOL': 0.82,
            'ADA': 0.80, 'AVAX': 0.78, 'MATIC': 0.77, 'DOT': 0.76,
            'LINK': 0.75, 'UNI': 0.74, 'ATOM': 0.73, 'NEAR': 0.72
        }
        
        # Default score with some randomness
        base = base_scores.get(symbol.replace('USDT', ''), 0.60)
        
        # Add some variation
        variation = np.random.uniform(-0.05, 0.05)
        
        return max(0, min(1, base + variation))
    
    def _apply_momentum_filter(
        self,
        symbols: List[str],
        date: datetime
    ) -> List[str]:
        """Filter universe by momentum scores."""
        momentum_scores = {}
        
        for symbol in symbols:
            try:
                # Load price data
                price_data = self._load_price_data(symbol, date)
                
                if price_data is not None and len(price_data) >= self.momentum_lookback:
                    # Calculate momentum
                    momentum = self._calculate_momentum(price_data)
                    momentum_scores[symbol] = momentum
                else:
                    # No data, assign neutral score
                    momentum_scores[symbol] = 0.0
                    
            except Exception as e:
                self.logger.warning(f"Error calculating momentum for {symbol}: {e}")
                momentum_scores[symbol] = 0.0
        
        # Sort by momentum and select top N
        sorted_symbols = sorted(
            momentum_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        selected = [symbol for symbol, _ in sorted_symbols[:self.selection_universe_size]]
        
        self.logger.info(
            f"Momentum filter selected {len(selected)} from {len(symbols)} symbols. "
            f"Top momentum: {sorted_symbols[0][0]} ({sorted_symbols[0][1]:.2%})"
        )
        
        return selected
    
    def _calculate_momentum(self, price_data: pd.DataFrame) -> float:
        """Calculate momentum score for an asset."""
        if len(price_data) < self.momentum_lookback:
            return 0.0
        
        # Simple momentum: return over lookback period
        current_price = price_data['close'].iloc[-1]
        lookback_price = price_data['close'].iloc[-self.momentum_lookback]
        
        if lookback_price > 0:
            momentum = (current_price / lookback_price) - 1
            
            # Adjust for volatility (risk-adjusted momentum)
            returns = price_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            
            if volatility > 0:
                # Sharpe-like momentum score
                risk_adjusted_momentum = momentum / volatility
                return risk_adjusted_momentum
            else:
                return momentum
        
        return 0.0
    
    def _apply_category_diversification(self, symbols: List[str]) -> List[str]:
        """Apply category limits to ensure diversification."""
        # Categorize symbols
        categorized = self._categorize_symbols(symbols)
        
        # Track selected symbols and their categories
        selected = []
        category_counts = {cat: 0 for cat in self.category_limits}
        
        # First pass: ensure minimum representation
        for category, assets in categorized.items():
            if assets and category in self.category_limits:
                # Add at least one from each represented category
                if len(assets) > 0:
                    selected.append(assets[0])
                    category_counts[category] = 1
        
        # Second pass: fill remaining slots respecting limits
        remaining_slots = self.selection_universe_size - len(selected)
        
        # Create pool of remaining candidates
        remaining_candidates = []
        for category, assets in categorized.items():
            for asset in assets[1:]:  # Skip first as already selected
                if asset not in selected:
                    remaining_candidates.append((asset, category))
        
        # Sort by original order (preserves momentum ranking)
        remaining_candidates = sorted(
            remaining_candidates,
            key=lambda x: symbols.index(x[0]) if x[0] in symbols else float('inf')
        )
        
        # Fill remaining slots
        for asset, category in remaining_candidates:
            if len(selected) >= self.selection_universe_size:
                break
                
            # Check category limit
            category_weight = category_counts[category] / self.selection_universe_size
            
            if category_weight < self.category_limits.get(category, 1.0):
                selected.append(asset)
                category_counts[category] += 1
        
        # Log category distribution
        self._log_category_distribution(selected, category_counts)
        
        return selected
    
    def _categorize_symbols(self, symbols: List[str]) -> Dict[str, List[str]]:
        """Categorize symbols into predefined categories."""
        categorized = {cat: [] for cat in self.asset_categories}
        categorized['other'] = []
        
        for symbol in symbols:
            # Remove 'USDT' suffix for matching
            clean_symbol = symbol.replace('USDT', '')
            
            found = False
            for category, assets in self.asset_categories.items():
                if clean_symbol in assets:
                    categorized[category].append(symbol)
                    found = True
                    break
            
            if not found:
                categorized['other'].append(symbol)
        
        # Remove empty categories
        return {k: v for k, v in categorized.items() if v}
    
    def _log_category_distribution(
        self,
        selected: List[str],
        category_counts: Dict[str, int]
    ):
        """Log the category distribution of selected universe."""
        distribution = []
        
        for category, count in category_counts.items():
            if count > 0:
                weight = count / len(selected)
                distribution.append(f"{category}: {count} ({weight:.1%})")
        
        self.logger.info(f"Category distribution: {', '.join(distribution)}")
    
    def _load_price_data(
        self,
        symbol: str,
        date: datetime
    ) -> Optional[pd.DataFrame]:
        """Load price data for momentum calculation."""
        # This is a simplified version - in production would use JsonStorage
        try:
            # Add USDT suffix if not present
            if not symbol.endswith('USDT'):
                symbol = symbol + 'USDT'
            
            # Mock implementation - return synthetic data
            dates = pd.date_range(
                end=date,
                periods=self.momentum_lookback + 1,
                freq='D'
            )
            
            # Generate synthetic prices with some trend
            base_price = 100
            trend = np.random.uniform(-0.001, 0.003)  # Daily trend
            volatility = np.random.uniform(0.02, 0.04)  # Daily volatility
            
            prices = []
            for i in range(len(dates)):
                price = base_price * (1 + trend) ** i
                price *= (1 + np.random.normal(0, volatility))
                prices.append(price)
            
            return pd.DataFrame({
                'close': prices
            }, index=dates)
            
        except Exception as e:
            self.logger.error(f"Error loading price data for {symbol}: {e}")
            return None
    
    def get_universe_metrics(
        self,
        universe: List[str],
        date: datetime
    ) -> Dict:
        """Get metrics about the selected universe."""
        categorized = self._categorize_symbols(universe)
        
        # Calculate diversity score (entropy)
        category_weights = []
        for category, assets in categorized.items():
            if assets:
                weight = len(assets) / len(universe)
                category_weights.append(weight)
        
        # Shannon entropy
        diversity_score = -sum(w * np.log(w) for w in category_weights if w > 0)
        max_entropy = -np.log(1 / len(category_weights)) if category_weights else 0
        normalized_diversity = diversity_score / max_entropy if max_entropy > 0 else 0
        
        return {
            'universe_size': len(universe),
            'categories': len(categorized),
            'diversity_score': normalized_diversity,
            'category_breakdown': {
                cat: len(assets) for cat, assets in categorized.items()
            },
            'top_symbols': universe[:5]
        }
    
    def validate_symbol(self, symbol: str) -> Tuple[bool, str]:
        """
        Validate if a symbol meets quality criteria.
        
        Returns:
            Tuple of (is_valid, reason)
        """
        clean_symbol = symbol.replace('USDT', '')
        
        if self.exclude_stablecoins and clean_symbol in self.stablecoins:
            return False, "Stablecoin excluded"
        
        if self.exclude_wrapped and clean_symbol in self.wrapped_tokens:
            return False, "Wrapped token excluded"
        
        # Additional validation could include:
        # - Market cap check
        # - Volume check
        # - Listing duration
        
        return True, "Valid"