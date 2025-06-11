# crypto_momentum_backtest/portfolio/rebalancer_enhanced.py
"""Enhanced portfolio rebalancing with dynamic triggers and regime awareness."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging


class EnhancedRebalancer:
    """
    Enhanced rebalancing logic with dynamic triggers based on market conditions.
    """
    
    def __init__(
        self,
        base_rebalance_frequency: str = 'weekly',
        deviation_thresholds: Dict[str, float] = None,
        min_trade_size: float = 0.005,  # 0.5% minimum trade
        max_turnover: Dict[str, float] = None,
        use_dynamic_rebalancing: bool = True,
        momentum_preservation: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize enhanced rebalancer.
        
        Args:
            base_rebalance_frequency: Base rebalancing frequency
            deviation_thresholds: Regime-specific deviation thresholds
            min_trade_size: Minimum trade size as fraction of portfolio
            max_turnover: Maximum allowed turnover by regime
            use_dynamic_rebalancing: Whether to use dynamic rebalancing
            momentum_preservation: Whether to preserve winning positions
            logger: Logger instance
        """
        self.base_rebalance_frequency = base_rebalance_frequency
        
        # Default deviation thresholds by market regime
        self.deviation_thresholds = deviation_thresholds or {
            'trending': 0.05,    # 5% deviation - rebalance quickly in trends
            'ranging': 0.15,     # 15% deviation - less frequent in ranging
            'volatile': 0.10,    # 10% deviation - moderate in volatile
            'crisis': 0.08       # 8% deviation - tighter control in crisis
        }
        
        self.min_trade_size = min_trade_size
        
        # Maximum turnover limits by regime
        self.max_turnover = max_turnover or {
            'trending': 0.30,    # 30% turnover allowed in trends
            'ranging': 0.15,     # 15% in ranging markets
            'volatile': 0.20,    # 20% in volatile
            'crisis': 0.10       # 10% in crisis - preserve capital
        }
        
        self.use_dynamic_rebalancing = use_dynamic_rebalancing
        self.momentum_preservation = momentum_preservation
        self.logger = logger or logging.getLogger(__name__)
        
        # Map frequency to pandas offset
        self.frequency_map = {
            'daily': 'D',
            'weekly': 'W',
            'biweekly': '2W',
            'monthly': 'ME',
            'quarterly': 'QE'
        }
        
        # Dynamic frequency adjustments by regime
        self.regime_frequency_map = {
            'trending': {
                'daily': 'daily',
                'weekly': 'daily',      # Speed up in trends
                'biweekly': 'weekly',
                'monthly': 'weekly'
            },
            'volatile': {
                'daily': 'daily',
                'weekly': 'weekly',
                'biweekly': 'biweekly',
                'monthly': 'biweekly'  # More frequent in volatile
            },
            'ranging': {
                'daily': 'weekly',      # Slow down in ranging
                'weekly': 'biweekly',
                'biweekly': 'monthly',
                'monthly': 'monthly'
            },
            'crisis': {
                'daily': 'daily',       # Maintain frequency in crisis
                'weekly': 'weekly',
                'biweekly': 'weekly',
                'monthly': 'weekly'
            }
        }
        
        # Track rebalancing history
        self.rebalance_history = []
    
    def get_rebalance_dates(
        self,
        start_date: datetime,
        end_date: datetime,
        market_regime: str = 'normal'
    ) -> List[datetime]:
        """
        Get scheduled rebalance dates with regime adjustments.
        
        Args:
            start_date: Start date
            end_date: End date
            market_regime: Current market regime
            
        Returns:
            List of rebalance dates
        """
        # Get adjusted frequency based on regime
        if self.use_dynamic_rebalancing and market_regime in self.regime_frequency_map:
            adjusted_freq = self.regime_frequency_map[market_regime].get(
                self.base_rebalance_frequency, self.base_rebalance_frequency
            )
        else:
            adjusted_freq = self.base_rebalance_frequency
        
        # Convert to pandas frequency
        freq = self.frequency_map.get(adjusted_freq, 'W')
        
        # Generate dates
        dates = pd.date_range(start_date, end_date, freq=freq)
        
        self.logger.info(
            f"Rebalance schedule: {adjusted_freq} "
            f"({len(dates)} dates for {market_regime} regime)"
        )
        
        return dates.tolist()
    
    def should_rebalance(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        market_regime: str = 'normal',
        days_since_last: int = 0,
        momentum_scores: Optional[pd.Series] = None
    ) -> Tuple[bool, str]:
        """
        Determine if rebalancing should occur based on dynamic triggers.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            market_regime: Current market regime
            days_since_last: Days since last rebalance
            momentum_scores: Optional momentum scores for each asset
            
        Returns:
            Tuple of (should_rebalance, reason)
        """
        # Get regime-specific threshold
        deviation_threshold = self.deviation_thresholds.get(market_regime, 0.10)
        
        # Calculate deviations
        weight_deviations = abs(current_weights - target_weights)
        max_deviation = weight_deviations.max()
        total_deviation = weight_deviations.sum()
        
        # Reason tracking
        reasons = []
        
        # 1. Check deviation trigger
        if max_deviation > deviation_threshold:
            reasons.append(f"Max deviation {max_deviation:.2%} > {deviation_threshold:.2%}")
        
        # 2. Check total portfolio drift
        if total_deviation > deviation_threshold * 2:
            reasons.append(f"Total drift {total_deviation:.2%} > {deviation_threshold * 2:.2%}")
        
        # 3. Time-based trigger (regime-dependent)
        time_triggers = {
            'trending': 3,    # 3 days max in trending
            'volatile': 5,    # 5 days in volatile
            'ranging': 14,    # 14 days in ranging
            'crisis': 2       # 2 days in crisis
        }
        
        max_days = time_triggers.get(market_regime, 7)
        if days_since_last > max_days:
            reasons.append(f"Time trigger: {days_since_last} days > {max_days} days")
        
        # 4. Momentum preservation check
        if self.momentum_preservation and momentum_scores is not None:
            # Check if high momentum positions are being reduced
            for asset in current_weights.index:
                if asset in momentum_scores.index:
                    current_weight = current_weights[asset]
                    target_weight = target_weights[asset]
                    momentum = momentum_scores[asset]
                    
                    # If reducing a high-momentum position significantly
                    if momentum > 0.5 and (current_weight - target_weight) > 0.05:
                        # Delay rebalancing for momentum preservation
                        self.logger.info(
                            f"Momentum preservation: delaying reduction of {asset} "
                            f"(momentum={momentum:.2f})"
                        )
                        return False, "Momentum preservation override"
        
        # 5. Crisis mode check
        if market_regime == 'crisis' and max_deviation > deviation_threshold * 0.5:
            # Lower threshold in crisis
            reasons.append(f"Crisis mode: deviation {max_deviation:.2%} > {deviation_threshold * 0.5:.2%}")
        
        should_rebalance = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else "No trigger met"
        
        return should_rebalance, reason
    
    def calculate_trades(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        current_values: pd.Series,
        market_regime: str = 'normal',
        momentum_scores: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Calculate required trades with regime-aware adjustments.
        
        Args:
            current_weights: Current weights
            target_weights: Target weights
            current_values: Current position values
            market_regime: Current market regime
            momentum_scores: Optional momentum scores
            
        Returns:
            DataFrame with trade details
        """
        total_value = current_values.sum()
        
        # Calculate base trades
        target_values = target_weights * total_value
        trade_values = target_values - current_values
        
        # Apply minimum trade size filter
        min_trade_value = total_value * self.min_trade_size
        trade_values[abs(trade_values) < min_trade_value] = 0
        
        # Apply momentum preservation adjustments
        if self.momentum_preservation and momentum_scores is not None:
            trade_values = self._apply_momentum_adjustments(
                trade_values, current_values, momentum_scores, market_regime
            )
        
        # Check turnover constraints
        turnover = abs(trade_values).sum() / total_value
        max_turnover = self.max_turnover.get(market_regime, 0.20)
        
        if turnover > max_turnover:
            # Scale down trades to meet turnover constraint
            scale_factor = max_turnover / turnover
            trade_values *= scale_factor
            
            self.logger.info(
                f"Scaled trades by {scale_factor:.2%} to meet "
                f"{market_regime} turnover limit of {max_turnover:.2%}"
            )
        
        # Create trade DataFrame
        trades = pd.DataFrame({
            'symbol': trade_values.index,
            'current_weight': current_weights,
            'target_weight': target_weights,
            'current_value': current_values,
            'target_value': target_values,
            'trade_value': trade_values,
            'trade_pct': trade_values / total_value
        })
        
        # Remove zero trades
        trades = trades[trades['trade_value'] != 0]
        
        # Sort by absolute trade value (largest first)
        trades = trades.sort_values('trade_value', key=abs, ascending=False)
        
        # Log rebalancing summary
        self._log_rebalance_summary(trades, turnover, market_regime)
        
        return trades
    
    def _apply_momentum_adjustments(
        self,
        trade_values: pd.Series,
        current_values: pd.Series,
        momentum_scores: pd.Series,
        market_regime: str
    ) -> pd.Series:
        """Apply momentum-based adjustments to preserve winning positions."""
        adjusted_trades = trade_values.copy()
        
        # Momentum preservation is stronger in trending markets
        preservation_strength = {
            'trending': 0.7,   # Preserve 70% of momentum positions
            'volatile': 0.3,   # Less preservation in volatile
            'ranging': 0.2,    # Minimal in ranging
            'crisis': 0.5      # Moderate in crisis
        }
        
        strength = preservation_strength.get(market_regime, 0.4)
        
        for asset in trade_values.index:
            if asset in momentum_scores.index:
                momentum = momentum_scores[asset]
                trade = trade_values[asset]
                
                # If reducing a high-momentum position
                if momentum > 0.3 and trade < 0:
                    # Reduce the sell trade
                    adjusted_trades[asset] = trade * (1 - strength * min(momentum, 1.0))
                    
                # If adding to a negative-momentum position
                elif momentum < -0.3 and trade > 0:
                    # Reduce the buy trade
                    adjusted_trades[asset] = trade * (1 + strength * min(abs(momentum), 1.0))
        
        return adjusted_trades
    
    def optimize_trade_execution(
        self,
        trades: pd.DataFrame,
        market_conditions: Dict[str, pd.Series],
        execution_style: str = 'normal'
    ) -> List[Dict]:
        """
        Optimize trade execution order and timing.
        
        Args:
            trades: DataFrame with trade details
            market_conditions: Dict with volatility, volume, spread data
            execution_style: 'aggressive', 'normal', or 'passive'
            
        Returns:
            Ordered list of trades with execution parameters
        """
        execution_plan = []
        
        # Calculate urgency scores for each trade
        for idx, trade in trades.iterrows():
            symbol = trade['symbol']
            
            # Base urgency on trade size
            urgency_score = abs(trade['trade_pct'])
            
            # Adjust for market conditions
            if symbol in market_conditions.get('volatility', {}):
                vol = market_conditions['volatility'][symbol]
                # Higher urgency for high volatility assets
                urgency_score *= (1 + vol)
            
            if symbol in market_conditions.get('spread', {}):
                spread = market_conditions['spread'][symbol]
                # Lower urgency for wide spreads (wait for better prices)
                urgency_score *= (1 - spread * 10)  # Assuming spread in decimal
            
            # Determine execution parameters
            if execution_style == 'aggressive' or urgency_score > 0.05:
                exec_params = {
                    'type': 'market',
                    'urgency': 'high',
                    'time_limit': 5,  # 5 minutes
                    'accept_slippage': 0.002  # 20 bps
                }
            elif execution_style == 'passive' or urgency_score < 0.01:
                exec_params = {
                    'type': 'limit',
                    'urgency': 'low',
                    'time_limit': 60,  # 1 hour
                    'accept_slippage': 0.0005  # 5 bps
                }
            else:
                exec_params = {
                    'type': 'adaptive',
                    'urgency': 'normal',
                    'time_limit': 30,  # 30 minutes
                    'accept_slippage': 0.001  # 10 bps
                }
            
            execution = {
                'symbol': symbol,
                'side': 'buy' if trade['trade_value'] > 0 else 'sell',
                'value': abs(trade['trade_value']),
                'urgency_score': urgency_score,
                'execution_params': exec_params
            }
            
            execution_plan.append(execution)
        
        # Sort by urgency score (highest first)
        execution_plan.sort(key=lambda x: x['urgency_score'], reverse=True)
        
        return execution_plan
    
    def _log_rebalance_summary(
        self,
        trades: pd.DataFrame,
        turnover: float,
        market_regime: str
    ):
        """Log rebalancing summary for monitoring."""
        if len(trades) == 0:
            self.logger.info("No rebalancing trades required")
            return
        
        # Calculate metrics
        total_trade_value = trades['trade_value'].abs().sum()
        num_buys = (trades['trade_value'] > 0).sum()
        num_sells = (trades['trade_value'] < 0).sum()
        
        # Store in history
        self.rebalance_history.append({
            'timestamp': datetime.now(),
            'regime': market_regime,
            'turnover': turnover,
            'num_trades': len(trades),
            'trade_value': total_trade_value
        })
        
        self.logger.info(
            f"Rebalancing - Regime: {market_regime}, "
            f"Trades: {len(trades)} (B:{num_buys}/S:{num_sells}), "
            f"Turnover: {turnover:.2%}, "
            f"Value: ${total_trade_value:,.0f}"
        )
    
    def get_rebalance_metrics(self, lookback_days: int = 30) -> Dict:
        """Get rebalancing metrics for performance analysis."""
        if not self.rebalance_history:
            return {}
        
        # Filter to recent history
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent_history = [
            h for h in self.rebalance_history
            if h['timestamp'] > cutoff
        ]
        
        if not recent_history:
            return {}
        
        # Calculate metrics
        turnovers = [h['turnover'] for h in recent_history]
        trade_counts = [h['num_trades'] for h in recent_history]
        
        # Group by regime
        regime_stats = {}
        for regime in ['trending', 'volatile', 'ranging', 'crisis']:
            regime_data = [h for h in recent_history if h['regime'] == regime]
            if regime_data:
                regime_stats[regime] = {
                    'count': len(regime_data),
                    'avg_turnover': np.mean([h['turnover'] for h in regime_data]),
                    'avg_trades': np.mean([h['num_trades'] for h in regime_data])
                }
        
        return {
            'total_rebalances': len(recent_history),
            'avg_turnover': np.mean(turnovers),
            'max_turnover': np.max(turnovers),
            'avg_trades_per_rebalance': np.mean(trade_counts),
            'regime_breakdown': regime_stats
        }