# crypto_momentum_backtest/backtest/engine.py
"""Main backtesting engine using VectorBT with full long/short support."""
import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
from pathlib import Path

from ..data.json_storage import JsonStorage
from ..data import UniverseManager
from ..signals import SignalGenerator
from ..portfolio import ERCOptimizer, ARPOptimizer
from ..portfolio.position_sizer import PositionSizer
from ..portfolio import Rebalancer
from ..risk import RiskManager
from ..costs.cost_model import CostModel
from ..costs.funding_rates import FundingRates
from ..utils.config import Config


class BacktestEngine:
    """
    Main backtesting engine orchestrating all components with long/short support.
    """
    
    def __init__(
        self,
        config: Config,
        data_dir: Path,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize backtest engine.
        
        Args:
            config: Configuration object
            data_dir: Data directory path
            logger: Logger instance
        """
        
        self.config = config
        self.data_dir = Path(data_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components with safe config access
        try:
            # Get configuration values with defaults
            universe_size = getattr(config.data, 'universe_size', 10) if hasattr(config, 'data') else 10
            
            # Signal parameters
            adx_period = getattr(config.signals, 'adx_period', 14) if hasattr(config, 'signals') else 14
            adx_threshold = getattr(config.signals, 'adx_threshold', 25.0) if hasattr(config, 'signals') else 25.0
            ewma_fast = getattr(config.signals, 'ewma_fast', 20) if hasattr(config, 'signals') else 20
            ewma_slow = getattr(config.signals, 'ewma_slow', 50) if hasattr(config, 'signals') else 50
            volume_filter = getattr(config.signals, 'volume_filter_multiple', 1.5) if hasattr(config, 'signals') else 1.5
            
            # Portfolio parameters
            max_position_size = getattr(config.portfolio, 'max_position_size', 0.10) if hasattr(config, 'portfolio') else 0.10
            rebalance_freq = getattr(config.portfolio, 'rebalance_frequency', 'weekly') if hasattr(config, 'portfolio') else 'weekly'
            
            # Risk parameters
            max_drawdown = getattr(config.risk, 'max_drawdown', 0.25) if hasattr(config, 'risk') else 0.25
            correlation_threshold = getattr(config.risk, 'correlation_threshold', 0.80) if hasattr(config, 'risk') else 0.80
            volatility_multiple = getattr(config.risk, 'volatility_multiple', 2.0) if hasattr(config, 'risk') else 2.0
            
            # Cost parameters
            maker_fee = getattr(config.costs, 'maker_fee', 0.001) if hasattr(config, 'costs') else 0.001
            taker_fee = getattr(config.costs, 'taker_fee', 0.001) if hasattr(config, 'costs') else 0.001
            base_spread = getattr(config.costs, 'base_spread', 0.0005) if hasattr(config, 'costs') else 0.0005
            funding_days = getattr(config.costs, 'funding_lookback_days', 30) if hasattr(config, 'costs') else 30
            borrow_rate = getattr(config.costs, 'borrow_rate', 0.0002) if hasattr(config, 'costs') else 0.0002
            
            # Strategy parameters - CRITICAL FOR LONG/SHORT
            long_short = getattr(config.strategy, 'long_short', True) if hasattr(config, 'strategy') else True
            
            # Override with explicit long/short if in config
            if hasattr(config, 'portfolio') and hasattr(config.portfolio, 'allow_short'):
                long_short = config.portfolio.allow_short
            
            self.long_short = long_short
            self.logger.info(f"Long/Short trading: {'ENABLED' if long_short else 'DISABLED'}")
            
        except AttributeError as e:
            self.logger.warning(f"Config attribute error: {e}, using defaults")
            # Set all defaults
            universe_size = 10
            adx_period = 14
            adx_threshold = 25.0
            ewma_fast = 20
            ewma_slow = 50
            volume_filter = 1.5
            max_position_size = 0.10
            rebalance_freq = 'weekly'
            max_drawdown = 0.25
            correlation_threshold = 0.80
            volatility_multiple = 2.0
            maker_fee = 0.001
            taker_fee = 0.001
            base_spread = 0.0005
            funding_days = 30
            borrow_rate = 0.0002
            long_short = True
            self.long_short = long_short
        
        # Initialize components
        self.storage = JsonStorage(data_dir)
        self.universe_manager = UniverseManager(
            data_dir,
            universe_size=universe_size,
            logger=self.logger
        )
        
        # Get signal configuration with defaults
        momentum_threshold = getattr(config.signals, 'momentum_threshold', 0.01) if hasattr(config, 'signals') else 0.01
        mean_ewm_threshold = getattr(config.signals, 'mean_return_ewm_threshold', 0.015) if hasattr(config, 'signals') else 0.015
        mean_simple_threshold = getattr(config.signals, 'mean_return_simple_threshold', 0.015) if hasattr(config, 'signals') else 0.015
        min_score_threshold = getattr(config.signals, 'min_score_threshold', 0.2) if hasattr(config, 'signals') else 0.2
        
        self.signal_generator = SignalGenerator(
            momentum_threshold=momentum_threshold,
            momentum_ewma_span=ewma_fast,
            mean_return_ewm_threshold=mean_ewm_threshold,
            mean_return_simple_threshold=mean_simple_threshold,
            adx_period=adx_period,
            adx_threshold=adx_threshold,
            use_volume_confirmation=True,
            volume_threshold=volume_filter,
            min_score_threshold=min_score_threshold,  # Important for long/short
            logger=self.logger
        )
        
        # Check if using ARP optimizer
        optimization_method = getattr(config.portfolio, 'optimization_method', 'enhanced_risk_parity') if hasattr(config, 'portfolio') else 'enhanced_risk_parity'
        
        if optimization_method == 'agnostic_risk_parity':
            # Get ARP-specific parameters
            target_volatility = getattr(config.portfolio, 'target_volatility', None) if hasattr(config, 'portfolio') else None
            arp_shrinkage_factor = getattr(config.portfolio, 'arp_shrinkage_factor', None) if hasattr(config, 'portfolio') else None
            
            self.optimizer = ARPOptimizer(
                target_volatility=target_volatility,
                budget_constraint='full_investment' if long_short else 'long_only',
                logger=self.logger
            )
            self.logger.info("Using Agnostic Risk Parity (ARP) optimizer")
            
            # Store ARP-specific config
            self.arp_shrinkage_factor = arp_shrinkage_factor
        else:
            # Existing ERC optimizer with long/short support
            self.optimizer = ERCOptimizer(
                max_position_size=max_position_size,
                allow_short=long_short,  # Pass long/short setting
                logger=self.logger
            )
            self.logger.info(f"Using Enhanced Risk Contribution (ERC) optimizer (allow_short={long_short})")
        
        self.position_sizer = PositionSizer(
            atr_period=adx_period,
            atr_multiplier=2.0,
            max_position_size=max_position_size,
            logger=self.logger
        )
        
        self.rebalancer = Rebalancer(
            rebalance_frequency=rebalance_freq,
            logger=self.logger
        )
        
        self.risk_manager = RiskManager(
            max_drawdown=max_drawdown,
            max_position_size=max_position_size,
            max_correlation=correlation_threshold,
            max_exchange_exposure=0.40,
            volatility_scale_threshold=volatility_multiple,
            logger=self.logger
        )
        
        self.cost_model = CostModel(
            maker_fee=maker_fee,
            taker_fee=taker_fee,
            base_spread=base_spread,
            logger=self.logger
        )
        
        self.funding_rates = FundingRates(
            lookback_days=funding_days,
            logger=self.logger
        )
        
        # Store borrow rate for short positions
        self.borrow_rate = borrow_rate

    def load_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        Load market data for symbols.
        
        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        data = {}
        
        for symbol in symbols:
            self.logger.info(f"Loading data for {symbol}")
            
            df = self.storage.load_range(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                columns=['open', 'high', 'low', 'close', 'volume']
            )
            
            if not df.empty:
                data[symbol] = df
            else:
                self.logger.warning(f"No data found for {symbol}")
        
        return data
    
    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate trading signals for all symbols with long/short support.
        
        Args:
            data: Market data by symbol
            
        Returns:
            Signals by symbol
        """
        signals = {}
        total_long_signals = 0
        total_short_signals = 0
        
        for symbol, df in data.items():
            self.logger.info(f"Generating signals for {symbol}")
            
            # Generate signals using the configured strategy
            signal_strategy = self.config.signals.signal_strategy
            if hasattr(signal_strategy, 'value'):
                signal_strategy = signal_strategy.value
            
            # Use momentum_score for long/short as it produces continuous scores
            if self.long_short and signal_strategy != 'momentum_score':
                self.logger.info(f"Switching to momentum_score for long/short trading")
                signal_strategy = 'momentum_score'
            
            signal_series = self.signal_generator.generate_signals(
                df,
                signal_type=signal_strategy,
                symbol=symbol
            )
            
            # For long/short, ensure we have both positive and negative signals
            if self.long_short:
                # Check signal distribution
                positive_signals = (signal_series > 0).sum()
                negative_signals = (signal_series < 0).sum()
                
                # If too imbalanced, adjust
                if negative_signals == 0 or positive_signals / (negative_signals + 1) > 5:
                    self.logger.warning(f"Signal imbalance for {symbol}: {positive_signals} long, {negative_signals} short")
                    # Center the signals
                    non_zero = signal_series[signal_series != 0]
                    if len(non_zero) > 0:
                        signal_series = signal_series - non_zero.mean() * 0.5
            
            # Convert to DataFrame format expected by engine
            signal_df = pd.DataFrame(index=df.index)
            signal_df['signal_strength'] = signal_series
            
            # For continuous signals (momentum_score), create discrete positions
            if signal_strategy == 'momentum_score':
                signal_df['position'] = 0
                # Use configurable thresholds
                long_threshold = getattr(self.signal_generator, 'min_score_threshold', 0.2)
                short_threshold = -long_threshold
                
                signal_df.loc[signal_series > long_threshold, 'position'] = 1
                signal_df.loc[signal_series < short_threshold, 'position'] = -1
                
                signal_df['long_signal'] = signal_df['position'] == 1
                signal_df['short_signal'] = signal_df['position'] == -1
            else:
                # For discrete signals
                signal_df['position'] = signal_series
                signal_df['long_signal'] = signal_series == 1
                signal_df['short_signal'] = signal_series == -1
            
            # Count signals
            long_count = signal_df['long_signal'].sum()
            short_count = signal_df['short_signal'].sum()
            total_long_signals += long_count
            total_short_signals += short_count
            
            self.logger.info(f"  {symbol}: {long_count} long, {short_count} short signals")
            
            signals[symbol] = signal_df
        
        self.logger.info(f"Total signals - Long: {total_long_signals}, Short: {total_short_signals}")
        
        # If no short signals generated and long/short is enabled, warn
        if self.long_short and total_short_signals == 0:
            self.logger.warning("No short signals generated! Check signal parameters.")
        
        return signals
    
    def run_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 1000000
    ) -> Dict:
        """
        Run complete backtest with long/short support.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Initial capital
            
        Returns:
            Backtest results
        """
        self.logger.info(
            f"Starting {'LONG/SHORT' if self.long_short else 'LONG-ONLY'} backtest "
            f"from {start_date} to {end_date} with ${initial_capital:,.0f}"
        )
        
        # Get universe for the period
        universe = self.universe_manager.get_universe(start_date)
        
        # Load data
        market_data = self.load_data(universe, start_date, end_date)
        
        if not market_data:
            raise ValueError("No market data loaded")
        
        # Generate signals
        signals = self.generate_signals(market_data)
        
        # Prepare data for VectorBT
        close_prices = pd.DataFrame({
            symbol: data['close']
            for symbol, data in market_data.items()
        })
        
        signal_matrix = pd.DataFrame({
            symbol: sig_df['position']
            for symbol, sig_df in signals.items()
        })
        
        # Align indices
        common_index = close_prices.index.intersection(signal_matrix.index)
        close_prices = close_prices.loc[common_index]
        signal_matrix = signal_matrix.loc[common_index]
        
        # Calculate returns
        returns = close_prices.pct_change()
        
        # Get rebalance dates
        rebalance_dates = self.rebalancer.get_rebalance_dates(
            start_date, end_date
        )
        
        # Initialize portfolio tracking
        portfolio_value = initial_capital
        positions = pd.DataFrame(0.0, index=common_index, columns=close_prices.columns)
        portfolio_values = []
        trades_log = []
        daily_pnl = []
        
        # Track borrowing costs for shorts
        cumulative_borrow_cost = 0
        
        # Simulate through time
        for i, date in enumerate(common_index):
            # Update risk state
            if i > 0:
                portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
                self.risk_manager.update_risk_state(
                    portfolio_value,
                    portfolio_returns,
                    positions.iloc[i-1]
                )
            
            # Check if rebalancing needed
            if date in rebalance_dates or i == 0:
                # Get target weights from optimizer
                lookback_returns = returns.iloc[max(0, i-60):i]
                current_signals = signal_matrix.iloc[i]
                
                if len(lookback_returns) > 20:
                    # Check if using ARP optimizer
                    if isinstance(self.optimizer, ARPOptimizer):
                        # ARP needs covariance matrix and signals
                        cov_matrix = lookback_returns.cov()
                        
                        # Get shrinkage factor if configured
                        shrinkage = getattr(self, 'arp_shrinkage_factor', None)
                        
                        # Optimize with ARP
                        target_weights = self.optimizer.optimize(
                            covariance_matrix=cov_matrix,
                            signals=current_signals,
                            shrinkage_factor=shrinkage
                        )
                        
                        # Ensure target_weights is a Series with correct index
                        if isinstance(target_weights, np.ndarray):
                            target_weights = pd.Series(target_weights, index=current_signals.index)
                    else:
                        # Existing ERC optimization
                        target_weights = self.optimizer.optimize(
                            lookback_returns,
                            pd.DataFrame(current_signals).T,
                            constraints=None
                        )
                else:
                    # Equal weight for initial period
                    active_signals = current_signals[current_signals != 0]
                    if len(active_signals) > 0:
                        # For long/short, allocate based on signal direction
                        if self.long_short:
                            target_weights = pd.Series(
                                active_signals / len(active_signals),
                                index=active_signals.index
                            )
                        else:
                            target_weights = pd.Series(
                                1.0 / len(active_signals),
                                index=active_signals.index
                            )
                    else:
                        target_weights = pd.Series(0.0, index=close_prices.columns)
                
                # Apply risk limits
                target_weights *= self.risk_manager.risk_state['exposure_level']
                
                # Calculate position sizes
                from ..signals.technical_indicators import TechnicalIndicators
                
                atr_data = pd.DataFrame({
                    symbol: TechnicalIndicators.atr(market_data[symbol])
                    for symbol in market_data.keys()
                })
                
                # Ensure we have the right data format
                if i < len(atr_data):
                    atr_series = atr_data.iloc[i]
                else:
                    atr_series = atr_data.iloc[-1]
                
                position_sizes = self.position_sizer.calculate_position_sizes(
                    prices=close_prices.iloc[i],
                    atr=atr_series,
                    signals=current_signals,
                    capital=portfolio_value
                )
                
                # For long/short, respect signal direction
                if self.long_short:
                    # Ensure position signs match signal signs
                    for symbol in position_sizes.index:
                        if current_signals[symbol] < 0 and position_sizes[symbol] > 0:
                            position_sizes[symbol] = -position_sizes[symbol]
                
                # Update positions
                positions.iloc[i] = position_sizes
                
                # Log trades
                if i > 0:
                    position_changes = positions.iloc[i] - positions.iloc[i-1]
                    for symbol in position_changes[position_changes != 0].index:
                        trades_log.append({
                            'date': date,
                            'symbol': symbol,
                            'side': 'buy' if position_changes[symbol] > 0 else 'sell',
                            'units': abs(position_changes[symbol]),
                            'price': close_prices.loc[date, symbol],
                            'value': abs(position_changes[symbol] * close_prices.loc[date, symbol]),
                            'position_type': 'long' if positions.iloc[i][symbol] > 0 else 'short' if positions.iloc[i][symbol] < 0 else 'flat'
                        })
            else:
                # Carry forward positions
                positions.iloc[i] = positions.iloc[i-1]
            
            # Calculate portfolio value with long/short P&L
            position_values = positions.iloc[i] * close_prices.iloc[i]
            
            # Calculate P&L from price changes
            if i > 0:
                price_changes = close_prices.iloc[i] - close_prices.iloc[i-1]
                position_pnl = (positions.iloc[i-1] * price_changes).sum()
                daily_pnl.append(position_pnl)
                
                # Calculate borrowing costs for short positions
                short_positions = positions.iloc[i-1][positions.iloc[i-1] < 0]
                if len(short_positions) > 0:
                    short_values = abs(short_positions * close_prices.iloc[i-1])
                    daily_borrow_cost = short_values.sum() * self.borrow_rate
                    cumulative_borrow_cost += daily_borrow_cost
                    position_pnl -= daily_borrow_cost
            else:
                position_pnl = 0
            
            # Apply costs if there were trades
            trade_costs = 0
            if i > 0 and not positions.iloc[i].equals(positions.iloc[i-1]):
                trade_costs_dict = self._calculate_trade_costs(
                    positions.iloc[i-1],
                    positions.iloc[i],
                    close_prices.iloc[i],
                    market_data
                )
                trade_costs = trade_costs_dict['total']
            
            # Update portfolio value
            if i == 0:
                portfolio_value = initial_capital
            else:
                portfolio_value = portfolio_values[-1] + position_pnl - trade_costs
            
            portfolio_values.append(portfolio_value)
        
        # Create results
        portfolio_series = pd.Series(
            portfolio_values,
            index=common_index,
            name='portfolio_value'
        )
        
        # Calculate metrics
        returns_series = portfolio_series.pct_change().dropna()
        
        # Log final statistics
        total_trades = len(trades_log)
        if total_trades > 0:
            long_trades = sum(1 for t in trades_log if t.get('position_type') == 'long')
            short_trades = sum(1 for t in trades_log if t.get('position_type') == 'short')
            self.logger.info(f"Total trades: {total_trades} (Long: {long_trades}, Short: {short_trades})")
            self.logger.info(f"Cumulative borrowing costs: ${cumulative_borrow_cost:,.2f}")
        
        results = {
            'portfolio': portfolio_series,
            'returns': returns_series,
            'positions': positions,
            'trades': pd.DataFrame(trades_log),
            'metrics': self._calculate_metrics(portfolio_series, returns_series),
            'risk_report': self.risk_manager.generate_risk_report(
                portfolio_value,
                returns_series,
                positions
            ),
            'cumulative_borrow_cost': cumulative_borrow_cost
        }
        
        return results
    
    def _calculate_trade_costs(
        self,
        prev_positions: pd.Series,
        curr_positions: pd.Series,
        prices: pd.Series,
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Calculate trading costs including considerations for shorts."""
        position_changes = curr_positions - prev_positions
        
        total_costs = {
            'exchange_fees': 0.0,
            'spread_costs': 0.0,
            'slippage': 0.0,
            'total': 0.0
        }
        
        for symbol in position_changes[position_changes != 0].index:
            trade_value = abs(position_changes[symbol] * prices[symbol])
            
            # Higher costs for short trades (typically)
            is_short_trade = curr_positions[symbol] < 0 or prev_positions[symbol] < 0
            
            costs = self.cost_model.calculate_trading_costs(
                trade_value=trade_value,
                trade_type='taker',
                volatility=0.15 * (1.2 if is_short_trade else 1.0),  # Higher vol for shorts
                volume=1e8
            )
            
            total_costs['exchange_fees'] += costs['exchange_fee']
            total_costs['spread_costs'] += costs['spread_cost']
            total_costs['slippage'] += costs['slippage']
        
        total_costs['total'] = sum([
            total_costs['exchange_fees'],
            total_costs['spread_costs'],
            total_costs['slippage']
        ])
        
        return total_costs
    
    def _calculate_metrics(
        self,
        portfolio_values: pd.Series,
        returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate performance metrics including long/short specific metrics."""
        # Basic metrics
        total_return = (portfolio_values.iloc[-1] - portfolio_values.iloc[0]) / portfolio_values.iloc[0]
        
        # Risk metrics
        sharpe_ratio = self.risk_manager.calculate_sharpe_ratio(returns)
        sortino_ratio = self.risk_manager.calculate_sortino_ratio(returns)
        max_drawdown = self.risk_manager.calculate_max_drawdown(returns)
        
        # Additional metrics
        volatility = returns.std() * np.sqrt(252)
        win_rate = (returns > 0).sum() / len(returns)
        
        # Long/short specific metrics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        
        profit_factor = abs(positive_returns.sum() / negative_returns.sum()) if len(negative_returns) > 0 and negative_returns.sum() != 0 else np.inf
        
        return {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 / len(returns)) - 1,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
            'best_day': returns.max(),
            'worst_day': returns.min(),
            'calmar_ratio': (total_return / abs(max_drawdown)) if max_drawdown != 0 else 0,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'long_short_enabled': self.long_short
        }