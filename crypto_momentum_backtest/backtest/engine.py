# crypto_momentum_backtest/backtest/engine.py
"""Main backtesting engine using VectorBT."""
import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
from pathlib import Path

from ..data.json_storage import JsonStorage
from ..data.universe_manager import UniverseManager
from ..signals.signal_generator import SignalGenerator
from ..portfolio.erc_optimizer import ERCOptimizer
from ..portfolio.position_sizer import PositionSizer
from ..portfolio.rebalancer import Rebalancer
from ..risk.risk_manager import RiskManager
from ..costs.cost_model import CostModel
from ..costs.funding_rates import FundingRates
from ..utils.config import Config


class BacktestEngine:
    """
    Main backtesting engine orchestrating all components.
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
        
        # Initialize components
        self.storage = JsonStorage(data_dir)
        self.universe_manager = UniverseManager(
            data_dir,
            universe_size=config.strategy.universe_size,
            logger=self.logger
        )
        
        self.signal_generator = SignalGenerator(
            adx_period=config.signals.adx_period,
            adx_threshold=config.signals.adx_threshold,
            ewma_fast=config.signals.ewma_fast,
            ewma_slow=config.signals.ewma_slow,
            volume_filter_multiple=config.signals.volume_filter_multiple,
            logger=self.logger
        )
        
        self.optimizer = ERCOptimizer(
            max_position_size=config.strategy.max_position_size,
            allow_short=config.strategy.long_short,
            logger=self.logger
        )
        
        self.position_sizer = PositionSizer(
            atr_period=config.risk.atr_period,
            atr_multiplier=config.risk.atr_multiplier,
            max_position_size=config.strategy.max_position_size,
            logger=self.logger
        )
        
        self.rebalancer = Rebalancer(
            rebalance_frequency=config.strategy.rebalance_frequency,
            logger=self.logger
        )
        
        self.risk_manager = RiskManager(
            max_drawdown=config.risk.max_drawdown_threshold,
            max_correlation=config.risk.max_correlation,
            max_exchange_exposure=config.risk.max_exchange_exposure,
            volatility_scale_threshold=config.risk.volatility_regime_multiplier,
            logger=self.logger
        )
        
        self.cost_model = CostModel(
            maker_fee=config.costs.maker_fee,
            taker_fee=config.costs.taker_fee,
            base_spread=config.costs.base_spread,
            logger=self.logger
        )
        
        self.funding_rates = FundingRates(
            lookback_days=config.costs.funding_lookback_days,
            logger=self.logger
        )
        
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
        Generate trading signals for all symbols.
        
        Args:
            data: Market data by symbol
            
        Returns:
            Signals by symbol
        """
        signals = {}
        
        for symbol, df in data.items():
            self.logger.info(f"Generating signals for {symbol}")
            
            signal_df = self.signal_generator.generate_signals(
                df,
                apply_filters=True,
                calculate_strength=True
            )
            
            signals[symbol] = signal_df
        
        return signals
    
    def run_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 1000000
    ) -> Dict:
        """
        Run complete backtest.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Initial capital
            
        Returns:
            Backtest results
        """
        self.logger.info(
            f"Starting backtest from {start_date} to {end_date} "
            f"with ${initial_capital:,.0f}"
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
                    target_weights = self.optimizer.optimize(
                        lookback_returns,
                        pd.DataFrame(current_signals).T,
                        constraints=None
                    )
                else:
                    # Equal weight for initial period
                    active_signals = current_signals[current_signals != 0]
                    if len(active_signals) > 0:
                        target_weights = pd.Series(
                            1.0 / len(active_signals),
                            index=active_signals.index
                        )
                    else:
                        target_weights = pd.Series(0.0, index=close_prices.columns)
                
                # Apply risk limits
                target_weights *= self.risk_manager.risk_state['exposure_level']
                
                # Calculate position sizes
                atr_data = pd.DataFrame({
                    symbol: self.signal_generator.indicators.atr(market_data[symbol])
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
                            'value': abs(position_changes[symbol] * close_prices.loc[date, symbol])
                        })
            else:
                # Carry forward positions
                positions.iloc[i] = positions.iloc[i-1]
            
            # Calculate portfolio value
            position_values = positions.iloc[i] * close_prices.iloc[i]
            cash = portfolio_value - position_values.sum()
            
            # Apply costs if there were trades
            if i > 0 and not positions.iloc[i].equals(positions.iloc[i-1]):
                trade_costs = self._calculate_trade_costs(
                    positions.iloc[i-1],
                    positions.iloc[i],
                    close_prices.iloc[i],
                    market_data
                )
                cash -= trade_costs['total']
            
            portfolio_value = position_values.sum() + cash
            portfolio_values.append(portfolio_value)
        
        # Create results
        portfolio_series = pd.Series(
            portfolio_values,
            index=common_index,
            name='portfolio_value'
        )
        
        # Calculate metrics
        returns_series = portfolio_series.pct_change().dropna()
        
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
            )
        }
        
        return results
    
    def _calculate_trade_costs(
        self,
        prev_positions: pd.Series,
        curr_positions: pd.Series,
        prices: pd.Series,
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Calculate trading costs."""
        position_changes = curr_positions - prev_positions
        
        total_costs = {
            'exchange_fees': 0.0,
            'spread_costs': 0.0,
            'slippage': 0.0,
            'total': 0.0
        }
        
        for symbol in position_changes[position_changes != 0].index:
            trade_value = abs(position_changes[symbol] * prices[symbol])
            
            # Simple cost calculation
            costs = self.cost_model.calculate_trading_costs(
                trade_value=trade_value,
                trade_type='taker',
                volatility=0.15,  # Default volatility
                volume=1e8  # Default volume
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
        """Calculate performance metrics."""
        # Basic metrics
        total_return = (portfolio_values.iloc[-1] - portfolio_values.iloc[0]) / portfolio_values.iloc[0]
        
        # Risk metrics
        sharpe_ratio = self.risk_manager.calculate_sharpe_ratio(returns)
        sortino_ratio = self.risk_manager.calculate_sortino_ratio(returns)
        max_drawdown = self.risk_manager.calculate_max_drawdown(returns)
        
        # Additional metrics
        volatility = returns.std() * np.sqrt(252)
        win_rate = (returns > 0).sum() / len(returns)
        
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
            'calmar_ratio': (total_return / abs(max_drawdown)) if max_drawdown != 0 else 0
        }
