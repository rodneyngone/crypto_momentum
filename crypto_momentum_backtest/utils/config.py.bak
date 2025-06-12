#!/usr/bin/env python3
"""
Configuration management for crypto momentum backtesting framework.

This module provides comprehensive configuration management with crypto-optimized
defaults and validation for all system components.
"""

import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, date
from enum import Enum

logger = logging.getLogger(__name__)


class RebalanceFrequency(Enum):
    """Enumeration of supported rebalancing frequencies."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class OptimizationMethod(Enum):
    """Enumeration of portfolio optimization methods."""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    ENHANCED_RISK_PARITY = "enhanced_risk_parity"
    MEAN_VARIANCE = "mean_variance"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"


class SignalStrategy(Enum):
    """Enumeration of signal generation strategies."""
    MOMENTUM = "momentum"
    MOMENTUM_SCORE = "momentum_score"
    DUAL_MOMENTUM = "dual_momentum"
    MULTI_TIMEFRAME = "multi_timeframe"
    MEAN_RETURN_EWM = "mean_return_ewm"
    MEAN_RETURN_SIMPLE = "mean_return_simple"
    ENSEMBLE = "ensemble"
    HYBRID_AND = "hybrid_and"
    HYBRID_OR = "hybrid_or"


@dataclass
class DataConfig:
    """Configuration for data management and universe selection."""
    
    # Date range
    start_date: Union[str, date] = "2022-01-01"
    end_date: Union[str, date] = "2023-12-31"
    
    # Universe selection
    universe_size: int = 10
    selection_size: int = 15  # Number to select from
    market_cap_threshold: float = 100_000_000  # $100M minimum market cap
    exclude_stablecoins: bool = True
    exclude_wrapped: bool = True  # Exclude wrapped tokens
    survivorship_bias: bool = True  # Handle delisted assets
    
    # Data sources and caching
    data_source: str = "binance"
    cache_directory: str = "data"
    cache_days: int = 30
    update_frequency: str = "daily"
    
    # Data quality
    min_trading_days: int = 100
    max_missing_data_pct: float = 0.05  # 5% max missing data
    
    def __post_init__(self):
        """Validate data configuration parameters."""
        if self.universe_size < 1 or self.universe_size > 50:
            raise ValueError(f"Universe size {self.universe_size} must be between 1 and 50")
        
        if self.market_cap_threshold < 1_000_000:  # $1M minimum
            logger.warning(f"Very low market cap threshold: ${self.market_cap_threshold:,.0f}")
        
        # Convert string dates to datetime if needed
        if isinstance(self.start_date, str):
            self.start_date = datetime.strptime(self.start_date, "%Y-%m-%d").date()
        if isinstance(self.end_date, str):
            self.end_date = datetime.strptime(self.end_date, "%Y-%m-%d").date()
        
        # Validate date range
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        
        logger.debug(f"DataConfig validated: {self.universe_size} assets, "
                    f"{self.start_date} to {self.end_date}")


@dataclass
class SignalConfig:
    """Configuration for signal generation with crypto-optimized defaults."""
    
    # Strategy selection
    signal_strategy: SignalStrategy = SignalStrategy.MOMENTUM
    
    # Momentum strategy parameters (working baseline)
    momentum_threshold: float = 0.01  # 1% threshold for crypto momentum
    momentum_ewma_span: int = 20
    
    # Legacy EWMA parameters for backward compatibility
    ewma_fast: int = 20
    ewma_slow: int = 50
    
    # Mean return EWM strategy parameters (CRYPTO-OPTIMIZED)
    mean_return_ewm_threshold: float = 0.015  # 1.5% - calibrated for crypto
    mean_return_ewm_span: int = 10            # More responsive for crypto volatility
    
    # Mean return simple strategy parameters (CRYPTO-OPTIMIZED)  
    mean_return_simple_threshold: float = 0.015  # 1.5% - calibrated for crypto
    mean_return_simple_window: int = 10          # Balanced for crypto noise/signal
    
    # Technical analysis parameters
    adx_period: int = 14
    adx_threshold: float = 20.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    
    # Enhanced signal parameters
    adx_periods: List[int] = field(default_factory=lambda: [7, 14, 21])
    base_ewma_span: int = 20
    adaptive_ewma: bool = True
    momentum_weights: Dict[str, float] = field(default_factory=lambda: {
        'price': 0.4,
        'volume': 0.2,
        'rsi': 0.2,
        'macd': 0.2
    })
    absolute_momentum_threshold: float = 0.05
    relative_momentum_threshold: float = 0.0
    momentum_lookback: int = 30
    
    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Score thresholds
    min_score_threshold: float = 0.3
    max_correlation: float = 0.7
    
    # Legacy compatibility parameters
    threshold: float = 0.01           # Legacy alias for momentum_threshold
    lookback: int = 20                # Legacy parameter
    
    # Volume confirmation and filtering
    use_volume_confirmation: bool = True
    volume_threshold: float = 1.2     # 20% above average volume
    volume_lookback: int = 20
    volume_filter_multiple: float = 1.5  # Legacy volume filter parameter
    
    # Signal filtering and risk controls
    min_signal_gap: int = 1         # Minimum periods between signals
    max_signals_per_period: int = 100
    signal_decay_periods: int = 5   # Periods before signal strength decays
    
    # Advanced features
    use_regime_detection: bool = False
    volatility_adjustment: bool = True
    correlation_filter: bool = False
    correlation_threshold: float = 0.8
    
    # Additional legacy parameters that might be needed
    signal_type: str = "momentum"     # Legacy alias for signal_strategy
    combine_signals: bool = False     # Legacy parameter (deprecated)
    
    def __post_init__(self):
        """Validate signal configuration with crypto-specific checks."""
        # Ensure legacy compatibility
        if self.threshold != self.momentum_threshold:
            self.momentum_threshold = self.threshold  # Use legacy value if different
        
        # Sync signal_type with signal_strategy for legacy compatibility
        if hasattr(self, 'signal_type') and self.signal_type:
            if self.signal_type != self.signal_strategy.value:
                # Convert string to enum if needed
                try:
                    self.signal_strategy = SignalStrategy(self.signal_type)
                except ValueError:
                    logger.warning(f"Unknown signal_type '{self.signal_type}', using default")
        
        # Validate crypto-optimized thresholds
        if self.mean_return_ewm_threshold > 0.05:
            logger.warning(f"EWM threshold {self.mean_return_ewm_threshold} may be too high for crypto markets")
        
        if self.mean_return_simple_threshold > 0.05:
            logger.warning(f"Simple threshold {self.mean_return_simple_threshold} may be too high for crypto markets")
        
        if self.momentum_threshold > 0.05:
            logger.warning(f"Momentum threshold {self.momentum_threshold} may be too high for crypto markets")
        
        # Validate technical parameters
        if self.adx_period < 5 or self.adx_period > 50:
            raise ValueError(f"ADX period {self.adx_period} must be between 5 and 50")
        
        if self.rsi_oversold >= self.rsi_overbought:
            raise ValueError("RSI oversold level must be less than overbought level")
        
        # Validate spans and windows
        if self.mean_return_ewm_span < 2:
            raise ValueError("EWM span must be at least 2")
        
        if self.mean_return_simple_window < 1:
            raise ValueError("Simple window must be at least 1")
        
        # Validate volume parameters
        if self.volume_threshold <= 0:
            raise ValueError("Volume threshold must be positive")
        
        if self.volume_filter_multiple <= 0:
            raise ValueError("Volume filter multiple must be positive")
        
        # Crypto-specific validations
        if self.mean_return_ewm_span > 30:
            logger.warning(f"EWM span {self.mean_return_ewm_span} may be too slow for crypto volatility")
        
        if self.momentum_ewma_span > 50:
            logger.warning(f"Momentum EWMA span {self.momentum_ewma_span} may be too slow for crypto")
        
        # Deprecation warnings
        if self.combine_signals:
            logger.warning("'combine_signals' parameter is deprecated and will be ignored")
        
        logger.debug("SignalConfig validated with crypto-optimized parameters")


@dataclass
class PortfolioConfig:
    """Configuration for portfolio construction and optimization."""
    
    # Initial capital and position sizing
    initial_capital: float = 1_000_000.0  # $1M starting capital
    max_position_size: float = 0.10       # 10% max per asset
    min_position_size: float = 0.01       # 1% minimum position
    
    # Rebalancing
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    base_rebalance_frequency: str = "weekly"  # Base frequency for dynamic rebalancing
    use_dynamic_rebalancing: bool = True   # Enable dynamic rebalancing
    rebalance_threshold: float = 0.05      # 5% drift threshold for rebalancing
    
    # Optimization method
    optimization_method: OptimizationMethod = OptimizationMethod.ENHANCED_RISK_PARITY
    concentration_mode: bool = True        # Enable concentration mode
    top_n_assets: int = 5                  # Top N assets for concentration
    concentration_weight: float = 0.6      # Weight for top assets
    momentum_tilt_strength: float = 0.5    # Strength of momentum tilt
    use_momentum_weighting: bool = True    # Enable momentum weighting
    target_volatility: float = 0.15        # 15% annual volatility target
    
    # Risk constraints
    max_concentration: float = 0.20        # 20% max in any single asset
    sector_max_weight: float = 0.30        # 30% max per sector (if applicable)
    category_max_weights: Dict[str, float] = field(default_factory=lambda: {
        'defi': 0.4,
        'layer1': 0.5,
        'layer2': 0.3,
        'meme': 0.2,
        'exchange': 0.3
    })
    
    # Transaction costs consideration
    consider_transaction_costs: bool = True
    min_trade_size: float = 100.0          # $100 minimum trade
    
    # Cash management
    target_cash_allocation: float = 0.0    # 0% cash allocation by default
    max_cash_allocation: float = 0.05      # 5% max cash allowed
    
    def __post_init__(self):
        """Validate portfolio configuration parameters."""
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        
        if not 0 < self.max_position_size <= 1:
            raise ValueError("Max position size must be between 0 and 1")
        
        if not 0 <= self.min_position_size <= self.max_position_size:
            raise ValueError("Min position size must be between 0 and max position size")
        
        if not 0 <= self.target_volatility <= 1:
            raise ValueError("Target volatility must be between 0 and 1")
        
        if self.max_concentration > 1:
            raise ValueError("Max concentration cannot exceed 100%")
        
        logger.debug(f"PortfolioConfig validated: ${self.initial_capital:,.0f} capital, "
                    f"{self.max_position_size:.1%} max position")


@dataclass
class RiskConfig:
    """Configuration for risk management and monitoring."""
    
    # Drawdown controls
    max_drawdown: float = 0.20             # 20% max drawdown
    daily_var_limit: float = 0.05          # 5% daily VaR limit
    
    # Position-level risk
    position_var_limit: float = 0.10       # 10% position VaR limit
    correlation_threshold: float = 0.80    # 80% correlation warning threshold
    max_correlation: float = 0.70          # 70% max correlation allowed
    max_exchange_exposure: float = 0.4     # 40% max per exchange
    
    # Volatility management
    volatility_lookback: int = 30          # 30-day volatility calculation
    volatility_multiple: float = 2.0       # 2x volatility limit
    volatility_regime_multiplier: float = 1.5  # Regime volatility multiplier
    
    # Stop-loss and take-profit
    use_stop_losses: bool = True
    stop_loss_pct: float = 0.05            # 5% stop loss
    use_take_profits: bool = False
    take_profit_pct: float = 0.10          # 10% take profit
    use_trailing_stops: bool = True        # Enable trailing stops
    trailing_stop_atr_multiplier: float = 2.5  # ATR multiplier for trailing stops
    atr_period: int = 14                   # ATR calculation period
    atr_multiplier: float = 2.0            # ATR multiplier
    
    # Regime detection
    regime_lookback: int = 60              # Lookback for regime detection
    crisis_vol_threshold: float = 1.0      # Crisis volatility threshold
    
    # Portfolio-level risk
    leverage_limit: float = 1.0            # No leverage by default
    sector_concentration_limit: float = 0.30  # 30% max per sector
    
    # Risk monitoring
    risk_monitoring_frequency: str = "daily"
    max_drawdown_threshold: float = 0.2    # Max drawdown threshold
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'drawdown_warning': 0.10,          # 10% drawdown warning
        'volatility_warning': 0.25,        # 25% volatility warning
        'correlation_warning': 0.70        # 70% correlation warning
    })
    
    def __post_init__(self):
        """Validate risk configuration parameters."""
        if not 0 < self.max_drawdown <= 1:
            raise ValueError("Max drawdown must be between 0 and 1")
        
        if not 0 < self.daily_var_limit <= 1:
            raise ValueError("Daily VaR limit must be between 0 and 1")
        
        if self.stop_loss_pct <= 0 or self.stop_loss_pct >= 1:
            raise ValueError("Stop loss percentage must be between 0 and 1")
        
        if self.leverage_limit < 0:
            raise ValueError("Leverage limit cannot be negative")
        
        # Crypto-specific risk warnings
        if self.max_drawdown < 0.15:
            logger.warning("Max drawdown < 15% may be too restrictive for crypto markets")
        
        if self.volatility_multiple < 1.5:
            logger.warning("Volatility multiple < 1.5x may be too restrictive for crypto")
        
        logger.debug(f"RiskConfig validated: {self.max_drawdown:.1%} max drawdown, "
                    f"{self.stop_loss_pct:.1%} stop loss")


@dataclass
class CostConfig:
    """Configuration for transaction costs and market impact modeling."""
    
    # Trading fees (Binance-style)
    maker_fee: float = 0.001               # 0.1% maker fee
    taker_fee: float = 0.001               # 0.1% taker fee
    
    # Market impact
    base_spread: float = 0.0005            # 0.05% base bid-ask spread
    impact_coefficient: float = 0.0001     # Market impact coefficient
    
    # Slippage modeling
    use_slippage: bool = True
    slippage_coefficient: float = 0.0002   # 0.02% slippage coefficient
    max_slippage: float = 0.005            # 0.5% maximum slippage
    
    # Funding and borrowing (for leverage/short positions)
    funding_rate: float = 0.0001           # 0.01% daily funding rate
    borrow_rate: float = 0.0002            # 0.02% daily borrowing rate
    funding_lookback_days: int = 30        # Funding rate lookback period
    
    # Order execution
    fill_probability: float = 0.95         # 95% order fill probability
    partial_fill_threshold: float = 0.50   # 50% partial fill threshold
    
    def __post_init__(self):
        """Validate cost configuration parameters."""
        if self.maker_fee < 0 or self.taker_fee < 0:
            raise ValueError("Trading fees cannot be negative")
        
        if self.maker_fee > 0.01 or self.taker_fee > 0.01:
            logger.warning("Trading fees > 1% seem unusually high")
        
        if not 0 <= self.fill_probability <= 1:
            raise ValueError("Fill probability must be between 0 and 1")
        
        logger.debug(f"CostConfig validated: {self.maker_fee:.3%} maker, "
                    f"{self.taker_fee:.3%} taker fee")


@dataclass
class BacktestConfig:
    """Configuration for backtesting engine and execution."""
    
    # Execution parameters
    execution_delay: int = 0               # Execution delay in periods
    market_hours_only: bool = False        # Crypto markets are 24/7
    
    # Performance calculation
    benchmark_symbol: str = "BTCUSDT"      # Benchmark for comparison
    risk_free_rate: float = 0.02           # 2% annual risk-free rate
    
    # Validation and testing
    walk_forward_analysis: bool = False
    walk_forward_splits: int = 5           # Number of walk-forward splits
    out_of_sample_periods: int = 252       # 1 year out-of-sample
    monte_carlo_runs: int = 1000           # Monte Carlo simulations
    
    # Output and reporting
    save_trades: bool = True
    save_positions: bool = True
    save_metrics: bool = True
    output_directory: str = "output"
    
    # Performance attribution
    calculate_attribution: bool = True
    attribution_frequency: str = "monthly"
    
    def __post_init__(self):
        """Validate backtest configuration parameters."""
        if self.execution_delay < 0:
            raise ValueError("Execution delay cannot be negative")
        
        if not 0 <= self.risk_free_rate <= 0.10:
            logger.warning(f"Risk-free rate {self.risk_free_rate:.1%} seems unusual")
        
        if self.monte_carlo_runs < 100:
            logger.warning("Monte Carlo runs < 100 may not provide reliable results")
        
        logger.debug(f"BacktestConfig validated: {self.benchmark_symbol} benchmark, "
                    f"{self.monte_carlo_runs} MC runs")


@dataclass
class Config:
    """Main configuration class containing all sub-configurations."""
    
    data: DataConfig = field(default_factory=DataConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    
    # Global settings
    random_seed: Optional[int] = 42
    log_level: str = "INFO"
    parallel_processing: bool = True
    max_workers: int = 4
    
    # Regime parameters (optional)
    regime_parameters: Optional[Dict[str, Dict[str, float]]] = None
    
    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            file_path: Path to YAML configuration file
            
        Returns:
            Config instance loaded from file
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValueError: If configuration validation fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if data is None:
                data = {}
            
            logger.info(f"Loading configuration from {file_path}")
            config = cls._from_dict(data)
            
            # Add backward compatibility attributes for legacy code
            config._add_legacy_compatibility()
            
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {e}")
            raise
    
    def _add_legacy_compatibility(self):
        """Add legacy compatibility attributes for existing run.py and other scripts."""
        
        # Create strategy-like object for backward compatibility
        class StrategyCompat:
            def __init__(self, config):
                # Map data section to strategy-like attributes
                self.universe_size = config.data.universe_size
                self.start_date = config.data.start_date
                self.end_date = config.data.end_date
                self.rebalance_frequency = config.portfolio.rebalance_frequency.value
                self.max_position_size = config.portfolio.max_position_size
                
                # Map signal parameters for legacy compatibility
                self.signal_type = config.signals.signal_strategy.value
                self.adx_threshold = config.signals.adx_threshold
                self.momentum_threshold = config.signals.momentum_threshold
                
                # Common legacy aliases
                self.threshold = config.signals.momentum_threshold
                self.mean_return_threshold = config.signals.mean_return_ewm_threshold
                self.ewma_fast = getattr(config.signals, 'momentum_ewma_span', 20)
                self.ewma_slow = 50  # Default for legacy compatibility
                
                # Trading strategy parameters
                self.long_short = True  # Enable both long and short positions
                self.allow_short = True  # Legacy alias
                self.long_only = False   # Legacy parameter
                
                # Additional legacy parameters that might be needed
                self.lookback_period = getattr(config.signals, 'lookback', 20)
                self.volume_filter = getattr(config.signals, 'use_volume_confirmation', True)
                self.min_volume_ratio = getattr(config.signals, 'volume_threshold', 1.2)
        
        # Add strategy attribute for backward compatibility
        self.strategy = StrategyCompat(self)
        
        logger.debug("Added legacy compatibility attributes to config")
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create Config instance from dictionary with validation and defaults."""
        try:
            # Extract section data with defaults
            data_config_data = data.get('data', {})
            signals_config_data = data.get('signals', {})
            portfolio_config_data = data.get('portfolio', {})
            risk_config_data = data.get('risk', {})
            costs_config_data = data.get('costs', {})
            backtest_config_data = data.get('backtest', {})
            
            # Convert enum string values
            cls._convert_enum_values(signals_config_data, portfolio_config_data, backtest_config_data)
            
            # Clean signals data of unknown parameters for backward compatibility
            signals_config_data = cls._clean_signals_config(signals_config_data)
            
            # Create sub-configurations
            config = cls(
                data=DataConfig(**data_config_data),
                signals=SignalConfig(**signals_config_data),
                portfolio=PortfolioConfig(**portfolio_config_data),
                risk=RiskConfig(**risk_config_data),
                costs=CostConfig(**costs_config_data),
                backtest=BacktestConfig(**backtest_config_data),
                random_seed=data.get('random_seed', 42),
                log_level=data.get('log_level', 'INFO'),
                parallel_processing=data.get('parallel_processing', True),
                max_workers=data.get('max_workers', 4),
                regime_parameters=data.get('regime_parameters', None)
            )
            
            # Validate cross-section dependencies
            config._validate_cross_dependencies()
            
            logger.info("Configuration loaded and validated successfully")
            logger.info(f"  Data: {config.data.universe_size} assets, {config.data.start_date} to {config.data.end_date}")
            logger.info(f"  Signals: {config.signals.signal_strategy.value} strategy")
            logger.info(f"  Portfolio: {config.portfolio.optimization_method.value}, {config.portfolio.rebalance_frequency.value}")
            logger.info(f"  Risk: {config.risk.max_drawdown:.1%} max drawdown")
            
            return config
            
        except Exception as e:
            logger.error(f"Error creating configuration from dictionary: {e}")
            raise ValueError(f"Invalid configuration: {e}")
    
    @staticmethod
    def _convert_enum_values(signals_data: Dict, portfolio_data: Dict, backtest_data: Dict) -> None:
        """Convert string enum values to enum instances."""
        # Convert signal strategy
        if 'signal_strategy' in signals_data:
            signals_data['signal_strategy'] = SignalStrategy(signals_data['signal_strategy'])
        
        # Convert rebalance frequency
        if 'rebalance_frequency' in portfolio_data:
            portfolio_data['rebalance_frequency'] = RebalanceFrequency(portfolio_data['rebalance_frequency'])
        
        # Convert optimization method
        if 'optimization_method' in portfolio_data:
            portfolio_data['optimization_method'] = OptimizationMethod(portfolio_data['optimization_method'])
    
    @staticmethod
    def _clean_signals_config(signals_data: Dict) -> Dict:
        """Clean signals configuration data of unknown/legacy parameters."""
        # Define valid SignalConfig parameters (including legacy compatibility)
        valid_params = {
            'signal_strategy', 'momentum_threshold', 'momentum_ewma_span',
            'mean_return_ewm_threshold', 'mean_return_ewm_span',
            'mean_return_simple_threshold', 'mean_return_simple_window',
            'adx_period', 'adx_threshold', 'rsi_period', 'rsi_oversold', 'rsi_overbought',
            'use_volume_confirmation', 'volume_threshold', 'volume_lookback',
            'min_signal_gap', 'max_signals_per_period', 'signal_decay_periods',
            'use_regime_detection', 'volatility_adjustment', 'correlation_filter', 'correlation_threshold',
            # Enhanced signal parameters
            'adx_periods', 'base_ewma_span', 'adaptive_ewma', 'momentum_weights',
            'absolute_momentum_threshold', 'relative_momentum_threshold', 'momentum_lookback',
            'macd_fast', 'macd_slow', 'macd_signal', 'min_score_threshold', 'max_correlation',
            # Legacy compatibility parameters
            'ewma_fast', 'ewma_slow', 'threshold', 'lookback',
            'volume_filter_multiple', 'signal_type', 'combine_signals'
        }
        
        # Create cleaned config with only valid parameters
        cleaned_data = {}
        removed_params = []
        
        for key, value in signals_data.items():
            if key in valid_params:
                cleaned_data[key] = value
            else:
                removed_params.append(key)
        
        # Log removed parameters for transparency
        if removed_params:
            logger.warning(f"Removed unknown signal parameters from config: {removed_params}")
            logger.info("These parameters may be from an older version or custom implementation")
        
        # Apply crypto-optimized defaults for missing critical parameters
        crypto_defaults = {
            'mean_return_ewm_threshold': 0.015,    # Crypto-optimized
            'mean_return_simple_threshold': 0.015, # Crypto-optimized  
            'momentum_threshold': 0.01,            # Crypto-optimized
            'mean_return_ewm_span': 10,            # More responsive for crypto
            'mean_return_simple_window': 10,       # Balanced for crypto
            # Legacy compatibility defaults
            'ewma_fast': 20,
            'ewma_slow': 50,
            'threshold': 0.01,
            'lookback': 20,
            'volume_filter_multiple': 1.5,
            'signal_type': 'momentum',
            'combine_signals': False
        }
        
        applied_defaults = []
        for param, default_value in crypto_defaults.items():
            if param not in cleaned_data:
                cleaned_data[param] = default_value
                applied_defaults.append(f"{param}={default_value}")
        
        if applied_defaults:
            logger.info(f"Applied crypto-optimized defaults: {', '.join(applied_defaults)}")
        
        return cleaned_data
    
    def _validate_cross_dependencies(self) -> None:
        """Validate dependencies between different configuration sections."""
        # Validate data period is sufficient for signal parameters
        data_days = (self.data.end_date - self.data.start_date).days
        min_required_days = max(
            self.signals.momentum_ewma_span,
            self.signals.mean_return_simple_window,
            self.signals.adx_period
        ) * 2  # 2x buffer
        
        if data_days < min_required_days:
            logger.warning(f"Data period ({data_days} days) may be insufficient for signal parameters (need ~{min_required_days} days)")
        
        # Validate position sizing constraints
        if self.portfolio.max_position_size * self.data.universe_size < 0.8:
            logger.warning("Max position size * universe size < 80% - portfolio may not be fully invested")
        
        # Validate risk limits are achievable
        if self.risk.max_drawdown < self.portfolio.max_position_size:
            logger.warning("Max drawdown limit is less than max position size - may be too restrictive")
    
    def to_yaml(self, file_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            file_path: Path where to save configuration
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary and handle enums
        config_dict = asdict(self)
        
        # Convert enum values to strings
        config_dict['signals']['signal_strategy'] = self.signals.signal_strategy.value
        config_dict['portfolio']['rebalance_frequency'] = self.portfolio.rebalance_frequency.value
        config_dict['portfolio']['optimization_method'] = self.portfolio.optimization_method.value
        
        # Handle base_rebalance_frequency which is already a string
        if 'base_rebalance_frequency' in config_dict['portfolio']:
            # It's already a string, no conversion needed
            pass
        
        # Convert dates to strings
        config_dict['data']['start_date'] = self.data.start_date.strftime('%Y-%m-%d')
        config_dict['data']['end_date'] = self.data.end_date.strftime('%Y-%m-%d')
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)
            
            logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration to {file_path}: {e}")
            raise
    
    def update_for_crypto_optimization(self) -> None:
        """Update configuration with crypto-optimized parameters based on empirical analysis."""
        logger.info("Applying crypto market optimizations...")
        
        # Update signal thresholds based on crypto analysis
        self.signals.mean_return_ewm_threshold = 0.015  # Empirically validated
        self.signals.mean_return_simple_threshold = 0.015
        self.signals.momentum_threshold = 0.01
        
        # Adjust time parameters for crypto volatility
        self.signals.mean_return_ewm_span = 10
        self.signals.mean_return_simple_window = 10
        
        # Increase risk limits for crypto markets
        if self.risk.max_drawdown < 0.15:
            self.risk.max_drawdown = 0.20  # Crypto markets are more volatile
        
        # Adjust rebalancing for crypto
        if self.portfolio.rebalance_frequency == RebalanceFrequency.QUARTERLY:
            self.portfolio.rebalance_frequency = RebalanceFrequency.MONTHLY
        
        logger.info("Crypto optimizations applied:")
        logger.info(f"  Mean return thresholds: {self.signals.mean_return_ewm_threshold}")
        logger.info(f"  EWM span: {self.signals.mean_return_ewm_span}")
        logger.info(f"  Max drawdown: {self.risk.max_drawdown:.1%}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of key configuration parameters."""
        return {
            'universe_size': self.data.universe_size,
            'date_range': f"{self.data.start_date} to {self.data.end_date}",
            'signal_strategy': self.signals.signal_strategy.value,
            'mean_return_ewm_threshold': self.signals.mean_return_ewm_threshold,
            'mean_return_simple_threshold': self.signals.mean_return_simple_threshold,
            'rebalance_frequency': self.portfolio.rebalance_frequency.value,
            'max_position_size': self.portfolio.max_position_size,
            'max_drawdown': self.risk.max_drawdown,
            'initial_capital': self.portfolio.initial_capital
        }
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        summary = self.get_summary()
        return (f"Config(universe={summary['universe_size']}, "
                f"strategy={summary['signal_strategy']}, "
                f"thresholds={summary['mean_return_ewm_threshold']}, "
                f"rebalance={summary['rebalance_frequency']})")