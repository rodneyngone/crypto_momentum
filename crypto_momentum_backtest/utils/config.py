"""Updated configuration management with mean return parameters."""
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class StrategyConfig:
    """Strategy configuration parameters."""
    universe_size: int = 20
    rebalance_frequency: str = "monthly"
    max_position_size: float = 0.20
    long_short: bool = True
    
@dataclass
class SignalConfig:
    """Signal generation configuration with mean return options."""
    # Original momentum parameters
    adx_period: int = 14
    adx_threshold: float = 20.0
    ewma_fast: int = 20
    ewma_slow: int = 50
    volume_filter_multiple: float = 1.5
    
    # New mean return parameters
    use_mean_return_signal: bool = False
    mean_return_window: int = 30
    mean_return_type: str = "ewm"  # "simple" or "ewm"
    mean_return_threshold: float = 0.01  # 1% default
    mean_return_ewm_span: Optional[int] = None  # If None, uses mean_return_window
    combine_signals: str = "override"  # "override", "and", "or"
    
@dataclass
class RiskConfig:
    """Risk management configuration."""
    atr_period: int = 14
    atr_multiplier: float = 2.0
    max_correlation: float = 0.8
    volatility_regime_multiplier: float = 1.5
    max_drawdown_threshold: float = 0.15
    max_exchange_exposure: float = 0.40
    
@dataclass
class CostConfig:
    """Transaction cost configuration."""
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004
    base_spread: float = 0.0001
    funding_lookback_days: int = 30
    
@dataclass
class Config:
    """Main configuration container."""
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'Config':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)
    
    @classmethod
    def from_json(cls, path: Path) -> 'Config':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        # Handle signals config with defaults for new parameters
        signals_data = data.get('signals', {})
        
        # Set default for mean_return_ewm_span if not provided
        if 'mean_return_ewm_span' not in signals_data and 'mean_return_window' in signals_data:
            signals_data['mean_return_ewm_span'] = signals_data['mean_return_window']
        
        return Config(
            strategy=StrategyConfig(**data.get('strategy', {})),
            signals=SignalConfig(**signals_data),
            risk=RiskConfig(**data.get('risk', {})),
            costs=CostConfig(**data.get('costs', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            'strategy': self.strategy.__dict__,
            'signals': self.signals.__dict__,
            'risk': self.risk.__dict__,
            'costs': self.costs.__dict__
        }
    
    def save_yaml(self, path: Path):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def describe_signals(self) -> str:
        """Get a description of the signal configuration."""
        if not self.signals.use_mean_return_signal:
            return f"Momentum signals: ADX>{self.signals.adx_threshold}, EWMA {self.signals.ewma_fast}/{self.signals.ewma_slow}"
        else:
            signal_type = f"{self.signals.mean_return_window}-day {self.signals.mean_return_type} mean return"
            threshold = f"{self.signals.mean_return_threshold:.1%}"
            
            if self.signals.combine_signals == "override":
                return f"Mean return only: {signal_type} > {threshold}"
            elif self.signals.combine_signals == "and":
                return f"Momentum AND {signal_type} > {threshold}"
            else:  # "or"
                return f"Momentum OR {signal_type} > {threshold}"