#!/usr/bin/env python3
"""
Script to check valid parameters for each configuration section.
"""

from crypto_momentum_backtest.utils.config import (
    DataConfig, SignalConfig, PortfolioConfig, 
    RiskConfig, CostConfig, BacktestConfig
)
import inspect
from dataclasses import fields


def get_valid_parameters(config_class):
    """Get valid parameters for a config class."""
    # For dataclasses, use fields
    try:
        valid_fields = {field.name: field.type for field in fields(config_class)}
        return valid_fields
    except:
        # Fallback to inspect
        signature = inspect.signature(config_class.__init__)
        return {
            param: p.annotation 
            for param, p in signature.parameters.items() 
            if param != 'self'
        }


def main():
    print("="*60)
    print("VALID CONFIGURATION PARAMETERS")
    print("="*60)
    
    config_classes = {
        'data': DataConfig,
        'signals': SignalConfig,
        'portfolio': PortfolioConfig,
        'risk': RiskConfig,
        'costs': CostConfig,
        'backtest': BacktestConfig
    }
    
    for section, config_class in config_classes.items():
        print(f"\n{section.upper()} SECTION:")
        print("-"*40)
        
        params = get_valid_parameters(config_class)
        
        for param, param_type in params.items():
            type_str = str(param_type).replace('typing.', '').replace('<class \'', '').replace('\'>', '')
            print(f"  {param}: {type_str}")
    
    print("\n" + "="*60)
    print("NOTES:")
    print("="*60)
    print("1. max_position_size belongs in 'portfolio' section, not 'risk'")
    print("2. For long/short trading, use 'strategy.long_short: true'")
    print("3. Signal strategy should be 'momentum_score' for long/short")
    print("4. Some parameters might require custom additions to config.py")


if __name__ == "__main__":
    main()