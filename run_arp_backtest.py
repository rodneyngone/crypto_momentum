#!/usr/bin/env python3
"""
Example script demonstrating Agnostic Risk Parity (ARP) in the crypto momentum backtest.

This shows how to configure and run a backtest using ARP instead of standard ERC.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Assume we're in the examples directory
import sys
sys.path.append(str(Path(__file__).parent.parent))

from crypto_momentum_backtest.utils.config import Config
from crypto_momentum_backtest.portfolio import ARPOptimizer
from crypto_momentum_backtest.data.json_storage import JsonStorage


def demonstrate_arp_standalone():
    """Demonstrate ARP optimizer in standalone mode."""
    print("=" * 60)
    print("Agnostic Risk Parity (ARP) Demonstration")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_assets = 5
    asset_names = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA']
    
    # Generate sample returns (252 days)
    returns = pd.DataFrame(
        np.random.multivariate_normal(
            mean=[0.001] * n_assets,
            cov=np.eye(n_assets) * 0.04 + np.ones((n_assets, n_assets)) * 0.01,
            size=252
        ),
        columns=asset_names
    )
    
    # Calculate covariance
    covariance = returns.cov()
    
    # Generate momentum signals (these would come from your signal generator)
    signals = pd.Series({
        'BTC': 0.8,   # Strong positive momentum
        'ETH': 0.6,   # Moderate positive
        'BNB': -0.3,  # Slight negative
        'SOL': 0.9,   # Very strong positive
        'ADA': 0.2    # Weak positive
    })
    
    print("\n1. Input Signals (Momentum Scores):")
    print(signals)
    
    # Initialize ARP optimizer
    optimizer = ARPOptimizer(
        target_volatility=0.15,  # 15% annual volatility target
        budget_constraint='full_investment',
        logger=logging.getLogger()
    )
    
    # Calculate ARP weights
    arp_weights = optimizer.optimize(covariance, signals)
    
    print("\n2. ARP Portfolio Weights:")
    print(arp_weights.round(4))
    print(f"\nSum of |weights|: {np.sum(np.abs(arp_weights)):.4f}")
    
    # Calculate risk contributions
    risk_metrics = optimizer.calculate_risk_contributions(arp_weights, covariance)
    
    print("\n3. Risk Analysis:")
    print(f"Equal Risk Score: {risk_metrics['equal_risk_score']:.4f} (1.0 = perfect parity)")
    print("\nAsset Risk Contributions:")
    for asset, contrib in zip(asset_names, risk_metrics['asset_contributions']):
        print(f"  {asset}: {contrib:.4f}")
    
    # Compare with different shrinkage factors
    print("\n4. Shrinkage Analysis (Q = φC + (1-φ)I):")
    print("φ=0: Pure ARP (Q=I), φ=1: Pure Markowitz (Q=C)")
    
    for shrinkage in [0.0, 0.25, 0.5, 0.75, 1.0]:
        weights_shrunk = optimizer.optimize(covariance, signals, shrinkage_factor=shrinkage)
        vol = np.sqrt(weights_shrunk @ covariance @ weights_shrunk) * np.sqrt(252)
        print(f"\nφ={shrinkage:.2f}: Volatility={vol:.2%}")
        print(f"  Weights: {dict(weights_shrunk.round(3))}")


def create_arp_config():
    """Create a configuration file for ARP backtesting."""
    config_dict = {
        'data': {
            'universe_size': 10,
            'start_date': '2022-01-01',
            'end_date': '2023-12-31'
        },
        'signals': {
            'signal_strategy': 'momentum_score',  # Good for ARP - produces continuous scores
            'min_score_threshold': 0.2,
            'use_volume_confirmation': True
        },
        'portfolio': {
            'optimization_method': 'agnostic_risk_parity',
            'arp_shrinkage_factor': None,  # Pure ARP
            'target_volatility': 0.20,      # 20% volatility for crypto
            'base_rebalance_frequency': 'weekly',
            'use_dynamic_rebalancing': True
        },
        'risk': {
            'max_drawdown': 0.25,
            'use_trailing_stops': True
        }
    }
    
    # Save config
    import yaml
    config_path = Path('config_arp.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    print(f"\n5. Created ARP configuration: {config_path}")
    print("\nTo run a full backtest with ARP:")
    print(f"  python run.py --config {config_path}")
    
    return config_path


def explain_arp_benefits():
    """Explain the key benefits of ARP over traditional methods."""
    print("\n" + "=" * 60)
    print("Why Use Agnostic Risk Parity (ARP)?")
    print("=" * 60)
    
    benefits = """
1. **Rotation Invariance**: ARP is invariant to arbitrary rotations of the asset space.
   Traditional risk parity depends on the choice of "base assets" (stocks, bonds, etc.),
   but in crypto, what are the fundamental assets? ARP solves this philosophical problem.

2. **Unknown-Unknown Protection**: By assuming signal correlations are unknowable (Q=I),
   ARP protects against overconfident hedging that can backfire when correlations shift.

3. **Equal PC Risk**: ARP equalizes risk across ALL principal components of returns,
   not just across assets. This provides more robust diversification.

4. **Signal Integration**: Momentum/trend signals are incorporated directly into weights,
   unlike traditional risk parity which ignores expected returns.

5. **Crypto Suitability**: Crypto markets have unstable correlations and unclear asset
   hierarchies, making ARP's agnostic approach particularly valuable.

Mathematical Comparison:
- Markowitz: π = C⁻¹ · E[r]     (Assumes we know expected returns perfectly)
- Risk Parity: π ∝ 1/σᵢ         (Ignores expected returns entirely)  
- ARP: π = C⁻¹/² · p            (Uses signals without assuming their correlations)

The C⁻¹/² term ensures equal risk contribution across eigenvectors of C.
    """
    
    print(benefits)


def main():
    """Run all demonstrations."""
    # 1. Demonstrate standalone ARP usage
    demonstrate_arp_standalone()
    
    # 2. Create config file for full backtest
    create_arp_config()
    
    # 3. Explain benefits
    explain_arp_benefits()
    
    print("\n" + "=" * 60)
    print("ARP implementation ready for use!")
    print("=" * 60)


if __name__ == "__main__":
    main()