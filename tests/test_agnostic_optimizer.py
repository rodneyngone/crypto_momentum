#!/usr/bin/env python3
"""
Test script for Agnostic Risk Parity optimizer.

Verifies that ARP achieves equal risk allocation across principal components
when signals are i.i.d. and visualizes the risk decomposition.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from crypto_momentum_backtest.portfolio.agnostic_optimizer import AgnosticRiskParityOptimizer


def generate_test_data(n_assets=10, n_periods=252):
    """Generate synthetic test data with known properties."""
    np.random.seed(42)
    
    # Generate returns with factor structure
    n_factors = 3
    factor_loadings = np.random.randn(n_assets, n_factors)
    factor_returns = np.random.randn(n_periods, n_factors) * 0.02
    idiosyncratic = np.random.randn(n_periods, n_assets) * 0.01
    
    returns = factor_returns @ factor_loadings.T + idiosyncratic
    
    # Calculate covariance
    covariance = np.cov(returns.T)
    
    # Generate i.i.d. signals (uncorrelated with unit variance)
    signals = np.random.randn(n_assets)
    signals = signals / np.std(signals)  # Normalize to unit variance
    
    return returns, covariance, signals


def test_equal_risk_contribution():
    """Test that ARP achieves equal risk contribution across principal components."""
    print("=" * 60)
    print("Testing Agnostic Risk Parity Implementation")
    print("=" * 60)
    
    # Generate test data
    returns, covariance, signals = generate_test_data()
    n_assets = len(signals)
    
    # Initialize optimizer
    optimizer = AgnosticRiskParityOptimizer(budget_constraint='full_investment')
    
    # Get ARP weights
    arp_weights = optimizer.optimize(covariance, signals)
    
    print(f"\n1. Portfolio Weights:")
    print(f"   Sum of weights: {np.sum(arp_weights):.6f}")
    print(f"   Sum of |weights|: {np.sum(np.abs(arp_weights)):.6f}")
    print(f"   Long positions: {np.sum(arp_weights > 0)}")
    print(f"   Short positions: {np.sum(arp_weights < 0)}")
    
    # Calculate risk contributions
    risk_metrics = optimizer.calculate_risk_contributions(arp_weights, covariance)
    
    print(f"\n2. Risk Contributions:")
    print(f"   Equal risk score: {risk_metrics['equal_risk_score']:.4f} (1.0 = perfect parity)")
    print(f"   Concentration ratio: {risk_metrics['concentration_ratio']:.4f}")
    
    # Check principal component contributions
    pc_contributions = risk_metrics['pc_contributions']
    pc_std = np.std(pc_contributions)
    pc_mean = np.mean(pc_contributions)
    
    print(f"\n3. Principal Component Risk Contributions:")
    print(f"   Mean: {pc_mean:.4f} (ideal: {1/n_assets:.4f})")
    print(f"   Std Dev: {pc_std:.4f} (ideal: 0.0)")
    print(f"   Min: {np.min(pc_contributions):.4f}")
    print(f"   Max: {np.max(pc_contributions):.4f}")
    
    # Test passes if PC contributions are approximately equal
    tolerance = 0.05  # 5% tolerance
    is_equal_risk = pc_std / pc_mean < tolerance
    
    print(f"\n4. Test Result: {'PASSED' if is_equal_risk else 'FAILED'}")
    print(f"   Coefficient of variation: {pc_std/pc_mean:.4f} (<{tolerance} required)")
    
    return risk_metrics, arp_weights, covariance


def test_shrinkage_parameter():
    """Test the shrinkage parameter interpolation between ARP and Markowitz."""
    print("\n" + "=" * 60)
    print("Testing Shrinkage Parameter")
    print("=" * 60)
    
    returns, covariance, signals = generate_test_data()
    optimizer = AgnosticRiskParityOptimizer(budget_constraint='full_investment')
    
    # Test different shrinkage values
    shrinkage_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []
    
    for phi in shrinkage_values:
        weights = optimizer.optimize(covariance, signals, shrinkage_factor=phi)
        risk_metrics = optimizer.calculate_risk_contributions(weights, covariance)
        
        results.append({
            'shrinkage': phi,
            'weights': weights,
            'equal_risk_score': risk_metrics['equal_risk_score'],
            'concentration': risk_metrics['concentration_ratio']
        })
        
        print(f"\nφ = {phi:.2f}:")
        print(f"  Equal risk score: {risk_metrics['equal_risk_score']:.4f}")
        print(f"  Concentration: {risk_metrics['concentration_ratio']:.4f}")
    
    return results


def visualize_risk_decomposition(risk_metrics, weights, covariance):
    """Visualize risk contributions across assets and principal components."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Asset weights
    ax = axes[0, 0]
    asset_names = [f'Asset {i+1}' for i in range(len(weights))]
    colors = ['green' if w > 0 else 'red' for w in weights]
    ax.bar(asset_names, weights, color=colors, alpha=0.7)
    ax.set_title('Portfolio Weights')
    ax.set_xlabel('Assets')
    ax.set_ylabel('Weight')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Asset risk contributions
    ax = axes[0, 1]
    asset_contrib = risk_metrics['asset_contributions']
    ax.bar(asset_names, asset_contrib, alpha=0.7)
    ax.axhline(y=1/len(weights), color='red', linestyle='--', 
               label=f'Equal contribution ({1/len(weights):.3f})')
    ax.set_title('Asset Risk Contributions')
    ax.set_xlabel('Assets')
    ax.set_ylabel('Risk Contribution')
    ax.legend()
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Principal component contributions
    ax = axes[1, 0]
    pc_contrib = risk_metrics['pc_contributions']
    pc_names = [f'PC {i+1}' for i in range(len(pc_contrib))]
    ax.bar(pc_names, pc_contrib, alpha=0.7, color='orange')
    ax.axhline(y=1/len(pc_contrib), color='red', linestyle='--',
               label=f'Equal contribution ({1/len(pc_contrib):.3f})')
    ax.set_title('Principal Component Risk Contributions (ARP Target)')
    ax.set_xlabel('Principal Components')
    ax.set_ylabel('Risk Contribution')
    ax.legend()
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Eigenvalue spectrum
    ax = axes[1, 1]
    eigenvalues, _ = np.linalg.eigh(covariance)
    eigenvalues = sorted(eigenvalues, reverse=True)
    ax.plot(range(1, len(eigenvalues)+1), eigenvalues, 'o-')
    ax.set_title('Eigenvalue Spectrum of Covariance Matrix')
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare_optimizers():
    """Compare ARP with traditional Markowitz optimization."""
    print("\n" + "=" * 60)
    print("Comparing ARP vs Markowitz")
    print("=" * 60)
    
    returns, covariance, signals = generate_test_data(n_assets=8)
    optimizer = AgnosticRiskParityOptimizer(budget_constraint='full_investment')
    
    # Get both sets of weights
    comparison = optimizer.compare_with_markowitz(covariance, signals)
    
    # Calculate risk metrics for both
    arp_metrics = optimizer.calculate_risk_contributions(comparison['arp'], covariance)
    mark_metrics = optimizer.calculate_risk_contributions(comparison['markowitz'], covariance)
    
    print("\nARP Portfolio:")
    print(f"  Equal risk score: {arp_metrics['equal_risk_score']:.4f}")
    print(f"  Concentration: {arp_metrics['concentration_ratio']:.4f}")
    print(f"  PC risk std dev: {np.std(arp_metrics['pc_contributions']):.4f}")
    
    print("\nMarkowitz Portfolio:")
    print(f"  Equal risk score: {mark_metrics['equal_risk_score']:.4f}")
    print(f"  Concentration: {mark_metrics['concentration_ratio']:.4f}")
    print(f"  PC risk std dev: {np.std(mark_metrics['pc_contributions']):.4f}")
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ARP PC contributions
    ax = axes[0]
    pc_names = [f'PC{i+1}' for i in range(len(arp_metrics['pc_contributions']))]
    x = np.arange(len(pc_names))
    width = 0.35
    
    ax.bar(x - width/2, arp_metrics['pc_contributions'], width, label='ARP', alpha=0.8)
    ax.bar(x + width/2, mark_metrics['pc_contributions'], width, label='Markowitz', alpha=0.8)
    ax.axhline(y=1/len(pc_names), color='red', linestyle='--', label='Equal contribution')
    
    ax.set_xlabel('Principal Components')
    ax.set_ylabel('Risk Contribution')
    ax.set_title('PC Risk Contributions: ARP vs Markowitz')
    ax.set_xticks(x)
    ax.set_xticklabels(pc_names)
    ax.legend()
    
    # Weight comparison
    ax = axes[1]
    asset_names = [f'A{i+1}' for i in range(len(comparison['arp']))]
    x = np.arange(len(asset_names))
    
    ax.bar(x - width/2, comparison['arp'], width, label='ARP', alpha=0.8)
    ax.bar(x + width/2, comparison['markowitz'], width, label='Markowitz', alpha=0.8)
    
    ax.set_xlabel('Assets')
    ax.set_ylabel('Weight')
    ax.set_title('Portfolio Weights: ARP vs Markowitz')
    ax.set_xticks(x)
    ax.set_xticklabels(asset_names)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    return fig


def main():
    """Run all tests and generate visualizations."""
    # Create output directory for plots
    output_dir = Path("tests/output")
    output_dir.mkdir(exist_ok=True)
    
    # Test 1: Equal risk contribution
    risk_metrics, weights, covariance = test_equal_risk_contribution()
    
    # Visualize risk decomposition
    fig1 = visualize_risk_decomposition(risk_metrics, weights, covariance)
    fig1.savefig(output_dir / "arp_risk_decomposition.png", dpi=150, bbox_inches='tight')
    
    # Test 2: Shrinkage parameter
    shrinkage_results = test_shrinkage_parameter()
    
    # Test 3: Compare with Markowitz
    fig2 = compare_optimizers()
    fig2.savefig(output_dir / "arp_vs_markowitz.png", dpi=150, bbox_inches='tight')
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print(f"Plots saved to {output_dir}")
    print("=" * 60)
    
    plt.show()


if __name__ == "__main__":
    main()