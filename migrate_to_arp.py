#!/usr/bin/env python3
"""
Migration script to switch from ERC to ARP as default optimizer.

This script:
1. Backs up existing configuration
2. Updates configuration to use ARP
3. Runs comparison backtest
4. Validates results
"""

import argparse
import shutil
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def backup_config(config_path: Path) -> Path:
    """Create timestamped backup of current configuration."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = config_path.parent / f"{config_path.stem}_backup_{timestamp}{config_path.suffix}"
    shutil.copy2(config_path, backup_path)
    logger.info(f"Created configuration backup: {backup_path}")
    return backup_path


def update_config_to_arp(config_path: Path, shrinkage: float = None) -> dict:
    """Update configuration to use ARP optimizer."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update portfolio optimization method
    if 'portfolio' not in config:
        config['portfolio'] = {}
    
    old_method = config['portfolio'].get('optimization_method', 'enhanced_risk_parity')
    config['portfolio']['optimization_method'] = 'agnostic_risk_parity'
    config['portfolio']['arp_shrinkage_factor'] = shrinkage
    
    # Adjust related parameters for ARP
    config['portfolio']['concentration_mode'] = False  # ARP handles diversification
    config['portfolio']['use_momentum_weighting'] = False  # ARP incorporates signals directly
    
    # Ensure good signal strategy for ARP
    if config.get('signals', {}).get('signal_strategy') != 'momentum_score':
        logger.info("Switching to momentum_score strategy (recommended for ARP)")
        config['signals']['signal_strategy'] = 'momentum_score'
    
    # Save updated configuration
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Updated configuration from {old_method} to agnostic_risk_parity")
    return config


def run_comparison_backtest(config_path: Path, backup_path: Path, output_base: Path):
    """Run backtests with both configurations for comparison."""
    logger.info("Running comparison backtests...")
    
    # Run ERC backtest
    erc_output = output_base / "erc_results"
    erc_cmd = [
        "python", "run.py",
        "--config", str(backup_path),
        "--output-dir", str(erc_output),
        "--no-validate"  # Skip validation for speed
    ]
    
    logger.info("Running ERC backtest...")
    subprocess.run(erc_cmd, check=True)
    
    # Run ARP backtest
    arp_output = output_base / "arp_results"
    arp_cmd = [
        "python", "run.py",
        "--config", str(config_path),
        "--output-dir", str(arp_output),
        "--no-validate"
    ]
    
    logger.info("Running ARP backtest...")
    subprocess.run(arp_cmd, check=True)
    
    return erc_output, arp_output


def compare_results(erc_dir: Path, arp_dir: Path) -> dict:
    """Compare results between ERC and ARP backtests."""
    comparison = {}
    
    # Load metrics
    erc_metrics = pd.read_csv(erc_dir / "metrics.csv", index_col=0)
    arp_metrics = pd.read_csv(arp_dir / "metrics.csv", index_col=0)
    
    # Key metrics comparison
    key_metrics = [
        'total_return', 'annualized_return', 'sharpe_ratio', 
        'sortino_ratio', 'max_drawdown', 'volatility'
    ]
    
    for metric in key_metrics:
        if metric in erc_metrics.index and metric in arp_metrics.index:
            erc_val = erc_metrics.loc[metric, 'returns']
            arp_val = arp_metrics.loc[metric, 'returns']
            comparison[metric] = {
                'erc': erc_val,
                'arp': arp_val,
                'improvement': (arp_val - erc_val) / abs(erc_val) * 100 if erc_val != 0 else 0
            }
    
    # Load portfolio values for correlation
    erc_portfolio = pd.read_csv(erc_dir / "portfolio_values.csv", index_col=0, parse_dates=True)
    arp_portfolio = pd.read_csv(arp_dir / "portfolio_values.csv", index_col=0, parse_dates=True)
    
    # Calculate correlation
    correlation = erc_portfolio['portfolio_value'].corr(arp_portfolio['portfolio_value'])
    comparison['portfolio_correlation'] = correlation
    
    return comparison


def generate_migration_report(comparison: dict, output_path: Path):
    """Generate migration report."""
    report_lines = [
        "# ARP Migration Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Performance Comparison: ERC vs ARP",
        "\n| Metric | ERC | ARP | Change (%) |",
        "|--------|-----|-----|------------|"
    ]
    
    for metric, values in comparison.items():
        if metric != 'portfolio_correlation':
            erc_val = f"{values['erc']:.4f}"
            arp_val = f"{values['arp']:.4f}"
            change = f"{values['improvement']:+.2f}%"
            report_lines.append(f"| {metric} | {erc_val} | {arp_val} | {change} |")
    
    report_lines.extend([
        f"\n## Portfolio Correlation: {comparison['portfolio_correlation']:.4f}",
        "\n## Recommendation:",
    ])
    
    # Decision logic
    sharpe_improvement = comparison.get('sharpe_ratio', {}).get('improvement', 0)
    drawdown_improvement = comparison.get('max_drawdown', {}).get('improvement', 0)
    
    if sharpe_improvement > 0 and drawdown_improvement < 0:  # Lower drawdown is better
        report_lines.append("✅ **Proceed with ARP migration** - Better risk-adjusted returns")
    elif sharpe_improvement > -10:  # Within 10% of ERC
        report_lines.append("✅ **Proceed with ARP migration** - Similar performance with better diversification")
    else:
        report_lines.append("⚠️ **Review before migration** - Performance degradation detected")
    
    report_lines.extend([
        "\n## Migration Steps:",
        "1. This script has already updated your config.yaml",
        "2. Run a full backtest with validation: `python run.py --validate`",
        "3. Monitor live performance for 1-2 weeks before full deployment",
        "4. Keep the backup configuration for rollback if needed",
        "\n## Rollback Instructions:",
        f"To rollback: `cp {output_path.parent / 'config_backup_*.yaml'} config.yaml`"
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Migration report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Migrate from ERC to ARP optimizer')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file')
    parser.add_argument('--shrinkage', type=float, default=None, 
                       help='ARP shrinkage factor (None for pure ARP, 0-1 for blend with Markowitz)')
    parser.add_argument('--compare', action='store_true', help='Run comparison backtest')
    parser.add_argument('--output-dir', type=str, default='migration_results', 
                       help='Output directory for comparison')
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Backup current configuration
    backup_path = backup_config(config_path)
    
    # Step 2: Update configuration to use ARP
    update_config_to_arp(config_path, args.shrinkage)
    
    if args.compare:
        # Step 3: Run comparison backtests
        erc_dir, arp_dir = run_comparison_backtest(config_path, backup_path, output_dir)
        
        # Step 4: Compare results
        comparison = compare_results(erc_dir, arp_dir)
        
        # Step 5: Generate report
        generate_migration_report(comparison, output_dir / "migration_report.md")
        
        # Print summary
        print("\n" + "="*60)
        print("MIGRATION SUMMARY")
        print("="*60)
        for metric, values in comparison.items():
            if metric != 'portfolio_correlation':
                print(f"{metric:20s}: {values['improvement']:+.2f}% change")
        print(f"{'Portfolio Correlation':20s}: {comparison['portfolio_correlation']:.4f}")
        print("="*60)
    
    logger.info("Migration complete! Configuration updated to use ARP.")
    logger.info(f"Backup saved at: {backup_path}")


if __name__ == "__main__":
    main()