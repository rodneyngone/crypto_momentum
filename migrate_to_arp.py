#!/usr/bin/env python3
"""
Migration script to switch from ERC to ARP as default optimizer.
Includes automatic config.py patching if needed.
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
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def patch_config_file():
    """Patch config.py to include ARP in OptimizationMethod enum if missing."""
    config_path = Path("crypto_momentum_backtest/utils/config.py")
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Check if AGNOSTIC_RISK_PARITY already exists
    if 'AGNOSTIC_RISK_PARITY' in content:
        logger.info("Config already supports ARP optimization method")
        return False
    
    # Find the OptimizationMethod enum
    pattern = r'(class OptimizationMethod\(Enum\):.*?)(ENHANCED_RISK_PARITY = "enhanced_risk_parity")'
    replacement = r'\1\2\n    AGNOSTIC_RISK_PARITY = "agnostic_risk_parity"'
    
    # Apply the patch
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    if new_content != content:
        # Backup original
        backup_path = config_path.with_suffix('.py.bak')
        shutil.copy2(config_path, backup_path)
        logger.info(f"Created config backup: {backup_path}")
        
        # Write patched version
        with open(config_path, 'w') as f:
            f.write(new_content)
        
        logger.info("Patched config.py to support ARP optimization method")
        return True
    else:
        logger.error("Failed to patch config.py - please add AGNOSTIC_RISK_PARITY manually")
        return False


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


def restore_erc_config(backup_path: Path, target_path: Path):
    """Restore ERC configuration for comparison."""
    with open(backup_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure it's using ERC
    if 'portfolio' not in config:
        config['portfolio'] = {}
    
    # If it was already ARP, change to enhanced_risk_parity
    if config['portfolio'].get('optimization_method') == 'agnostic_risk_parity':
        config['portfolio']['optimization_method'] = 'enhanced_risk_parity'
        config['portfolio'].pop('arp_shrinkage_factor', None)
        config['portfolio']['concentration_mode'] = True
        config['portfolio']['use_momentum_weighting'] = True
    
    # Save ERC config
    erc_config_path = target_path.parent / f"{target_path.stem}_erc{target_path.suffix}"
    with open(erc_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return erc_config_path


def run_comparison_backtest(config_path: Path, backup_path: Path, output_base: Path):
    """Run backtests with both configurations for comparison."""
    logger.info("Running comparison backtests...")
    
    # Create ERC config from backup
    erc_config_path = restore_erc_config(backup_path, config_path)
    
    # Run ERC backtest
    erc_output = output_base / "erc_results"
    erc_cmd = [
        "python", "run.py",
        "--config", str(erc_config_path),
        "--output-dir", str(erc_output),
        "--no-validate"  # Skip validation for speed
    ]
    
    logger.info("Running ERC backtest...")
    try:
        subprocess.run(erc_cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"ERC backtest failed: {e}")
        raise
    
    # Run ARP backtest
    arp_output = output_base / "arp_results"
    arp_cmd = [
        "python", "run.py",
        "--config", str(config_path),
        "--output-dir", str(arp_output),
        "--no-validate"
    ]
    
    logger.info("Running ARP backtest...")
    try:
        subprocess.run(arp_cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"ARP backtest failed: {e}")
        raise
    
    # Clean up temporary ERC config
    erc_config_path.unlink()
    
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
    parser.add_argument('--patch-config', action='store_true', default=True,
                       help='Automatically patch config.py if needed')
    
    args = parser.parse_args()
    
    # Step 0: Patch config.py if needed
    if args.patch_config:
        patch_config_file()
    
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Backup current configuration
    backup_path = backup_config(config_path)
    
    # Step 2: Update configuration to use ARP
    update_config_to_arp(config_path, args.shrinkage)
    
    if args.compare:
        # Step 3: Run comparison backtests
        try:
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
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            logger.info("Migration to ARP completed but comparison could not be run")
    
    logger.info("Migration complete! Configuration updated to use ARP.")
    logger.info(f"Backup saved at: {backup_path}")


if __name__ == "__main__":
    main()