#!/usr/bin/env python3
"""
Simple fix: Just reduce the threshold in your existing config structure.
"""

import yaml
import shutil
from pathlib import Path

def simple_threshold_fix():
    """Simple fix - just reduce thresholds in existing configs"""
    
    print("SIMPLE THRESHOLD FIX")
    print("="*30)
    
    # Backup current config
    if Path("config.yaml").exists():
        shutil.copy("config.yaml", "config_backup_simple.yaml")
        print("✓ Backed up config.yaml")
    
    # Try to load config_mean_return.yaml (the one that was working before)
    config_files = ["config_mean_return.yaml", "config.yaml"]
    
    config_loaded = False
    for config_file in config_files:
        if Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"✓ Loaded {config_file}")
                config_loaded = True
                break
            except Exception as e:
                print(f"✗ Error loading {config_file}: {e}")
                continue
    
    if not config_loaded:
        print("✗ Could not load any config file")
        return False
    
    # Show current config
    print(f"\nCurrent config structure:")
    if 'signals' in config:
        for key, value in config['signals'].items():
            print(f"  {key}: {value}")
    
    # Find and reduce thresholds
    signals = config.get('signals', {})
    
    # Look for threshold-like parameters and reduce them
    threshold_params = ['threshold', 'signal_threshold', 'momentum_threshold', 
                       'mean_threshold', 'ewm_threshold']
    
    changes_made = []
    for param in threshold_params:
        if param in signals:
            old_value = signals[param]
            if isinstance(old_value, (int, float)) and old_value > 0.005:
                # Reduce threshold by 75% but keep minimum of 0.002
                new_value = max(0.002, old_value * 0.25)
                signals[param] = new_value
                changes_made.append(f"{param}: {old_value} -> {new_value}")
    
    # Also check for any parameter that might be too high
    for key, value in signals.items():
        if isinstance(value, (int, float)) and value >= 0.02 and 'period' not in key.lower():
            old_value = value
            new_value = max(0.005, value * 0.25)
            signals[key] = new_value
            changes_made.append(f"{key}: {old_value} -> {new_value}")
    
    # Save the fixed config
    with open("config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\nChanges made:")
    for change in changes_made:
        print(f"  {change}")
    
    if not changes_made:
        print("  No threshold changes needed - may already be optimized")
    
    print(f"\n✓ Saved fixed config.yaml")
    return True

def create_test_variants():
    """Create test variants with different thresholds"""
    
    try:
        with open("config.yaml", 'r') as f:
            base_config = yaml.safe_load(f)
    except:
        print("Could not load config.yaml for variants")
        return
    
    # Create variants with different threshold levels
    thresholds = [0.001, 0.002, 0.005, 0.007, 0.01]
    
    for threshold in thresholds:
        variant_config = base_config.copy()
        
        # Apply threshold to all threshold-like parameters
        if 'signals' in variant_config:
            for key, value in variant_config['signals'].items():
                if 'threshold' in key.lower():
                    variant_config['signals'][key] = threshold
        
        # Save variant
        filename = f"config_threshold_{threshold:.3f}.yaml"
        with open(filename, 'w') as f:
            yaml.dump(variant_config, f, default_flow_style=False)
        
        print(f"✓ Created {filename}")

def main():
    """Main function"""
    success = simple_threshold_fix()
    
    if success:
        print(f"\n" + "="*30)
        print("SIMPLE FIX COMPLETE!")
        print("="*30)
        
        print("\nNext steps:")
        print("1. Test the fix:")
        print("   python run.py --no-validate")
        print()
        print("2. If still having issues, try:")
        print("   python config_fixer.py")
        print()
        print("3. Create threshold test variants:")
        
        create_variants = input("Create test variants with different thresholds? (y/n): ").lower().strip()
        if create_variants == 'y':
            create_test_variants()
            print("\nTest different thresholds with:")
            print("   copy config_threshold_0.001.yaml config.yaml")
            print("   python run.py --no-validate")

if __name__ == "__main__":
    main()