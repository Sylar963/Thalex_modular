#!/usr/bin/env python3
"""
Enable or disable hedging in the trading system.
This script modifies the market_config.py file to enable/disable hedging.
"""

import sys
import os
import re
import shutil
from pathlib import Path

def enable_hedging(enable=True):
    """Enable or disable hedging in the market_config.py file"""
    # Find the market_config.py file
    config_path = Path("thalex_py/Thalex_modular/config/market_config.py")
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return False
    
    # Create a backup
    backup_path = config_path.with_suffix(".py.bak")
    shutil.copy(config_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Read the config file
    with open(config_path, "r") as f:
        content = f.read()
    
    # Find the hedging section and update the enabled value
    pattern = r'("hedging"\s*:\s*{\s*"enabled"\s*:\s*)(?:True|False)'
    
    # Set the new value
    new_value = "True" if enable else "False"
    
    # Replace the value
    if re.search(pattern, content):
        content = re.sub(pattern, f"\\1{new_value}", content)
        
        # Write the updated content
        with open(config_path, "w") as f:
            f.write(content)
        
        print(f"Hedging {'enabled' if enable else 'disabled'} in {config_path}")
        return True
    else:
        print("Error: Could not find hedging configuration in the file")
        return False

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python enable_hedging.py [enable|disable]")
        return
    
    command = sys.argv[1].lower()
    
    if command == "enable":
        enable_hedging(True)
    elif command == "disable":
        enable_hedging(False)
    else:
        print(f"Unknown command: {command}")
        print("Usage: python enable_hedging.py [enable|disable]")

if __name__ == "__main__":
    main() 