#!/usr/bin/env python3
"""
Thalex Trading Bot Environment Setup Script

This script helps you set up your environment variables for the Thalex trading bot.
It will guide you through creating a .env file with the necessary configuration.
"""

import os
import shutil
from pathlib import Path

def create_env_file():
    """Create .env file from .example.env template"""
    example_env_path = Path(".example.env")
    env_path = Path(".env")
    
    if not example_env_path.exists():
        print("❌ Error: .example.env file not found!")
        print("Make sure you're running this script from the project root directory.")
        return False
    
    if env_path.exists():
        response = input("⚠️  .env file already exists. Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("Setup cancelled.")
            return False
    
    # Copy example file to .env
    shutil.copy2(example_env_path, env_path)
    print("✅ Created .env file from template")
    return True

def get_user_input(prompt, default="", required=True):
    """Get user input with optional default value"""
    if default:
        prompt_text = f"{prompt} [{default}]: "
    else:
        prompt_text = f"{prompt}: "
    
    while True:
        value = input(prompt_text).strip()
        if value:
            return value
        elif default:
            return default
        elif not required:
            return ""
        else:
            print("This field is required. Please enter a value.")

def setup_api_credentials():
    """Interactive setup for API credentials"""
    print("\n" + "="*60)
    print("🔑 API CREDENTIALS SETUP")
    print("="*60)
    
    print("\nYou need to get API credentials from Thalex exchange:")
    print("1. Login to your Thalex account")
    print("2. Navigate to API Management")  
    print("3. Create new API keys for testnet and/or production")
    print("4. Download the private key files")
    
    setup_testnet = input("\nSetup testnet credentials? (Y/n): ").strip().lower() != 'n'
    setup_production = input("Setup production credentials? (y/N): ").strip().lower() == 'y'
    
    credentials = {}
    
    if setup_testnet:
        print("\n📊 TESTNET CREDENTIALS:")
        credentials['THALEX_TEST_API_KEY_ID'] = get_user_input("Testnet API Key ID")
        print("Enter testnet private key (paste the entire key including BEGIN/END lines):")
        print("Press Enter twice when done:")
        private_key_lines = []
        while True:
            line = input()
            if line == "" and private_key_lines:
                break
            private_key_lines.append(line)
        credentials['THALEX_TEST_PRIVATE_KEY'] = '\n'.join(private_key_lines)
    
    if setup_production:
        print("\n🚨 PRODUCTION CREDENTIALS:")
        print("WARNING: These are for live trading with real money!")
        confirm = input("Are you sure you want to set up production credentials? (y/N): ").strip().lower()
        if confirm == 'y':
            credentials['THALEX_PROD_API_KEY_ID'] = get_user_input("Production API Key ID")
            print("Enter production private key (paste the entire key including BEGIN/END lines):")
            print("Press Enter twice when done:")
            private_key_lines = []
            while True:
                line = input()
                if line == "" and private_key_lines:
                    break
                private_key_lines.append(line)
            credentials['THALEX_PROD_PRIVATE_KEY'] = '\n'.join(private_key_lines)
    
    return credentials

def setup_basic_config():
    """Setup basic trading configuration"""
    print("\n" + "="*60)
    print("⚙️  BASIC CONFIGURATION")
    print("="*60)
    
    config = {}
    
    # Trading mode
    trading_mode = input("\nTrading mode (testnet/production) [testnet]: ").strip().lower()
    if trading_mode not in ['testnet', 'production']:
        trading_mode = 'testnet'
    config['TRADING_MODE'] = trading_mode
    config['NETWORK'] = 'test' if trading_mode == 'testnet' else 'prod'
    
    # Risk settings
    config['MAX_POSITION_SIZE'] = get_user_input("Maximum position size (BTC)", "0.01")
    config['BASE_QUOTE_SIZE'] = get_user_input("Base quote size (BTC)", "0.001")
    
    # Instruments
    config['PRIMARY_INSTRUMENT'] = get_user_input("Primary instrument", "BTC-PERPETUAL")
    
    return config

def update_env_file(credentials, config):
    """Update .env file with user-provided values"""
    env_path = Path(".env")
    
    if not env_path.exists():
        print("❌ Error: .env file not found!")
        return False
    
    # Read current .env file
    with open(env_path, 'r') as f:
        content = f.read()
    
    # Update credentials
    for key, value in credentials.items():
        # Handle multi-line private keys
        if 'PRIVATE_KEY' in key:
            # Replace the placeholder with the actual key
            placeholder = f'{key}="-----BEGIN PRIVATE KEY-----\nyour_'
            if 'test' in key.lower():
                placeholder += 'testnet_'
            else:
                placeholder += 'production_'
            placeholder += 'private_key_content_here\nmultiple_lines_are_supported\n-----END PRIVATE KEY-----"'
            
            replacement = f'{key}="{value}"'
            content = content.replace(placeholder, replacement, 1)
        else:
            # Replace simple key-value pairs
            placeholder = f'{key}="your_'
            if 'test' in key.lower():
                placeholder += 'testnet_'
            else:
                placeholder += 'production_'
            placeholder += 'api_key_id_here"'
            
            replacement = f'{key}="{value}"'
            content = content.replace(placeholder, replacement, 1)
    
    # Update configuration values
    for key, value in config.items():
        # Find and replace the configuration line
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith(f'{key}='):
                lines[i] = f'{key}="{value}"'
                break
        content = '\n'.join(lines)
    
    # Write updated content back to file
    with open(env_path, 'w') as f:
        f.write(content)
    
    print("✅ Updated .env file with your configuration")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['logs', 'metrics', 'backups', 'profiles']
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")

def main():
    """Main setup function"""
    print("🚀 Thalex Trading Bot Environment Setup")
    print("="*60)
    
    # Step 1: Create .env file
    if not create_env_file():
        return
    
    # Step 2: Setup API credentials
    credentials = setup_api_credentials()
    
    # Step 3: Setup basic configuration
    config = setup_basic_config()
    
    # Step 4: Update .env file
    if not update_env_file(credentials, config):
        return
    
    # Step 5: Create directories
    print("\n" + "="*60)
    print("📁 CREATING DIRECTORIES")
    print("="*60)
    create_directories()
    
    # Final instructions
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review and edit your .env file if needed")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Test the bot on testnet first: python start_quoter.py")
    print("4. Monitor logs in the logs/ directory")
    print("5. Check metrics in the metrics/ directory")
    print("\n⚠️  Important reminders:")
    print("- Never commit your .env file to version control")
    print("- Always test on testnet before using production")
    print("- Keep your API keys secure")
    print("- Start with small position sizes")
    print("\n🎉 Happy trading!")

if __name__ == "__main__":
    main() 