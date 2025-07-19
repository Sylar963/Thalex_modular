"""
Thalex API Keys Configuration

This file contains the API keys for connecting to Thalex exchange.
It now loads keys from environment variables.
"""

import os
from thalex import Network

# Define the names of the environment variables
THALEX_TEST_API_KEY_ID_ENV = "THALEX_TEST_API_KEY_ID"
THALEX_TEST_PRIVATE_KEY_ENV = "THALEX_TEST_PRIVATE_KEY"
THALEX_PROD_API_KEY_ID_ENV = "THALEX_PROD_API_KEY_ID"
THALEX_PROD_PRIVATE_KEY_ENV = "THALEX_PROD_PRIVATE_KEY"

# Instructions for setting environment variables:
# Create a .env file in your project root or set system environment variables.
# Example .env file content:
# THALEX_TEST_API_KEY_ID="your_test_api_key_id"
# THALEX_TEST_PRIVATE_KEY="""your_test_private_key""" # Note: This python triple quote is for example in python code
                                                     # In .env, use normal double quotes for multi-line
# THALEX_PROD_API_KEY_ID="your_prod_api_key_id"
# THALEX_PROD_PRIVATE_KEY="""your_prod_private_key"""
#
# The application logic that uses these keys should handle cases where
# environment variables are not set (os.getenv() will return None).

# API Key IDs loaded from environment variables - using lazy loading functions
def get_key_ids():
    """Get API key IDs from environment variables at runtime"""
    return {
        Network.TEST: os.getenv(THALEX_TEST_API_KEY_ID_ENV),
        Network.PROD: os.getenv(THALEX_PROD_API_KEY_ID_ENV)
    }

# Private Keys loaded from environment variables - using lazy loading functions
# Ensure the private key environment variable includes the full multi-line string,
# including -----BEGIN...----- and -----END...----- markers.
def get_private_keys():
    """Get private keys from environment variables at runtime"""
    return {
        Network.TEST: os.getenv(THALEX_TEST_PRIVATE_KEY_ENV),
        Network.PROD: os.getenv(THALEX_PROD_PRIVATE_KEY_ENV)
    }

# Use get_key_ids() and get_private_keys() functions instead of these variables
# to ensure environment variables are loaded before accessing keys

# Ensure you have a .env file or have set these environment variables
# where your application runs. For example, using python-dotenv package to load .env file.
#
# To verify if keys are loaded (example):
# if not key_ids[Network.TEST] or not private_keys[Network.TEST]:
#     print("Warning: Testnet API keys are not set in environment variables.")
# if not key_ids[Network.PROD] or not private_keys[Network.PROD]:
#     print("Warning: Production API keys are not set in environment variables.") 