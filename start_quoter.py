#!/usr/bin/env python3
"""
Thalex Avellaneda Market Maker
High-performance market maker using the Avellaneda-Stoikov model
"""

import asyncio
import signal
import sys
import os
import time
from datetime import datetime

# Add the project directory to Python path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

import thalex as th
from thalex_py.Thalex_modular.config.market_config import BOT_CONFIG
from thalex_py.Thalex_modular.avellaneda_quoter import AvellanedaQuoter
from thalex_py.Thalex_modular.logging import LoggerFactory

# Global variables
quoter = None
shutdown_event = asyncio.Event()

async def shutdown():
    """Gracefully shut down the application"""
    print("Shutting down...")
    
    if quoter:
        await quoter.shutdown()
        
    # Final cleanup
    await LoggerFactory.shutdown()
    print("Shutdown complete.")

def signal_handler():
    """Handle termination signals"""
    print("Received termination signal")
    shutdown_event.set()

async def main():
    """Main entry point"""
    global quoter
    
    # Create a directory for logs
    os.makedirs("logs", exist_ok=True)
    
    # Initialize Thalex client
    network = BOT_CONFIG["market"]["network"]
    print(f"Starting on {network} network")
    thalex = th.Thalex(network=network)
    
    # Create quoter
    quoter = AvellanedaQuoter(thalex)
    
    # Set reference to quoter in thalex client for rate limit tracking
    thalex.quoter = quoter
    
    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        # Start the quoter
        start_task = asyncio.create_task(quoter.start())
        
        # Wait for shutdown signal
        await shutdown_event.wait()
        
        # Cancel the start task
        if not start_task.done():
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                pass
    except Exception as e:
        print(f"Fatal error in main loop: {str(e)}")
    finally:
        await shutdown()

if __name__ == "__main__":
    # Print startup banner
    print("="*80)
    print(f"Thalex Avellaneda Market Maker - Starting at {datetime.now()}")
    print(f"Version: 2.0.0 (High Performance)")
    print("="*80)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1) 