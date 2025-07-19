#!/usr/bin/env python3
"""
Thalex Avellaneda Market Maker
High-performance market maker using the Avellaneda-Stoikov model
"""

# Added: Load environment variables from .env file at the very beginning
from dotenv import load_dotenv
load_dotenv(override=True)

import asyncio
import signal
import sys
import os
import time
import argparse
import logging
from datetime import datetime

# Add the project directory to Python path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import thalex as th
from thalex_py.Thalex_modular.config.market_config import BOT_CONFIG, MARKET_CONFIG, TRADING_CONFIG
from thalex_py.Thalex_modular.avellaneda_quoter import AvellanedaQuoter
from thalex_py.Thalex_modular.thalex_logging import LoggerFactory
from thalex_py.Thalex_modular.ringbuffer.volume_candle_buffer import VolumeBasedCandleBuffer

# Global variables
quoter = None
shutdown_event = asyncio.Event()

# Ensure all required log directories exist
def create_log_directories():
    """Create all required log directories"""
    # Base logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Create subdirectories based on the LoggerFactory structure
    subdirs = [
        "market", "orders", "risk", 
        "performance", "exchange", "positions"
    ]
    for subdir in subdirs:
        os.makedirs(os.path.join("logs", subdir), exist_ok=True)
    
    # Also create metrics directory
    os.makedirs("metrics", exist_ok=True)

async def shutdown():
    """Gracefully shut down the application"""
    print("Shutting down...")
    
    try:
        # Set a timeout for the shutdown process
        if quoter:
            try:
                # Wait for quoter shutdown with a timeout
                print("Shutting down quoter...")
                shutdown_task = asyncio.create_task(quoter.shutdown())
                try:
                    await asyncio.wait_for(shutdown_task, timeout=10.0)
                    print("Quoter shutdown completed successfully")
                except asyncio.TimeoutError:
                    print("Quoter shutdown timed out after 10 seconds")
            except Exception as e:
                print(f"Error during quoter shutdown: {str(e)}")
        
        # Final cleanup with timeout
        try:
            # Wait for logger factory shutdown with a timeout
            print("Shutting down loggers...")
            logger_shutdown_task = asyncio.create_task(LoggerFactory.shutdown())
            try:
                await asyncio.wait_for(logger_shutdown_task, timeout=5.0)
                print("Logger shutdown completed successfully")
            except asyncio.TimeoutError:
                print("Logger shutdown timed out after 5 seconds")
        except Exception as e:
            print(f"Error during logger shutdown: {str(e)}")
            
    except Exception as e:
        print(f"Error during application shutdown: {str(e)}")
    
    print("Shutdown complete.")

def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown"""
    
    # Define the signal handler
    def handle_signal(sig, frame):
        print(f"Received signal {sig}, initiating shutdown...")
        # Set the shutdown event to trigger graceful shutdown
        shutdown_event.set()
        
    # Register the signal handlers
    signal.signal(signal.SIGINT, handle_signal)  # Ctrl+C
    signal.signal(signal.SIGTERM, handle_signal)  # Termination signal

async def run_quoter(test_mode=False, config_override=None):
    """Run the Avellaneda quoter with integrated volume candle prediction and dynamic grid updates"""
    global quoter
    
    try:
        # Initialize logging
        logger = LoggerFactory.configure_component_logger("start_quoter", "quoter.log")
        logger.info("Initializing Avellaneda quoter with real-time volume candle predictions")
        
        # Ensure log directories exist
        create_log_directories()
        
        # Display configuration information
        logger.info(f"Market: {MARKET_CONFIG['underlying']} on {MARKET_CONFIG['network']}")
        logger.info(f"Quote levels: {TRADING_CONFIG['quoting']['levels']}")
        logger.info(f"Grid spacing: {TRADING_CONFIG['order']['bid_step']} ticks")
        logger.info(f"Volume candle threshold: {TRADING_CONFIG['volume_candle']['threshold']} BTC")
        
        # Apply any configuration overrides
        if config_override:
            for key, value in config_override.items():
                if key in TRADING_CONFIG:
                    for subkey, subvalue in value.items():
                        if subkey in TRADING_CONFIG[key]:
                            logger.info(f"Overriding config: {key}.{subkey} = {subvalue}")
                            TRADING_CONFIG[key][subkey] = subvalue
        
        # Initialize Thalex client - always use the configured network
        network = MARKET_CONFIG["network"]
        logger.info(f"Connecting to Thalex {network} network")
        
        # Create real Thalex client
        try:
            logger.info("Creating Thalex client...")
            thalex_client = th.Thalex(network=network)
            logger.info("Thalex client created successfully")
        except Exception as e:
            logger.error(f"Error creating Thalex client: {str(e)}")
            print(f"Failed to create Thalex client: {str(e)}")
            return
        
        # Initialize the quoter
        try:
            logger.info("Creating Avellaneda quoter with volume candle predictions")
            quoter = AvellanedaQuoter(thalex_client)
            logger.info("Avellaneda quoter instance created")
        except Exception as e:
            logger.error(f"Error creating quoter: {str(e)}", exc_info=True)
            print(f"Failed to create quoter: {str(e)}")
            return
        
        # Start the quoter with explicit timeout
        try:
            logger.info("Starting quoter...")
            # Add timeout for the quoter start method
            start_task = asyncio.create_task(quoter.start())
            try:
                # Wait for 60 seconds max for startup
                success = await asyncio.wait_for(start_task, timeout=60.0)
                if not success:
                    logger.error("Quoter returned False from start method")
                    print("Quoter failed to start - check logs for details")
                    return
            except asyncio.TimeoutError:
                logger.error("Quoter start method timed out after 60 seconds")
                print("Quoter startup timed out after 60 seconds")
                return
        except Exception as e:
            logger.error(f"Error starting quoter: {str(e)}", exc_info=True)
            print(f"Failed to start quoter: {str(e)}")
            return
            
        logger.info("Quoter started successfully - running indefinitely")
        print("Quoter started successfully - running indefinitely (press Ctrl+C to stop)")
        
        # Keep the main task running until shutdown event is triggered
        while not shutdown_event.is_set():
            try:
                # Wait for the shutdown event or timeout after 60 seconds
                await asyncio.wait_for(shutdown_event.wait(), timeout=60)
            except asyncio.TimeoutError:
                # Timeout occurred, log a heartbeat and continue
                logger.info("Quoter running... (heartbeat)")
                # Verify that the quoter is still functioning
                if quoter and hasattr(quoter, 'thalex') and hasattr(quoter.thalex, 'connected'):
                    if not quoter.thalex.connected():
                        logger.warning("WebSocket disconnected - attempting reconnect")
                        try:
                            # Try to reconnect
                            await quoter.thalex.connect()
                            logger.info("WebSocket reconnected successfully")
                        except Exception as conn_err:
                            logger.error(f"Failed to reconnect: {str(conn_err)}")
                continue
                
        # If we reach here, shutdown was triggered
        logger.info("Shutdown event detected, initiating clean shutdown...")
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}", exc_info=True)
        print(f"Unexpected error: {str(e)}")
    finally:
        # Cleanup
        if quoter:
            logger.info("Shutting down quoter...")
            try:
                await quoter.shutdown()
                logger.info("Quoter shutdown complete")
            except Exception as e:
                logger.error(f"Error during quoter shutdown: {str(e)}")
                print(f"Error during quoter shutdown: {str(e)}")

async def async_main(test_mode=False, config_override=None):
    """Async main function to run the quoter with proper shutdown handling"""
    try:
        # Set up signal handlers
        setup_signal_handlers()
        
        # Run quoter until shutdown
        await run_quoter(test_mode, config_override)
    finally:
        # Always ensure clean shutdown
        await shutdown()

def main():
    """Parse arguments and run the quoter"""
    parser = argparse.ArgumentParser(description="Thalex Avellaneda Market Maker with Volume Candle Predictions")
    parser.add_argument("--test", action="store_true", help="Run on the test network")
    parser.add_argument("--gamma", type=float, help="Override gamma (risk aversion) parameter")
    parser.add_argument("--kappa", type=float, help="Override kappa (market depth) parameter")
    parser.add_argument("--levels", type=int, help="Override number of quote levels")
    parser.add_argument("--spacing", type=int, help="Override grid spacing in ticks")
    parser.add_argument("--vol-threshold", type=float, help="Override volume candle threshold")
    
    args = parser.parse_args()
    
    # Build configuration overrides
    config_override = {}
    if args.gamma:
        config_override.setdefault("avellaneda", {})["gamma"] = args.gamma
    if args.kappa:
        config_override.setdefault("avellaneda", {})["kappa"] = args.kappa
    if args.levels:
        config_override.setdefault("quoting", {})["levels"] = args.levels
    if args.spacing:
        config_override.setdefault("order", {})["bid_step"] = args.spacing
        config_override.setdefault("order", {})["ask_step"] = args.spacing
    if args.vol_threshold:
        config_override.setdefault("volume_candle", {})["threshold"] = args.vol_threshold
    
    # Run with modern asyncio API
    try:
        # Set up signal handlers before running
        asyncio.run(async_main(test_mode=args.test, config_override=config_override))
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        print("Main loop exited")

if __name__ == "__main__":
    # Print startup banner
    print("="*80)
    print(f"Thalex Avellaneda Market Maker - Starting at {datetime.now()}")
    print(f"Version: 2.2.0 (High Performance)")
    print("="*80)
    
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1) 