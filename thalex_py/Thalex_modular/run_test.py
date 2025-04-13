import asyncio
import time
import signal
import sys

async def run_test():
    # Import the main modules here to avoid circular import issues
    from thalex import Thalex
    from thalex_py.Thalex_modular.config.market_config import BOT_CONFIG
    from thalex_py.Thalex_modular.avellaneda_quoter import AvellanedaQuoter
    
    # Initialize Thalex client
    thalex = Thalex(network=BOT_CONFIG["market"]["network"])
    
    # Create quoter
    quoter = AvellanedaQuoter(thalex)
    
    # Set up signal handler to stop after timeout
    def signal_handler(sig, frame):
        print("\nTest completed - 120 seconds elapsed")
        asyncio.create_task(quoter.shutdown())
        
    # Register signal handler for clean shutdown
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(120)  # Set alarm for 120 seconds
    
    print("Starting Avellaneda Quoter - Test run for 120 seconds")
    
    try:
        # Start the quoter
        await quoter.start()
    except Exception as e:
        print(f"Error in test: {str(e)}")
    finally:
        print("Test finished")
        # Ensure shutdown
        await quoter.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Fatal error: {str(e)}") 