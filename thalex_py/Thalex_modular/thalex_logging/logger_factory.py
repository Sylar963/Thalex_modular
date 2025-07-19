import asyncio
import os
from typing import Dict, Optional
from .async_logger import AsyncLogger

class LoggerFactory:
    """
    Factory class to manage AsyncLogger instances.
    Ensures single instance per component and provides easy access to loggers.
    """
    _instances: Dict[str, AsyncLogger] = {}
    _started = False
    
    # Centralized logs configuration
    LOGS_BASE_DIR = 'logs'
    LOGS_STRUCTURE = {
        # Component type to subdirectory mapping
        'market_maker': 'market',
        'order_manager': 'orders',
        'risk_manager': 'risk',
        
        'performance': 'performance',
        'exchange': 'exchange',
        'position_tracker': 'positions',
        'orderbook': 'market',
        'volume_candle_buffer': 'market'
        # Default for other components will be the main logs directory
    }
    
    @classmethod
    async def initialize(cls):
        """Initialize all loggers"""
        if not cls._started:
            for logger in cls._instances.values():
                await logger.start()
            cls._started = True
    
    @classmethod
    async def shutdown(cls):
        """Shutdown all loggers with timeout protection"""
        if cls._started:
            try:
                # Create a list of tasks to shutdown all loggers with timeout
                shutdown_tasks = []
                
                # Create a task for each logger
                for name, logger in cls._instances.items():
                    try:
                        # Add the stop task with a timeout
                        shutdown_tasks.append(
                            asyncio.create_task(cls._stop_logger_with_timeout(name, logger, timeout=2.0))
                        )
                    except Exception as e:
                        print(f"Error creating shutdown task for logger {name}: {str(e)}")
                
                # Wait for all loggers to complete or timeout
                if shutdown_tasks:
                    await asyncio.gather(*shutdown_tasks, return_exceptions=True)
                
            except Exception as e:
                print(f"Error during logger shutdown: {str(e)}")
            finally:
                # Mark as stopped regardless of errors
                cls._started = False
                
    @classmethod
    async def _stop_logger_with_timeout(cls, name, logger, timeout=2.0):
        """Attempt to stop a logger with a timeout to prevent hanging"""
        try:
            # Create a task for the stop operation
            stop_task = asyncio.create_task(logger.stop())
            
            # Wait for the task to complete with a timeout
            try:
                await asyncio.wait_for(stop_task, timeout=timeout)
                print(f"Logger {name} stopped successfully")
            except asyncio.TimeoutError:
                print(f"Logger {name} stop operation timed out after {timeout}s")
        except Exception as e:
            print(f"Error stopping logger {name}: {str(e)}")
    
    @classmethod
    def get_log_file_path(cls, component_name: str, log_file: str) -> str:
        """
        Get the full path for a log file based on component type.
        Organizes logs into subdirectories by component type.
        
        Args:
            component_name: Name of the component
            log_file: Log file name
            
        Returns:
            Full path to the log file
        """
        # If it's already an absolute path, return it directly
        if os.path.isabs(log_file):
            return log_file
            
        # Determine subdirectory based on component type
        subdirectory = None
        for key, value in cls.LOGS_STRUCTURE.items():
            if key in component_name:
                subdirectory = value
                break
                
        # Construct the log path
        if subdirectory:
            log_dir = os.path.join(cls.LOGS_BASE_DIR, subdirectory)
        else:
            log_dir = cls.LOGS_BASE_DIR
            
        # Ensure the directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Return the full path
        return os.path.join(log_dir, log_file)
    
    @classmethod
    def get_logger(
        cls,
        name: str,
        buffer_size: int = 10000,
        flush_interval: float = 0.1,
        flush_threshold: int = 1000,
        log_file: Optional[str] = None
    ) -> AsyncLogger:
        """
        Get or create an AsyncLogger instance.
        
        Args:
            name: Logger name/component identifier
            buffer_size: Size of the ring buffer
            flush_interval: How often to flush (seconds)
            flush_threshold: Buffer threshold to trigger flush
            log_file: Optional log file path
            
        Returns:
            AsyncLogger instance
        """
        if name not in cls._instances:
            # Process log file path if provided
            processed_log_file = None
            if log_file:
                processed_log_file = cls.get_log_file_path(name, log_file)
            
            # Create new logger instance
            logger = AsyncLogger(
                name=name,
                buffer_size=buffer_size,
                flush_interval=flush_interval,
                flush_threshold=flush_threshold,
                log_file=processed_log_file
            )
            cls._instances[name] = logger
            
            # Start logger if factory is already initialized
            if cls._started:
                asyncio.create_task(logger.start())
                
        return cls._instances[name]
    
    @classmethod
    def configure_component_logger(
        cls,
        component_name: str,
        log_file: Optional[str] = None,
        high_frequency: bool = False
    ) -> AsyncLogger:
        """
        Configure a logger with appropriate settings for different component types.
        
        Args:
            component_name: Name of the component
            log_file: Optional log file path
            high_frequency: Whether this is a high-frequency component
            
        Returns:
            Configured AsyncLogger instance
        """
        if high_frequency:
            # High-frequency components need larger buffers and faster flushing
            return cls.get_logger(
                name=component_name,
                buffer_size=50000,
                flush_interval=0.05,  # 50ms
                flush_threshold=5000,
                log_file=log_file
            )
        else:
            # Standard components use default settings
            return cls.get_logger(
                name=component_name,
                log_file=log_file
            )
            
    @classmethod
    def get_market_maker_logger(cls) -> AsyncLogger:
        """Get logger configured for market maker component"""
        return cls.configure_component_logger(
            "market_maker",
            log_file="market_maker.log",
            high_frequency=True
        )
        
    @classmethod
    def get_order_manager_logger(cls) -> AsyncLogger:
        """Get logger configured for order manager component"""
        return cls.configure_component_logger(
            "order_manager",
            log_file="order_manager.log",
            high_frequency=True
        )
        
    @classmethod
    def get_risk_manager_logger(cls) -> AsyncLogger:
        """Get logger configured for risk manager component"""
        return cls.configure_component_logger(
            "risk_manager",
            log_file="risk_manager.log",
            high_frequency=False
        ) 