import logging
import os
import sys
from datetime import datetime

class LoggerFactory:
    """Factory for creating and configuring loggers"""
    
    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DEFAULT_LEVEL = logging.ERROR
    
    # Centralized logs configuration
    LOGS_BASE_DIR = 'logs'
    LOGS_STRUCTURE = {
        # Component type to subdirectory mapping
        'market_maker': 'market',
        'avellaneda': 'market',  # Also route avellaneda to market directory
        'order_manager': 'orders',
        'risk_manager': 'risk',
        'performance': 'performance',
        'exchange': 'exchange',
        'position_tracker': 'positions',
        'orderbook': 'market',
        'volume_candle_buffer': 'market',
        'thalex_client': 'exchange',
        'thalex_bot': 'market'
        # Default for other components will be the main logs directory
    }
    
    @staticmethod
    def get_log_file_path(component_name: str, log_file: str) -> str:
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
        for key, value in LoggerFactory.LOGS_STRUCTURE.items():
            if key in component_name:
                subdirectory = value
                break
                
        # Construct the log path
        if subdirectory:
            log_dir = os.path.join(LoggerFactory.LOGS_BASE_DIR, subdirectory)
        else:
            log_dir = LoggerFactory.LOGS_BASE_DIR
            
        # Ensure the directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Return the full path
        return os.path.join(log_dir, log_file)
    
    @staticmethod
    def configure_component_logger(
        component_name: str, 
        log_file: str = None,
        log_level: int = None,
        high_frequency: bool = False
    ) -> logging.Logger:
        """
        Configure a logger for a specific component
        
        Args:
            component_name: Name of the component
            log_file: Optional log file name
            log_level: Log level (defaults to INFO)
            high_frequency: Whether this logger will be used for high-frequency logging
            
        Returns:
            Configured logger
        """
        # Set default log level if not provided
        if log_level is None:
            # Enable debug logging for market maker and volume candle components
            if component_name in ["avellaneda_market_maker", "volume_candle_buffer"]:
                log_level = logging.DEBUG
            else:
                log_level = LoggerFactory.DEFAULT_LEVEL
                
        # Create logger
        logger = logging.getLogger(component_name)
        logger.setLevel(log_level)
        
        # Clear any existing handlers
        logger.handlers = []
        
        # Create formatter
        formatter = logging.Formatter(LoggerFactory.DEFAULT_FORMAT)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            try:
                # Get the full path for the log file
                log_path = LoggerFactory.get_log_file_path(component_name, log_file)
                
                # Create file handler
                file_handler = logging.FileHandler(log_path)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
                # Log the file location for debugging
                logger.debug(f"Logging to file: {log_path}")
            except Exception as e:
                # Fall back to just console logging
                logger.warning(f"Could not set up file logging: {str(e)}")
        
        # Set high-frequency optimizations if requested
        if high_frequency:
            for handler in logger.handlers:
                # Buffer output to avoid excessive I/O
                if isinstance(handler, logging.FileHandler):
                    handler.setLevel(logging.INFO)  # Reduce file logging for high-frequency components
                    
                # Increase buffer size for StreamHandlers
                if hasattr(handler, 'setBufferSize'):
                    handler.setBufferSize(4096)
                    
        return logger
        
    @staticmethod
    def get_bot_logger():
        """Get the main bot logger"""
        return LoggerFactory.configure_component_logger(
            "thalex_bot",
            log_file="bot.log"
        )
        
    @staticmethod
    def get_performance_logger():
        """Get the performance logger"""
        return LoggerFactory.configure_component_logger(
            "performance",
            log_file="performance.log",
            log_level=logging.INFO
        )
        
    @staticmethod
    def shutdown():
        """Placeholder for compatibility with async logger factory"""
        # This synchronous logger doesn't need explicit shutdown
        logging.shutdown() 