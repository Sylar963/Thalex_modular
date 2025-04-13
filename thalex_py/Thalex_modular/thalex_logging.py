import logging
import os
import sys
from datetime import datetime

class LoggerFactory:
    """Factory for creating and configuring loggers"""
    
    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DEFAULT_LEVEL = logging.INFO
    
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
                # Create logs directory if it doesn't exist
                log_dir = 'logs'
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                    
                # Full path to log file
                log_path = log_file if os.path.isabs(log_file) else os.path.join(log_dir, log_file)
                
                # Create file handler
                file_handler = logging.FileHandler(log_path)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
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
            log_file="thalex_bot.log"
        )
        
    @staticmethod
    def get_performance_logger():
        """Get the performance logger"""
        return LoggerFactory.configure_component_logger(
            "performance",
            log_file="performance.log",
            log_level=logging.INFO
        ) 