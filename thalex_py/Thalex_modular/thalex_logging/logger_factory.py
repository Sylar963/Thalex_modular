import asyncio
from typing import Dict, Optional
from .async_logger import AsyncLogger

class LoggerFactory:
    """
    Factory class to manage AsyncLogger instances.
    Ensures single instance per component and provides easy access to loggers.
    """
    _instances: Dict[str, AsyncLogger] = {}
    _started = False
    
    @classmethod
    async def initialize(cls):
        """Initialize all loggers"""
        if not cls._started:
            for logger in cls._instances.values():
                await logger.start()
            cls._started = True
    
    @classmethod
    async def shutdown(cls):
        """Shutdown all loggers"""
        if cls._started:
            for logger in cls._instances.values():
                await logger.stop()
            cls._started = False
    
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
            # Create new logger instance
            logger = AsyncLogger(
                name=name,
                buffer_size=buffer_size,
                flush_interval=flush_interval,
                flush_threshold=flush_threshold,
                log_file=log_file
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