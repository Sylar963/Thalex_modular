import asyncio
import logging
import time
from typing import List, Optional
from collections import deque
import numpy as np

class AsyncLogger:
    """
    Asynchronous logger with ring buffer for high-performance logging.
    Batches log messages and writes them asynchronously to avoid I/O blocking.
    """
    def __init__(
        self,
        name: str,
        buffer_size: int = 10000,
        flush_interval: float = 0.1,  # 100ms
        flush_threshold: int = 1000,
        log_file: Optional[str] = None,
        level: int = logging.INFO
    ):
        self.name = name
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.flush_threshold = min(flush_threshold, buffer_size)
        
        # Initialize standard Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Add handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Always add stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        
        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Initialize ring buffer for log messages
        self.buffer = deque(maxlen=buffer_size)
        
        # Async control
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()
        
        # Performance metrics
        self.message_count = 0
        self.last_flush_time = time.time()
        self.flush_count = 0
        
    async def start(self):
        """Start the async logging task"""
        if not self._running:
            self._running = True
            self._flush_task = asyncio.create_task(self._flush_loop())
            
    async def stop(self):
        """Stop the async logging task and flush remaining messages with timeout protection"""
        if self._running:
            try:
                # Set running flag to false to stop the flush loop
                self._running = False
                
                # Cancel the flush task if it exists
                if self._flush_task and not self._flush_task.done():
                    self._flush_task.cancel()
                    try:
                        # Wait for the task to be cancelled with a timeout
                        await asyncio.wait_for(asyncio.shield(self._flush_task), timeout=1.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        # This is expected when we cancel the task or it times out
                        pass
                    except Exception as e:
                        self.logger.error(f"Error cancelling flush task: {str(e)}")
                
                # Try to flush the remaining messages with a timeout
                try:
                    await asyncio.wait_for(self._flush_messages(), timeout=1.0)
                except asyncio.TimeoutError:
                    self.logger.warning(f"Final message flush timed out for {self.name}")
                except Exception as e:
                    self.logger.error(f"Error during final message flush: {str(e)}")
            except Exception as e:
                self.logger.error(f"Error during logger shutdown: {str(e)}")
                
            # Add a final direct log message to indicate shutdown
            try:
                self.logger.info(f"AsyncLogger {self.name} stopped")
            except:
                # If the logger is unavailable, just continue
                pass
            
    def log(self, level: int, message: str, *args, **kwargs):
        """Add a log message to the buffer"""
        try:
            # Create log record
            record = {
                'time': time.time(),
                'level': level,
                'message': message,
                'args': args,
                'kwargs': kwargs
            }
            
            # Add to buffer
            self.buffer.append(record)
            self.message_count += 1
            
            # If buffer is getting full, trigger immediate flush
            if len(self.buffer) >= self.flush_threshold:
                asyncio.create_task(self._flush_messages())
                
        except Exception as e:
            # Fallback to immediate logging if buffering fails
            self.logger.error(f"Error buffering log message: {str(e)}")
            self.logger.log(level, message, *args, **kwargs)
            
    async def _flush_loop(self):
        """Main loop for flushing messages periodically"""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_messages()
            except Exception as e:
                self.logger.error(f"Error in flush loop: {str(e)}")
                
    async def _flush_messages(self):
        """Flush buffered messages to the logger"""
        async with self._lock:
            try:
                # Get all messages from buffer
                messages = list(self.buffer)
                self.buffer.clear()
                
                # Process messages
                for record in messages:
                    self.logger.log(
                        record['level'],
                        record['message'],
                        *record['args'],
                        **record['kwargs']
                    )
                    
                # Update metrics
                now = time.time()
                self.flush_count += 1
                messages_per_second = len(messages) / (now - self.last_flush_time) if messages else 0
                self.last_flush_time = now
                
                # Log performance metrics occasionally
                if self.flush_count % 100 == 0 and messages:
                    self.logger.debug(
                        f"Logging metrics - Messages/sec: {messages_per_second:.1f}, "
                        f"Total messages: {self.message_count}, "
                        f"Flush count: {self.flush_count}"
                    )
                    
            except Exception as e:
                self.logger.error(f"Error flushing messages: {str(e)}")
                
    def debug(self, message: str, *args, **kwargs):
        """Log a debug message"""
        self.log(logging.DEBUG, message, *args, **kwargs)
        
    def info(self, message: str, *args, **kwargs):
        """Log an info message"""
        self.log(logging.INFO, message, *args, **kwargs)
        
    def warning(self, message: str, *args, **kwargs):
        """Log a warning message"""
        self.log(logging.WARNING, message, *args, **kwargs)
        
    def error(self, message: str, *args, **kwargs):
        """Log an error message"""
        # Error messages bypass the buffer for immediate logging
        self.logger.error(message, *args, **kwargs)
        
    def critical(self, message: str, *args, **kwargs):
        """Log a critical message"""
        # Critical messages bypass the buffer for immediate logging
        self.logger.critical(message, *args, **kwargs)
        
    def exception(self, message: str, *args, exc_info=True, **kwargs):
        """Log an exception with traceback"""
        # Exception messages bypass the buffer for immediate logging
        self.logger.exception(message, *args, exc_info=exc_info, **kwargs) 