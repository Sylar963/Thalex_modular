import numpy as np
import time
import threading
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import logging
from collections import deque
import os
import csv
from ..thalex_logging import LoggerFactory

TRADE_ID_CACHE_SIZE = 10000 # Define cache size
SIGNAL_EVAL_LOG_MAX_SIZE = 1000 # Max in-memory signal events before older ones (if not written) are pushed out
LOGS_DIR = "/home/aladhimarkets/Thalex_SimpleQuouter/logs" # User-provided log directory

class VolumeCandle:
    """Simple container for volume-based candle data"""
    def __init__(self):
        self.open_price = 0.0
        self.high_price = 0.0
        self.low_price = float('inf')
        self.close_price = 0.0
        self.volume = 0.0
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.volume_delta = 0.0  # buy_volume - sell_volume
        self.delta_ratio = 0.0   # normalized delta (-1 to 1)
        self.trade_count = 0
        self.start_time = 0
        self.end_time = 0
        self.is_complete = False

    def update(self, price: float, volume: float, is_buy: bool, timestamp: int) -> None:
        """Update candle with new trade data"""
        # Update prices
        if self.trade_count == 0:
            self.open_price = price
            self.high_price = price
            self.low_price = price
            self.start_time = timestamp
        else:
            self.high_price = max(self.high_price, price)
            self.low_price = min(self.low_price, price)
        
        self.close_price = price
        self.end_time = timestamp
        
        # Update volumes
        self.volume += volume
        if is_buy:
            self.buy_volume += volume
        else:
            self.sell_volume += volume
        
        # Update volume delta
        self.volume_delta = self.buy_volume - self.sell_volume
        if self.volume > 0:
            self.delta_ratio = self.volume_delta / self.volume
        
        self.trade_count += 1

    def __repr__(self) -> str:
        return (f"VolumeCandle(O:{self.open_price:.2f}, H:{self.high_price:.2f}, "
                f"L:{self.low_price:.2f}, C:{self.close_price:.2f}, V:{self.volume:.4f}, "
                f"Δ:{self.volume_delta:.4f}, Ratio:{self.delta_ratio:.2f})")


class VolumeBasedCandleBuffer:
    """Buffer for volume-based candles with predictive indicators"""
    
    # --- Signal Calculation Constants ---
    MIN_CANDLES_FOR_SIGNALS = 3  # Reduced for faster signal generation
    MOMENTUM_PRICE_FACTOR = 100.0  # Increased sensitivity for small price moves
    MOMENTUM_DELTA_SENSITIVITY_FACTOR = 5.0  # Increased delta sensitivity  
    REVERSAL_PRICE_CHANGE_FACTOR = 50.0  # Increased reversal sensitivity
    REVERSAL_DELTA_CHANGE_FACTOR = 10.0  # Increased delta change sensitivity
    VOLATILITY_PRICE_RANGE_FACTOR = 50.0  # Increased volatility sensitivity
    EXHAUSTION_DELTA_TREND_THRESHOLD = 0.1  # Lowered threshold for exhaustion
    EXHAUSTION_STRENGTH_FACTOR = 5.0  # Increased exhaustion strength
    SIGNIFICANT_SIGNAL_CHANGE_THRESHOLD = 0.05  # Lowered for more logging
    SIGNAL_LOGGING_FREQUENCY = 3  # More frequent logging

    # --- Parameter Prediction Constants ---
    GAMMA_ADJ_VOLATILITY_FACTOR = 0.4
    GAMMA_ADJ_REVERSAL_FACTOR = 0.3
    KAPPA_ADJ_VOLATILITY_FACTOR = -0.3 # Negative as it reduces market depth
    RESERVATION_OFFSET_MOMENTUM_THRESHOLD = 0.2
    RESERVATION_OFFSET_EXHAUSTION_THRESHOLD = 0.5
    RESERVATION_OFFSET_MOMENTUM_FACTOR = 0.0003
    RESERVATION_OFFSET_REVERSAL_THRESHOLD = 0.6
    RESERVATION_OFFSET_REVERSAL_FACTOR = 0.0005 # Applied negatively to price_direction
    TREND_DIRECTION_MOMENTUM_THRESHOLD = 0.3
    VOL_ADJ_VOLATILITY_FACTOR = 0.4
    
    def __init__(
        self,
        volume_threshold: float = 1.0,     # Volume required to complete a candle
        max_candles: int = 100,            # Maximum candles to store
        max_time_seconds: int = 300,       # Maximum time before forcing candle close
        ema_periods: Dict[str, int] = None, # EMA periods for tracking
        exchange_client = None,            # Optional exchange client for API access
        instrument: str = None,            # Instrument to track
        use_exchange_data: bool = False,   # Whether to use exchange data
        fetch_interval_seconds: int = 60,  # How often to fetch exchange data
        lookback_hours: int = 1,           # How far back to look for historical data
        evaluation_horizons: Optional[List[int]] = None # Added for signal evaluation
    ):
        # Configure logging
        self.logger = LoggerFactory.configure_component_logger(
            "volume_candle_buffer",
            log_file="volume_candles.log",
            high_frequency=True
        )
        self.logger.info(f"Initializing volume candle buffer: threshold={volume_threshold}, "
                         f"max_candles={max_candles}, max_time={max_time_seconds}s")
        
        # Configuration
        self.volume_threshold = volume_threshold
        self.max_time_seconds = max_time_seconds
        
        # Current candle
        self.current_candle = VolumeCandle()
        
        # Completed candles buffer
        self.candles = deque(maxlen=max_candles)
        self.max_candles = max_candles
        
        # Technical indicators
        self.ema_periods = ema_periods or {"fast": 8, "med": 21, "slow": 55}
        self.ema_values = {name: 0.0 for name in self.ema_periods.keys()}
        
        # Delta indicators
        self.delta_ema = {name: 0.0 for name in self.ema_periods.keys()}
        self.cumulative_delta = 0.0
        
        # Predictive signals
        self.signals = {
            "momentum": 0.0,     # -1 to 1, direction and strength
            "reversal": 0.0,     # 0 to 1, probability of reversal
            "volatility": 0.0,   # 0 to 1, expected volatility increase
            "exhaustion": 0.0    # 0 to 1, sign of buying/selling exhaustion
        }
        
        # Performance tracking
        self._last_update_time = 0
        self._prediction_accuracy = []
        
        # Statistics
        self.updates_count = 0
        self.candles_completed = 0
        self.predictions_made = 0
        self.last_signal_strength = 0.0

        # Exchange API integration
        self.exchange_client = exchange_client
        self.instrument = instrument
        self.use_exchange_data = use_exchange_data
        self.fetch_interval_seconds = fetch_interval_seconds
        self.lookback_hours = lookback_hours
        
        # Thread-related
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._fetch_thread = None
        self._stop_event = threading.Event()
        self._processed_trade_ids_set = set()
        self._processed_trade_ids_queue = deque(maxlen=TRADE_ID_CACHE_SIZE)
        
        # --- Signal Evaluation Setup ---
        self.evaluation_horizons = evaluation_horizons if evaluation_horizons else [1, 3, 5, 10] # Default horizons
        self.signal_evaluation_log = deque(maxlen=SIGNAL_EVAL_LOG_MAX_SIZE)
        self.signal_eval_log_file_path = os.path.join(LOGS_DIR, "signal_evaluation.csv")
        self._setup_signal_eval_logger()
        self.signal_eval_log_header_written = False
        # Attempt to write header if file is new or empty
        self._write_signal_eval_header_if_needed()
        
        # Start data fetching thread if needed
        if self.use_exchange_data and self.exchange_client and self.instrument:
            self.logger.info(f"Exchange data integration enabled for {self.instrument}")
            self._initialize_exchange_data()
            self._start_fetch_thread()
        else:
            if use_exchange_data:
                self.logger.warning("Exchange data requested but missing required parameters "
                                   f"(exchange_client: {exchange_client is not None}, "
                                   f"instrument: {instrument})")

    def _setup_signal_eval_logger(self) -> None:
        """Sets up a dedicated logger for signal evaluation CSV data."""
        self.signal_eval_logger = logging.getLogger(__name__ + ".SignalEval")
        # Prevent propagation to the root logger or other handlers like the main component logger
        self.signal_eval_logger.propagate = False 
        
        # Ensure logs directory exists
        if not os.path.exists(LOGS_DIR):
            try:
                os.makedirs(LOGS_DIR)
                self.logger.info(f"Created logs directory: {LOGS_DIR}")
            except OSError as e:
                self.logger.error(f"Failed to create logs directory {LOGS_DIR}: {e}")
                # If dir creation fails, logger won't be able to write. 
                # Could disable this logger or let it fail silently on write attempt.
                return

        # Add file handler if not already present (e.g. during re-init or multiple instances)
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == self.signal_eval_log_file_path for h in self.signal_eval_logger.handlers):
            fh = logging.FileHandler(self.signal_eval_log_file_path, mode='a') # Append mode
            # No formatter needed as we will write raw CSV lines
            self.signal_eval_logger.addHandler(fh)
            self.signal_eval_logger.setLevel(logging.INFO) # Or whatever level is appropriate

    def _write_signal_eval_header_if_needed(self) -> None:
        """Writes the CSV header if the file is new or empty."""
        try:
            file_exists_and_not_empty = os.path.exists(self.signal_eval_log_file_path) and os.path.getsize(self.signal_eval_log_file_path) > 0
            if not file_exists_and_not_empty:
                # Ensure consistent order of signal columns, matching data logging
                signal_keys_sorted = sorted(self.signals.keys()) 
                header = ['initial_timestamp', 'entry_price'] \
                         + [f'signal_{s_name}' for s_name in signal_keys_sorted] \
                         + sum([[f'future_price_h{h}', f'pnl_h{h}'] for h in self.evaluation_horizons], []) \
                         + ['candles_completed_for_eval']
                
                with open(self.signal_eval_log_file_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                self.signal_eval_log_header_written = True # Mark as written for this session
                self.logger.info(f"Signal evaluation CSV header written to {self.signal_eval_log_file_path}")
        except Exception as e:
            self.logger.error(f"Failed to write signal evaluation CSV header: {e}")

    def update(self, price: float, volume: float, is_buy: bool, timestamp: int = None, trade_id: str = None) -> Optional[VolumeCandle]:
        """
        Update with new trade data. Returns completed candle if applicable.
        
        Args:
            price: Trade price
            volume: Trade volume
            is_buy: Whether it was a buy trade
            timestamp: Unix timestamp in milliseconds
            trade_id: Optional trade ID for deduplication
            
        Returns:
            Completed VolumeCandle if one was finished, None otherwise
        """
        with self._lock:
            # Check for duplicate trade if ID is provided
            if trade_id and trade_id in self._processed_trade_ids_set:
                self.logger.debug(f"Skipping duplicate trade {trade_id}")
                return None
                
            # Add to processed trades if ID provided
            if trade_id:
                # If trade_id is new and the queue is already full, 
                # the oldest item (at index 0 of the queue) will be evicted.
                # We must remove this soon-to-be-evicted item from the set.
                if trade_id not in self._processed_trade_ids_set and len(self._processed_trade_ids_queue) == TRADE_ID_CACHE_SIZE:
                    evicted_id = self._processed_trade_ids_queue[0] # This is the item that will be pushed out
                    self._processed_trade_ids_set.remove(evicted_id)
                
                # Add the new trade_id to the queue (and it's implicitly managed by maxlen)
                # Add it to the set for fast lookups.
                # If trade_id was already in the queue/set, this effectively does nothing for set, 
                # and deque doesn't change order for existing items on append.
                # If we want to move to end on re-occurrence, we'd need remove then append.
                # For simple deduplication, this is okay.
                self._processed_trade_ids_queue.append(trade_id) # Deque handles its own eviction if full
                self._processed_trade_ids_set.add(trade_id) # Add/update in set
            
            # Set timestamp if not provided
            if timestamp is None:
                timestamp = int(time.time() * 1000)
            
            self.updates_count += 1
            
            # Debug log every 100th update to avoid excessive logging
            if self.updates_count % 100 == 0:
                self.logger.debug(f"Volume candle update #{self.updates_count}: price={price:.2f}, "
                              f"volume={volume:.6f}, is_buy={is_buy}, current_candle_volume={self.current_candle.volume:.6f}")
                
            # Update current candle
            self.current_candle.update(price, volume, is_buy, timestamp)
            
            # Check if candle should be completed
            completed_candle = None
            if (self.current_candle.volume >= self.volume_threshold or 
                    (timestamp - self.current_candle.start_time) >= self.max_time_seconds * 1000):
                
                # Mark as complete
                self.current_candle.is_complete = True
                completed_candle = self.current_candle
                self.candles_completed += 1
                
                # Add to buffer
                self.candles.append(self.current_candle)
                
                # Update indicators
                self._update_indicators(completed_candle)
                
                # Calculate predictive signals
                self._calculate_signals()
                
                # Log candle completion with detailed info
                reason = "volume threshold reached" if completed_candle.volume >= self.volume_threshold else "time limit reached"
                duration_seconds = (completed_candle.end_time - completed_candle.start_time) / 1000
                
                self.logger.info(
                    f"Volume candle #{self.candles_completed} completed ({reason}): "
                    f"O:{completed_candle.open_price:.2f}, H:{completed_candle.high_price:.2f}, "
                    f"L:{completed_candle.low_price:.2f}, C:{completed_candle.close_price:.2f}, "
                    f"V:{completed_candle.volume:.6f}, Δ:{completed_candle.delta_ratio:.2f}, "
                    f"Trades:{completed_candle.trade_count}, Duration:{duration_seconds:.1f}s"
                )
                
                # Reset current candle
                self.current_candle = VolumeCandle()
                
                # --- Update and log signal evaluations based on the newly completed candle ---
                if completed_candle:
                    self._update_and_log_signal_evaluations(completed_candle)
                
            self._last_update_time = timestamp
            return completed_candle

    def _update_and_log_signal_evaluations(self, current_completed_candle: VolumeCandle) -> None:
        """
        Iterates through pending signal evaluations, updates them with new candle data,
        and logs them to CSV if all evaluation horizons are met.
        """
        # Iterate over a copy of the items if modifying deque, or manage indices carefully.
        # Since we append to right and potentially remove from left later (if we cap strictly by CSV write),
        # direct iteration should be fine for updating in place.
        
        items_to_log_and_remove_indices = []

        for i, event in enumerate(self.signal_evaluation_log):
            if not event['is_complete']:
                event['candles_passed'] += 1
                all_horizons_filled = True
                for h in self.evaluation_horizons:
                    if event['future_prices_at_horizon'][h] is None and event['candles_passed'] >= h:
                        # If candles_passed is exactly h, this is the first time we can log this horizon.
                        # If candles_passed > h (e.g. if a horizon was missed due to some issue or startup),
                        # we might log it here too, or decide to only log if candles_passed == h.
                        # For simplicity, let's assume current_completed_candle is the h-th candle after the signal.
                        # This requires that _update_and_log_signal_evaluations is called for *every* completed candle.
                        if event['candles_passed'] == h: # Ensure it's exactly the h-th candle
                             event['future_prices_at_horizon'][h] = current_completed_candle.close_price
                             # Basic P&L: (future_price - entry_price). 
                             # Directionality (buy/sell based on signal) will be handled in Phase B analysis.
                             event['pnl_at_horizon'][h] = current_completed_candle.close_price - event['entry_price']
                    
                    if event['future_prices_at_horizon'][h] is None:
                        all_horizons_filled = False # Still waiting for data for this horizon
                
                if all_horizons_filled:
                    event['is_complete'] = True
                    # Prepare data for CSV logging
                    row_data = [
                        event['initial_timestamp'],
                        event['entry_price']
                    ]
                    # Add signal values in a consistent order (sorted by key for example)
                    # This order must match the header from _write_signal_eval_header_if_needed
                    # self.signals.keys() in header writing implies a specific initial order.
                    # For safety, let's sort keys from self.signals (used in header) and event signals.
                    signal_keys_sorted = sorted(self.signals.keys()) # Matches header generation if signals keys are fixed
                    for s_key in signal_keys_sorted:
                        row_data.append(event['signals'].get(s_key, None)) # Use .get for safety
                    
                    for h in self.evaluation_horizons:
                        row_data.append(event['future_prices_at_horizon'][h])
                        row_data.append(event['pnl_at_horizon'][h])
                    row_data.append(event['candles_passed']) # Actually, this is max_horizon when complete.
                                                            # Let's log the actual max candles passed which is event['candles_passed']

                    try:
                        # Use a context manager for writing to ensure file is closed.
                        # The logger's FileHandler keeps the file open, so direct write is also an option.
                        # For simplicity with CSV, direct write via open() is fine here as it's one row.
                        with open(self.signal_eval_log_file_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(row_data)
                        # Mark for removal if we want to strictly use deque as a temporary buffer
                        # items_to_log_and_remove_indices.append(i) # Not removing for now, deque maxlen handles memory.
                    except Exception as e:
                        self.logger.error(f"Failed to write signal eval data to CSV: {e} for event {event}")
        
        # If we were to remove from deque after writing (not strictly needed due to maxlen):
        # for i in sorted(items_to_log_and_remove_indices, reverse=True):
        #     # This is complex with deque if removing from middle. Better to let maxlen handle it, 
        #     # or only remove from left if they are guaranteed to be processed in order.
        #     # Since `is_complete` flag is used, maxlen is the simplest memory management here.
        pass # End of method

    def _initialize_exchange_data(self) -> None:
        """Initialize with historical exchange data"""
        if not self.exchange_client or not self.instrument:
            return
            
        try:
            self.logger.info(f"Initializing with historical data for {self.instrument} from exchange")
            trades = self.fetch_exchange_trades(self.lookback_hours)
            
            if not trades:
                self.logger.warning("No historical trades found")
                return
                
            self.logger.info(f"Processing {len(trades)} historical trades")
            
            # Sort trades by timestamp to ensure correct order
            trades.sort(key=lambda t: t.get('timestamp', 0))
            
            # Process each trade
            for trade in trades:
                try:
                    # Extract trade data
                    parsed_trade = self._parse_trade_data(trade)
                    if parsed_trade:
                        price, volume, is_buy, timestamp, trade_id = parsed_trade
                        self.update(price, volume, is_buy, timestamp, trade_id)
                except Exception as e: # Should be caught by _parse_trade_data, but as a safeguard
                    self.logger.error(f"Error processing historical trade: {str(e)} for trade {trade}")
                    
            self.logger.info(f"Historical data initialization complete, processed {len(trades)} trades")
            
        except Exception as e:
            self.logger.error(f"Error initializing exchange data: {str(e)}")

    def _parse_trade_data(self, trade: Dict[str, Any]) -> Optional[Tuple[float, float, bool, int, str]]:
        """Helper method to parse raw trade data."""
        try:
            price = float(trade.get('price', 0))
            volume = float(trade.get('size', 0))
            is_buy = trade.get('side', '').lower() == 'buy'
            timestamp = int(trade.get('timestamp', 0))
            trade_id = str(trade.get('tradeId', ''))

            if price > 0 and volume > 0:
                return price, volume, is_buy, timestamp, trade_id
            else:
                self.logger.warning(f"Invalid trade data (price/volume <= 0): {trade}")
                return None
        except Exception as e:
            self.logger.error(f"Error parsing trade data: {trade}, Error: {str(e)}")
            return None

    def _start_fetch_thread(self) -> None:
        """Start background thread to fetch exchange data periodically"""
        if self._fetch_thread and self._fetch_thread.is_alive():
            self.logger.warning("Fetch thread already running")
            return
            
        self._stop_event.clear()
        self._fetch_thread = threading.Thread(
            target=self._fetch_data_loop,
            name="ExchangeDataFetcher",
            daemon=True
        )
        self._fetch_thread.start()
        self.logger.info(f"Started exchange data fetch thread (interval: {self.fetch_interval_seconds}s)")

    def _fetch_data_loop(self) -> None:
        """Background loop to fetch exchange data periodically"""
        while not self._stop_event.is_set():
            try:
                # Calculate a shorter interval for checking stop event
                check_interval = min(5, self.fetch_interval_seconds / 4)
                
                # Wait for the fetch interval, checking for stop event periodically
                elapsed = 0
                while elapsed < self.fetch_interval_seconds and not self._stop_event.is_set():
                    time.sleep(check_interval)
                    elapsed += check_interval
                
                # If stop event was set during wait, exit loop
                if self._stop_event.is_set():
                    break
                    
                # Fetch and process new trades
                self.logger.debug(f"Fetching new exchange trades for {self.instrument}")
                # Use a shorter lookback to avoid processing too many trades
                lookback_hours = min(0.5, self.lookback_hours)
                trades = self.fetch_exchange_trades(lookback_hours)
                
                if trades:
                    self.logger.info(f"Processing {len(trades)} new exchange trades")
                    
                    # Process each trade
                    with self._lock:
                        for trade in trades:
                            try:
                                # Extract trade data
                                parsed_trade = self._parse_trade_data(trade)
                                if parsed_trade:
                                    price, volume, is_buy, timestamp, trade_id = parsed_trade
                                    self.update(price, volume, is_buy, timestamp, trade_id)
                            except Exception as e: # Should be caught by _parse_trade_data, but as a safeguard
                                self.logger.error(f"Error processing exchange trade: {str(e)} for trade {trade}")
                else:
                    self.logger.debug("No new exchange trades found")
                
            except Exception as e:
                self.logger.error(f"Error in exchange data fetch loop: {str(e)}")
                # Sleep briefly before retrying to avoid tight error loops
                time.sleep(5)

    def fetch_exchange_trades(self, hours: int = 1) -> List[Dict]:
        """
        Fetch trades from Thalex exchange API
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of trade dictionaries
        """
        if not self.exchange_client:
            self.logger.warning("No exchange client provided, cannot fetch trades")
            return []
            
        try:
            # Calculate time window
            end_time = int(time.time() * 1000)
            start_time = end_time - (hours * 60 * 60 * 1000)
            
            # Fetch trades
            self.logger.info(f"Fetching trades for {self.instrument} from {start_time} to {end_time}")
            trades = self.exchange_client.get_trade_history(
                instrument=self.instrument,
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )
            
            self.logger.info(f"Fetched {len(trades)} trades from exchange")
            return trades
        except Exception as e:
            self.logger.error(f"Error fetching trades: {str(e)}")
            return []

    def stop(self) -> None:
        """Stop the background data fetch thread"""
        if self._fetch_thread and self._fetch_thread.is_alive():
            self.logger.info("Stopping exchange data fetch thread")
            self._stop_event.set()
            self._fetch_thread.join(timeout=10)
            if self._fetch_thread.is_alive():
                self.logger.warning("Exchange data fetch thread did not stop cleanly")
            else:
                self.logger.info("Exchange data fetch thread stopped")
        
    def _update_indicators(self, candle: VolumeCandle) -> None:
        """Update technical indicators with new candle"""
        # Update price EMAs
        for name, period in self.ema_periods.items():
            alpha = 2.0 / (period + 1)
            
            # Initialize if first value
            if self.ema_values[name] == 0.0:
                self.ema_values[name] = candle.close_price
                self.delta_ema[name] = candle.delta_ratio
            else:
                # Update EMAs
                self.ema_values[name] = candle.close_price * alpha + self.ema_values[name] * (1 - alpha)
                self.delta_ema[name] = candle.delta_ratio * alpha + self.delta_ema[name] * (1 - alpha)
        
        # Update cumulative delta
        self.cumulative_delta += candle.volume_delta
        
        # Log indicator updates
        self.logger.debug(
            f"Updated indicators: EMAs={', '.join([f'{k}:{v:.2f}' for k, v in self.ema_values.items()])}, "
            f"Delta_EMAs={', '.join([f'{k}:{v:.2f}' for k, v in self.delta_ema.items()])}, "
            f"Cumulative_Delta={self.cumulative_delta:.6f}"
        )
        
    def _calculate_signals(self) -> None:
        """Calculate predictive signals from candle data"""
        if len(self.candles) < self.MIN_CANDLES_FOR_SIGNALS:
            return
        
        self.predictions_made += 1
            
        # Get recent candles
        recent = list(self.candles)[-self.MIN_CANDLES_FOR_SIGNALS:]
        
        # Save old signals for logging change
        old_signals = self.signals.copy()
        
        # 1. Momentum Signal - combines price trend and volume delta
        price_momentum = (recent[-1].close_price / recent[0].open_price) - 1.0
        delta_momentum = np.mean([c.delta_ratio for c in recent])
        
        # Combine price and volume momentum (volume confirming price or diverging)
        self.signals["momentum"] = np.clip(
            price_momentum * self.MOMENTUM_PRICE_FACTOR * np.sign(delta_momentum) * min(1.0, abs(delta_momentum) * self.MOMENTUM_DELTA_SENSITIVITY_FACTOR),
            -1.0, 1.0
        )
        
        # 2. Reversal Signal - looks for divergence between price and volume delta
        price_direction = np.sign(recent[-1].close_price - recent[0].open_price)
        delta_direction = np.sign(self.delta_ema["fast"] - self.delta_ema["slow"])
        
        # Potential reversal if directions differ
        if price_direction != 0 and delta_direction != 0 and price_direction != delta_direction:
            # Strength of divergence
            price_change = abs(recent[-1].close_price / recent[0].open_price - 1.0)
            delta_change = abs(self.delta_ema["fast"] - self.delta_ema["slow"])
            
            self.signals["reversal"] = min(1.0, price_change * self.REVERSAL_PRICE_CHANGE_FACTOR) * min(1.0, delta_change * self.REVERSAL_DELTA_CHANGE_FACTOR)
        else:
            self.signals["reversal"] = 0.0
        
        # 3. Volatility Signal - predicts increasing volatility
        price_range = np.mean([c.high_price/c.low_price - 1.0 for c in recent])
        volume_variability = np.std([c.volume for c in recent]) / np.mean([c.volume for c in recent])
        delta_variability = np.std([c.delta_ratio for c in recent])
        
        self.signals["volatility"] = min(1.0, (price_range * self.VOLATILITY_PRICE_RANGE_FACTOR + volume_variability + delta_variability) / 3)
        
        # 4. Exhaustion Signal - detects potential buying/selling exhaustion
        # Delta dropping while price continues in same direction
        recent_deltas = [c.delta_ratio for c in recent]
        delta_trend = recent_deltas[-1] - recent_deltas[0]
        
        if (price_direction > 0 and delta_trend < -self.EXHAUSTION_DELTA_TREND_THRESHOLD) or \
           (price_direction < 0 and delta_trend > self.EXHAUSTION_DELTA_TREND_THRESHOLD):
            self.signals["exhaustion"] = min(1.0, abs(delta_trend) * self.EXHAUSTION_STRENGTH_FACTOR)
        else:
            self.signals["exhaustion"] = 0.0
            
        # Calculate overall signal strength for logging
        self.last_signal_strength = abs(self.signals["momentum"]) + self.signals["reversal"] + \
                                 self.signals["volatility"] + self.signals["exhaustion"]
        
        # Log signal changes if they're significant
        signal_change = sum(abs(self.signals[k] - old_signals[k]) for k in self.signals.keys())
        
        # Always log on significant changes or periodically
        if signal_change > self.SIGNIFICANT_SIGNAL_CHANGE_THRESHOLD or self.predictions_made % self.SIGNAL_LOGGING_FREQUENCY == 0:
            self.logger.info(
                f"Prediction #{self.predictions_made} - Signals: "
                f"Momentum={self.signals['momentum']:.2f} ({old_signals['momentum']:.2f}), "
                f"Reversal={self.signals['reversal']:.2f} ({old_signals['reversal']:.2f}), "
                f"Volatility={self.signals['volatility']:.2f} ({old_signals['volatility']:.2f}), "
                f"Exhaustion={self.signals['exhaustion']:.2f} ({old_signals['exhaustion']:.2f}), "
                f"Strength={self.last_signal_strength:.2f}"
            )
        
        # --- Log signal event for evaluation ---
        # This assumes _calculate_signals is called right after a candle completes
        # and `self.candles[-1]` is that completed candle.
        if self.candles:
            completed_candle = self.candles[-1] # The candle that just completed and triggered these signals
            signal_event = {
                'initial_timestamp': completed_candle.end_time,
                'entry_price': completed_candle.close_price,
                'signals': self.signals.copy(), 
                'candles_passed': 0,
                'future_prices_at_horizon': {h: None for h in self.evaluation_horizons},
                'pnl_at_horizon': {h: None for h in self.evaluation_horizons},
                'is_complete': False
            }
            self.signal_evaluation_log.append(signal_event)
    
    def get_predicted_parameters(self) -> Dict[str, float]:
        """
        Generate predicted parameters for the Avellaneda model based on signals
        
        Returns:
            Dictionary of predicted parameters for the Avellaneda model
        """
        if len(self.candles) < self.MIN_CANDLES_FOR_SIGNALS:
            return {
                "gamma_adjustment": 0.0,
                "kappa_adjustment": 0.0,
                "reservation_price_offset": 0.0,
                "trend_direction": 0,
                "volatility_adjustment": 0.0
            }
        
        # Fetch signals
        momentum = self.signals["momentum"]
        reversal = self.signals["reversal"]
        volatility = self.signals["volatility"]
        exhaustion = self.signals["exhaustion"]
        
        # 1. Gamma adjustment (risk aversion)
        # Increase gamma when volatility expected to rise or around reversal points
        gamma_adj = volatility * self.GAMMA_ADJ_VOLATILITY_FACTOR + reversal * self.GAMMA_ADJ_REVERSAL_FACTOR
        
        # 2. Kappa adjustment (market depth)
        # Reduce market depth parameter when market might be thin (high volatility)
        kappa_adj = volatility * self.KAPPA_ADJ_VOLATILITY_FACTOR # Factor is already negative
        
        # 3. Reservation price offset (predictive skew)
        # Skew reservation price based on momentum and potential reversals
        reservation_offset = 0.0
        if abs(momentum) > self.RESERVATION_OFFSET_MOMENTUM_THRESHOLD:
            # Strong momentum - skew in direction of momentum
            if exhaustion < self.RESERVATION_OFFSET_EXHAUSTION_THRESHOLD:  # Not showing exhaustion yet
                reservation_offset = momentum * self.RESERVATION_OFFSET_MOMENTUM_FACTOR  # Small adjustment
        elif reversal > self.RESERVATION_OFFSET_REVERSAL_THRESHOLD:
            # Strong reversal signal - skew against recent price direction
            last_candle = list(self.candles)[-1]
            prev_candle = list(self.candles)[-2]
            price_direction = np.sign(last_candle.close_price - prev_candle.close_price)
            reservation_offset = -price_direction * reversal * self.RESERVATION_OFFSET_REVERSAL_FACTOR
        
        # 4. Trend direction (-1, 0, 1)
        trend_direction = 0
        if abs(momentum) > self.TREND_DIRECTION_MOMENTUM_THRESHOLD:
            trend_direction = np.sign(momentum)
        
        # 5. Volatility adjustment
        # Increase volatility estimate when expecting higher volatility
        vol_adj = volatility * self.VOL_ADJ_VOLATILITY_FACTOR
        
        return {
            "gamma_adjustment": gamma_adj,
            "kappa_adjustment": kappa_adj,
            "reservation_price_offset": reservation_offset,
            "trend_direction": int(trend_direction),
            "volatility_adjustment": vol_adj
        }
    
    def get_signal_metrics(self) -> Dict[str, float]:
        """Get current signal metrics for debugging/monitoring"""
        return {
            "signals": self.signals,
            "ema_price": {k: v for k, v in self.ema_values.items()},
            "delta_ema": {k: v for k, v in self.delta_ema.items()},
            "cumulative_delta": self.cumulative_delta,
            "candle_count": len(self.candles),
            "current_volume": self.current_candle.volume if self.current_candle else 0,
            "using_exchange_data": self.use_exchange_data,
            "exchange_fetch_interval": self.fetch_interval_seconds
        }
        
    def evaluate_prediction_accuracy(self, actual_price: float, prediction_horizon: int = 5) -> float:
        """
        Evaluate prediction accuracy by comparing predicted direction with actual outcome
        
        Args:
            actual_price: The current actual price to compare with past predictions
            prediction_horizon: How many candles back to check predictions
            
        Returns:
            Accuracy score (0-1)
        """
        if len(self._prediction_accuracy) > 20:
            self._prediction_accuracy.pop(0)
            
        if len(self.candles) <= prediction_horizon:
            return 0.0
            
        # Get prediction made N candles ago
        old_prediction = list(self.candles)[-prediction_horizon].close_price
        pred_direction = self.signals["momentum"] * prediction_horizon
        
        # Calculate if prediction was correct
        actual_change = (actual_price / old_prediction) - 1.0
        prediction_correct = np.sign(pred_direction) == np.sign(actual_change)
        
        # Record accuracy
        self._prediction_accuracy.append(1.0 if prediction_correct else 0.0)
        
        # Log accuracy
        avg_accuracy = np.mean(self._prediction_accuracy)
        self.logger.debug(f"Prediction accuracy: {avg_accuracy:.2f} (last prediction: {'correct' if prediction_correct else 'incorrect'})")
        
        # Return rolling accuracy
        return avg_accuracy 