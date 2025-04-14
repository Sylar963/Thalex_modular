"""
Hedge manager that coordinates the hedging operations.
This is the main entry point for the hedging system that interacts with the trading bot.
"""

from typing import Dict, List, Tuple, Optional, Any
import time
import threading
from datetime import datetime
import json
from pathlib import Path
import os
import math

from .hedge_config import HedgeConfig
from .hedge_strategy import HedgeStrategy, create_hedge_strategy
from .hedge_execution import HedgeExecution, OrderSide, OrderStatus, HedgeOrder
from ...thalex_logging import LoggerFactory
from ...models.position_tracker import PositionTracker, Fill

# Create a simple Position class for tracking positions
class Position:
    """Simple position tracking class"""
    def __init__(self, symbol: str, size: float = 0.0, price: float = 0.0):
        self.symbol = symbol
        self.size = size
        self.price = price


class HedgePosition:
    """Tracks a hedge position for a specific asset pair"""
    
    def __init__(
        self,
        primary_asset: str,
        hedge_asset: str,
        primary_position: float = 0.0,
        primary_price: float = 0.0,
        hedge_position: float = 0.0,
        hedge_price: float = 0.0,
        entry_time: Optional[float] = None,
        pnl: float = 0.0,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize a hedge position
        
        Args:
            primary_asset: Primary asset being hedged
            hedge_asset: Asset used for hedging
            primary_position: Size of primary position
            primary_price: Price of primary position
            hedge_position: Size of hedge position
            hedge_price: Price of hedge position
            entry_time: Time when position was entered
            pnl: Current profit/loss
            metadata: Additional position metadata
        """
        self.primary_asset = primary_asset
        self.hedge_asset = hedge_asset
        self.primary_position = primary_position
        self.primary_price = primary_price
        self.hedge_position = hedge_position
        self.hedge_price = hedge_price
        self.entry_time = entry_time or time.time()
        self.last_update_time = self.entry_time
        self.pnl = pnl
        self.metadata = metadata or {}
        
        # Tracking for primary notional value
        self.primary_notional = abs(primary_position * primary_price)
        self.hedge_notional = abs(hedge_position * hedge_price)
        self.hedge_ratio = abs(self.hedge_notional / self.primary_notional) if self.primary_notional > 0 else 0
    
    def update_primary(self, position: float, price: float):
        """Update primary position"""
        old_position = self.primary_position
        self.primary_position = position
        self.primary_price = price
        self.primary_notional = abs(position * price)
        if self.primary_notional > 0:
            self.hedge_ratio = abs(self.hedge_notional / self.primary_notional)
        else:
            self.hedge_ratio = 0
        self.last_update_time = time.time()
        
        # Return True if position direction changed
        return (old_position > 0 and position <= 0) or (old_position < 0 and position >= 0) or (old_position == 0 and position != 0)
    
    def update_hedge(self, position: float, price: float):
        """Update hedge position"""
        self.hedge_position = position
        self.hedge_price = price
        self.hedge_notional = abs(position * price)
        if self.primary_notional > 0:
            self.hedge_ratio = abs(self.hedge_notional / self.primary_notional)
        else:
            self.hedge_ratio = 0
        self.last_update_time = time.time()
    
    def calculate_pnl(self, current_primary_price: float, current_hedge_price: float) -> float:
        """Calculate current PnL for the hedged position"""
        # PnL from primary position
        if self.primary_position > 0:  # Long
            primary_pnl = (current_primary_price - self.primary_price) * self.primary_position
        elif self.primary_position < 0:  # Short
            primary_pnl = (self.primary_price - current_primary_price) * abs(self.primary_position)
        else:
            primary_pnl = 0
            
        # PnL from hedge position
        if self.hedge_position > 0:  # Long
            hedge_pnl = (current_hedge_price - self.hedge_price) * self.hedge_position
        elif self.hedge_position < 0:  # Short
            hedge_pnl = (self.hedge_price - current_hedge_price) * abs(self.hedge_position)
        else:
            hedge_pnl = 0
            
        # Total PnL
        self.pnl = primary_pnl + hedge_pnl
        return self.pnl
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary"""
        return {
            "primary_asset": self.primary_asset,
            "hedge_asset": self.hedge_asset,
            "primary_position": self.primary_position,
            "primary_price": self.primary_price,
            "primary_notional": self.primary_notional,
            "hedge_position": self.hedge_position,
            "hedge_price": self.hedge_price,
            "hedge_notional": self.hedge_notional,
            "hedge_ratio": self.hedge_ratio,
            "entry_time": self.entry_time,
            "last_update_time": self.last_update_time,
            "pnl": self.pnl,
            "metadata": self.metadata
        }


class HedgeManager:
    """
    Manages hedging across multiple assets.
    This is the main entry point for the hedging system.
    """
    
    def __init__(
        self, 
        config_path: Optional[str] = None, 
        exchange_client: Any = None,
        strategy_type: str = "notional"
    ):
        """
        Initialize the hedge manager
        
        Args:
            config_path: Path to configuration file
            exchange_client: Exchange API client
            strategy_type: Type of hedge strategy to use ("notional" or "delta_neutral")
        """
        # Initialize configuration
        self.config = HedgeConfig(config_path)
        
        # Initialize logger
        self.logger = LoggerFactory.configure_component_logger(
            "hedge_manager",
            log_file="hedge_manager.log"
        )
        
        # Create strategy
        self.strategy = create_hedge_strategy(strategy_type, self.config)
        
        # Create execution module
        self.execution = HedgeExecution(self.config, exchange_client)
        
        # Initialize state
        self.active_hedges: Dict[str, Dict[str, HedgePosition]] = {}  # primary_asset -> {hedge_asset -> position}
        self.position_history: List[HedgePosition] = []
        self.market_prices: Dict[str, float] = {}  # asset -> price
        self.is_running = False
        self.lock = threading.RLock()
        
        # Track positions per asset
        self.positions: Dict[str, Position] = {}
        
        # Last update timestamps
        self.last_rebalance_time = 0
        self.last_pnl_calculation = 0
        self.state_file = Path("hedge_state.json")
        
        self.logger.info(f"Hedge manager initialized with strategy: {strategy_type}")
        
        # Try to load state
        self._load_state()
    
    def start(self):
        """Start the hedge manager"""
        if self.is_running:
            self.logger.warning("Hedge manager already running")
            return
            
        self.is_running = True
        self.logger.info("Hedge manager started")
        
        # Start rebalance thread
        self.rebalance_thread = threading.Thread(target=self._rebalance_loop, daemon=True)
        self.rebalance_thread.start()
    
    def stop(self):
        """Stop the hedge manager"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Save state
        self._save_state()
        
        self.logger.info("Hedge manager stopped")
    
    def update_position(self, asset: str, position: float, price: float) -> Dict:
        """
        Update a position and rebalance hedges
        
        Args:
            asset: Asset symbol
            position: Current position size
            price: Current price
            
        Returns:
            Dict with hedge operations and status
        """
        with self.lock:
            self.logger.info(f"Updating position for {asset}: {position} @ {price}")
            
            # Check if this asset is configured for hedging
            hedge_assets = self.config.get_hedge_assets(asset)
            if not hedge_assets:
                self.logger.warning(f"Asset {asset} not configured for hedging")
                return {"status": "no_config", "hedges": []}
            
            # Update position tracker
            if asset not in self.positions:
                self.positions[asset] = Position(asset)
                
            old_position = self.positions[asset].size
            self.positions[asset].size = position
            self.positions[asset].price = price

            # Update market prices
            self.market_prices[asset] = price
            self.logger.info(f"Updated position and price for {asset}: {old_position} -> {position} @ {price}")
            
            # Initialize hedge positions if they don't exist
            if asset not in self.active_hedges:
                self.active_hedges[asset] = {}
            
            # Get correlation factors for this asset's hedge pairs
            correlation_factors = self.config.get_correlation_factors(asset)
            hedge_operations = []
            
            self.logger.info(f"Processing hedges for {asset} with {len(hedge_assets)} hedge assets")
            
            # Calculate required hedges
            for i, hedge_asset in enumerate(hedge_assets):
                self.logger.info(f"Processing hedge {i+1}/{len(hedge_assets)}: {hedge_asset}")
                # Get correlation factor for this hedge pair
                correlation_factor = correlation_factors[i] if i < len(correlation_factors) else 1.0
                hedge_config = self.config.get_hedge_pair_config(asset)
                
                # Create hedge position if it doesn't exist
                if hedge_asset not in self.active_hedges[asset]:
                    self.logger.info(f"Creating new hedge position for {asset}/{hedge_asset}")
                    self.active_hedges[asset][hedge_asset] = HedgePosition(
                        primary_asset=asset,
                        hedge_asset=hedge_asset,
                        primary_position=position,
                        primary_price=price,
                        hedge_position=0.0,
                        hedge_price=0.0,
                        entry_time=time.time()
                    )
                
                # Get current hedge position
                hedge_position = self.active_hedges[asset][hedge_asset]
                
                # Update primary position
                hedge_position.update_primary(position, price)
                
                # Get current price for hedge asset
                hedge_price = self.market_prices.get(hedge_asset, 0.0)
                if hedge_price <= 0:
                    self.logger.warning(f"No price available for hedge asset {hedge_asset}")
                    continue
                
                # Calculate required hedge position
                required_hedge = self._calculate_required_hedge(
                    primary_position=position,
                    primary_price=price,
                    hedge_price=hedge_price,
                    correlation_factor=correlation_factor,
                    strategy=self.strategy
                )
                
                self.logger.info(f"Calculated required hedge for {asset}/{hedge_asset}: {required_hedge} @ {hedge_price}")
                self.logger.info(f"Current hedge position: {hedge_position.hedge_position}")
                
                # Check if hedge needs to be adjusted
                min_hedge_size = hedge_config.get("min_hedge_size", 0.01)
                deviation_threshold = self.config.get_hedge_settings().get("deviation_threshold", 0.05)
                
                # Calculate hedge adjustment
                hedge_adjustment = required_hedge - hedge_position.hedge_position
                
                # Only adjust if needed
                if abs(hedge_adjustment) >= min_hedge_size:
                    # Calculate deviation percentage
                    deviation = abs(hedge_adjustment / required_hedge) if required_hedge != 0 else float('inf')
                    
                    # Only rebalance if deviation is significant
                    if deviation >= deviation_threshold or hedge_position.hedge_position == 0:
                        self.logger.info(f"Hedge adjustment needed: {hedge_adjustment} ({deviation:.2%} deviation)")
                        
                        # Execute the hedge
                        if hedge_adjustment > 0:
                            # Need to buy
                            self.logger.info(f"Executing BUY order for {hedge_asset}: {abs(hedge_adjustment)} @ {hedge_price}")
                            success = self._execute_hedge_order(
                                hedge_asset=hedge_asset,
                                side=OrderSide.BUY,
                                size=abs(hedge_adjustment),
                                price=hedge_price,
                                metadata={"primary_asset": asset, "position": position}
                            )
                        else:
                            # Need to sell
                            self.logger.info(f"Executing SELL order for {hedge_asset}: {abs(hedge_adjustment)} @ {hedge_price}")
                            success = self._execute_hedge_order(
                                hedge_asset=hedge_asset,
                                side=OrderSide.SELL,
                                size=abs(hedge_adjustment),
                                price=hedge_price,
                                metadata={"primary_asset": asset, "position": position}
                            )
                        
                        if success:
                            self.logger.info(f"Hedge order executed successfully")
                            # Update hedge position
                            old_hedge = hedge_position.hedge_position
                            hedge_position.update_hedge(required_hedge, hedge_price)
                            
                            # Record the operation
                            hedge_operations.append({
                                "hedge_asset": hedge_asset,
                                "side": "BUY" if hedge_adjustment > 0 else "SELL",
                                "size": abs(hedge_adjustment),
                                "price": hedge_price,
                                "old_position": old_hedge,
                                "new_position": hedge_position.hedge_position
                            })
                        else:
                            self.logger.error(f"Failed to execute hedge order for {hedge_asset}")
                    else:
                        self.logger.info(f"Hedge adjustment not needed (deviation {deviation:.2%} < threshold {deviation_threshold:.2%})")
                else:
                    self.logger.info(f"Hedge adjustment too small: {abs(hedge_adjustment)} < min size {min_hedge_size}")
                
                # Calculate PnL
                hedge_position.calculate_pnl(price, hedge_price)
            
            # Save state after update
            self._save_state()
            
            return {
                "status": "hedged" if hedge_operations else "no_change",
                "hedges": hedge_operations
            }
    
    def update_market_price(self, asset: str, price: float):
        """
        Update market price for an asset
        
        Args:
            asset: Asset symbol
            price: Current market price
        """
        with self.lock:
            self.market_prices[asset] = price
    
    def get_hedged_position(self, primary_asset: str, hedge_asset: Optional[str] = None) -> Optional[HedgePosition]:
        """
        Get current hedge position for an asset pair
        
        Args:
            primary_asset: Primary asset symbol
            hedge_asset: Optional hedge asset symbol
            
        Returns:
            HedgePosition if found, None otherwise
        """
        with self.lock:
            if primary_asset not in self.active_hedges:
                return None
                
            if hedge_asset is None:
                # Return first hedge if no specific one requested
                for asset, position in self.active_hedges[primary_asset].items():
                    return position
                return None
                
            return self.active_hedges[primary_asset].get(hedge_asset)
    
    def get_all_hedged_positions(self) -> Dict[str, Dict[str, HedgePosition]]:
        """
        Get all active hedge positions
        
        Returns:
            Dict of primary_asset -> {hedge_asset -> position}
        """
        with self.lock:
            return self.active_hedges.copy()
    
    def calculate_portfolio_pnl(self) -> Dict:
        """
        Calculate PnL for all hedged positions
        
        Returns:
            Dict with PnL information
        """
        with self.lock:
            self.last_pnl_calculation = time.time()
            
            total_pnl = 0.0
            position_pnls = {}
            
            for primary_asset, hedges in self.active_hedges.items():
                position_pnls[primary_asset] = {}
                
                for hedge_asset, position in hedges.items():
                    # Get current prices
                    primary_price = self.market_prices.get(primary_asset, position.primary_price)
                    hedge_price = self.market_prices.get(hedge_asset, position.hedge_price)
                    
                    # Calculate PnL
                    pnl = position.calculate_pnl(primary_price, hedge_price)
                    total_pnl += pnl
                    
                    position_pnls[primary_asset][hedge_asset] = {
                        "pnl": pnl,
                        "primary_position": position.primary_position,
                        "primary_price": position.primary_price,
                        "hedge_position": position.hedge_position,
                        "hedge_price": position.hedge_price,
                        "current_primary_price": primary_price,
                        "current_hedge_price": hedge_price
                    }
            
            return {
                "total_pnl": total_pnl,
                "position_pnls": position_pnls,
                "timestamp": self.last_pnl_calculation
            }
    
    def on_fill(self, fill: Any):
        """
        Process a fill event from the primary trading system
        
        Args:
            fill: Fill event (can be standard Fill or custom TestFill)
        """
        with self.lock:
            # Handle different fill object formats
            # First check if it's our custom test fill format with instrument, price, size attributes
            if hasattr(fill, 'instrument') and hasattr(fill, 'price') and hasattr(fill, 'size'):
                asset = fill.instrument
                price = fill.price
                is_buy = fill.is_buy
                size = fill.size if is_buy else -fill.size
            # If it's the standard Fill object from the trading system
            elif hasattr(fill, 'order_id') and hasattr(fill, 'fill_price') and hasattr(fill, 'fill_size'):
                # Extract information from standard Fill object
                # We'll need to determine the asset from the Fill context or metadata
                # For now, we'll use a placeholder and it needs to be implemented based on actual Fill data
                asset = fill.instrument if hasattr(fill, 'instrument') else "unknown"
                price = fill.fill_price
                is_buy = fill.side.lower() == "buy" if hasattr(fill, 'side') else True
                size = fill.fill_size if is_buy else -fill.fill_size
            else:
                self.logger.error(f"Unsupported fill object format: {fill}")
                return
            
            self.logger.info(f"Processing fill: {asset} {'BUY' if is_buy else 'SELL'} {abs(size)} @ {price}")
            
            # Update position tracking
            if asset not in self.positions:
                self.positions[asset] = Position(asset)
                
            position = self.positions[asset]
            old_size = position.size
            position.size += size
            position.price = price
            
            # Update market prices for the primary asset that had a fill
            self.market_prices[asset] = price
            
            # Update market prices for hedge assets using estimation based on correlation if needed
            self._update_market_prices_for_hedge_assets(asset, price)
            
            self.logger.info(f"Updated position for {asset}: {old_size:.4f} -> {position.size:.4f} @ {price:.2f}")
            
            # Update hedge for this position
            self.update_position(asset, position.size, price)
    
    def process_fills(self, fills: List[Fill]):
        """
        Process multiple fills from the primary trading system
        
        Args:
            fills: List of fills
        """
        for fill in fills:
            self.on_fill(fill)
    
    def _execute_hedge_order(
        self, 
        hedge_asset: str, 
        side: OrderSide, 
        size: float, 
        price: float,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Execute a hedge order
        
        Args:
            hedge_asset: Asset to trade
            side: Buy or sell
            size: Order size
            price: Current price (for limit orders)
            metadata: Order metadata
            
        Returns:
            True if order was successfully executed
        """
        # Get execution mode from config
        execution_mode = self.config.get_hedge_settings().get("execution_mode", "market")
        
        # Execute order based on mode
        if execution_mode == "market":
            order = self.execution.place_market_order(
                symbol=hedge_asset,
                side=side,
                size=size,
                metadata=metadata
            )
        else:
            # For limit orders, adjust price based on side for higher fill probability
            slippage = self.config.get_hedge_pair_config(hedge_asset).get("slippage_tolerance", 0.001)
            
            if side == OrderSide.BUY:
                # Pay slightly more for buys
                adjusted_price = price * (1 + slippage)
            else:
                # Accept slightly less for sells
                adjusted_price = price * (1 - slippage)
                
            order = self.execution.place_limit_order(
                symbol=hedge_asset,
                side=side,
                size=size,
                price=adjusted_price,
                metadata=metadata
            )
            
            # For limit orders, wait for fill or timeout
            timeout = self.config.get_hedge_settings().get("execution_timeout", 30)
            
            # Check limit order status (in a real system, this would use callbacks or polling)
            # For simplicity, we'll simulate eventual fill
            if order.status == OrderStatus.OPEN:
                # Simulate fill after some time
                order.update(OrderStatus.FILLED, order.size, price)
        
        # Return True if order was filled
        return order.status == OrderStatus.FILLED
    
    def _rebalance_loop(self):
        """Background thread to periodically rebalance hedges"""
        while self.is_running:
            try:
                # Get rebalance frequency from config
                rebalance_frequency = self.config.get_hedge_settings().get("rebalance_frequency", 300)
                
                # Check if it's time to rebalance
                current_time = time.time()
                if current_time - self.last_rebalance_time >= rebalance_frequency:
                    self._rebalance_all_hedges()
                    self.last_rebalance_time = current_time
                    
                # Calculate portfolio PnL periodically
                if current_time - self.last_pnl_calculation >= 60:  # Every minute
                    self.calculate_portfolio_pnl()
            except Exception as e:
                self.logger.error(f"Error in rebalance loop: {e}")
                
            # Sleep to prevent high CPU usage
            time.sleep(1)
    
    def _rebalance_all_hedges(self):
        """Rebalance all active hedges"""
        with self.lock:
            self.logger.info("Rebalancing all hedges")
            
            # First update all prices
            self._update_all_prices()
            
            for primary_asset, hedges in self.active_hedges.items():
                # Get current position and price for primary asset
                if primary_asset in self.positions:
                    position = self.positions[primary_asset]
                    price = self.market_prices.get(primary_asset, position.price)
                    
                    if price <= 0:
                        self.logger.warning(f"No price available for primary asset {primary_asset}, skipping rebalance")
                        continue
                    
                    # Update position and hedges
                    self.update_position(primary_asset, position.size, price)
            
            self.logger.info("Rebalance completed")
    
    def _update_all_prices(self):
        """Update prices for all assets from exchange client"""
        # First try to fetch prices directly from exchange
        assets_to_update = set()
        
        # Collect all assets we need prices for
        for primary_asset, hedges in self.active_hedges.items():
            assets_to_update.add(primary_asset)
            assets_to_update.update(hedges.keys())
        
        # Also add assets from positions that may not be in active_hedges yet
        assets_to_update.update(self.positions.keys())
        
        for asset in assets_to_update:
            # Try to fetch price from exchange
            price = self._fetch_price_from_exchange(asset)
            if price > 0:
                # Update the price in our cache
                old_price = self.market_prices.get(asset, 0.0)
                self.market_prices[asset] = price
                self.logger.info(f"Updated price for {asset}: {old_price} -> {price}")
            elif asset in self.market_prices and self.market_prices[asset] > 0:
                # We already have a price, keep using it
                self.logger.info(f"Using existing price for {asset}: {self.market_prices[asset]}")
            else:
                # Try to estimate price based on other assets
                for primary_asset, hedges in self.active_hedges.items():
                    if asset in hedges and primary_asset in self.market_prices and self.market_prices[primary_asset] > 0:
                        self._update_market_prices_for_hedge_assets(primary_asset, self.market_prices[primary_asset])
                        break
                    elif asset == primary_asset and any(hedge_asset in self.market_prices and self.market_prices[hedge_asset] > 0 for hedge_asset in hedges.keys()):
                        # Find a hedge asset with a price we can use for estimation
                        for hedge_asset in hedges.keys():
                            if hedge_asset in self.market_prices and self.market_prices[hedge_asset] > 0:
                                # Estimate price using the hedge asset price
                                correlation_factors = self.config.get_correlation_factors(primary_asset)
                                hedge_idx = list(hedges.keys()).index(hedge_asset)
                                correlation = correlation_factors[hedge_idx] if hedge_idx < len(correlation_factors) else 1.0
                                
                                # Use fallback as last resort - clearly mark as fallback
                                self.logger.warning(f"No real-time price available for {primary_asset}, using fallback estimation")
                                
                                if primary_asset == "BTC-PERPETUAL" and hedge_asset == "ETH-PERPETUAL":
                                    estimated_price = self.market_prices[hedge_asset] * 16.0
                                    self.market_prices[primary_asset] = estimated_price
                                    self.logger.warning(f"FALLBACK PRICE for {primary_asset}: {estimated_price} (estimated from {hedge_asset})")
                                elif primary_asset == "ETH-PERPETUAL" and hedge_asset == "BTC-PERPETUAL":
                                    estimated_price = self.market_prices[hedge_asset] / 16.0
                                    self.market_prices[primary_asset] = estimated_price
                                    self.logger.warning(f"FALLBACK PRICE for {primary_asset}: {estimated_price} (estimated from {hedge_asset})")
                                else:
                                    # For other asset pairs, log a specific warning
                                    self.logger.warning(f"Unable to estimate price for {primary_asset} - no fallback ratio defined")
                                
                                break
    
    def _save_state(self):
        """
        Save hedge manager state to file
        """
        try:
            # Convert state to dict
            state = {
                "active_hedges": {},
                "market_prices": self.market_prices
            }
            
            # Convert objects to dicts
            for primary_asset, hedges in self.active_hedges.items():
                state["active_hedges"][primary_asset] = {}
                for hedge_asset, position in hedges.items():
                    state["active_hedges"][primary_asset][hedge_asset] = position.to_dict()
            
            # Save to file
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
            self.logger.debug(f"Hedge state saved to {self.state_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving hedge state: {e}")
    
    def _load_state(self):
        """Load state from file"""
        if not self.state_file.exists():
            return
            
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                
            # Restore market prices
            self.market_prices = state.get("market_prices", {})
            
            # Restore active hedges
            hedge_state = state.get("active_hedges", {})
            for primary_asset, hedges in hedge_state.items():
                self.active_hedges[primary_asset] = {}
                for hedge_asset, position_data in hedges.items():
                    self.active_hedges[primary_asset][hedge_asset] = HedgePosition(
                        primary_asset=position_data["primary_asset"],
                        hedge_asset=position_data["hedge_asset"],
                        primary_position=position_data["primary_position"],
                        primary_price=position_data["primary_price"],
                        hedge_position=position_data["hedge_position"],
                        hedge_price=position_data["hedge_price"],
                        entry_time=position_data["entry_time"],
                        pnl=position_data["pnl"],
                        metadata=position_data["metadata"]
                    )
                    
            self.logger.info(f"Loaded hedge state from {self.state_file}")
        except Exception as e:
            self.logger.error(f"Error loading hedge state: {e}")

    def _calculate_required_hedge(
        self, 
        primary_position: float,
        primary_price: float,
        hedge_price: float,
        correlation_factor: float,
        strategy: Any
    ) -> float:
        """
        Calculate the required hedge position based on the primary position
        
        Args:
            primary_position: Primary position size
            primary_price: Primary position price
            hedge_price: Hedge asset price
            correlation_factor: Correlation factor between primary and hedge asset
            strategy: Hedge strategy object
            
        Returns:
            Required hedge position size
        """
        # Calculate monetary value of primary position
        primary_notional = primary_position * primary_price
        
        # Calculate required hedge notional
        hedge_notional = -primary_notional * correlation_factor
        
        # Convert to size in hedge asset
        if hedge_price > 0:
            hedge_size = hedge_notional / hedge_price
        else:
            hedge_size = 0
            
        self.logger.info(
            f"Hedge calculation: Primary={primary_position}@{primary_price} "
            f"(${primary_notional:.2f}), Hedge notional=${hedge_notional:.2f}, "
            f"Hedge size={hedge_size:.6f}@{hedge_price:.2f}, Correlation={correlation_factor:.2f}"
        )
        
        return hedge_size

    def _update_market_prices_for_hedge_assets(self, primary_asset: str, primary_price: float):
        """
        Update market prices for hedge assets based on primary asset price
        Uses correlation to estimate hedge asset price if not available
        
        Args:
            primary_asset: Asset that was filled
            primary_price: Current price of the primary asset
        """
        # Get hedge assets for the primary asset
        hedge_assets = self.config.get_hedge_assets(primary_asset)
        if not hedge_assets:
            return
            
        # Get correlation factors
        correlation_factors = self.config.get_correlation_factors(primary_asset)
        
        # Process each hedge asset
        for i, hedge_asset in enumerate(hedge_assets):
            # Try to fetch price from exchange client first - this is the primary source of truth
            fetched_price = self._fetch_price_from_exchange(hedge_asset)
            if fetched_price > 0:
                self.market_prices[hedge_asset] = fetched_price
                self.logger.info(f"Using real-time price for {hedge_asset}: {fetched_price} from exchange")
                continue
                
            # Skip if we already have a valid price
            if hedge_asset in self.market_prices and self.market_prices[hedge_asset] > 0:
                self.logger.info(f"Keeping existing valid price for {hedge_asset}: {self.market_prices[hedge_asset]}")
                continue
                
            # If we get here, we couldn't get a real-time price - use fallback as last resort
            self.logger.warning(f"No real-time price data available for {hedge_asset}, using fallback estimation")
            
            # Check if this is the first update - we need both prices to establish a baseline
            if primary_asset in self.market_prices and hedge_asset in self.market_prices:
                # We have both prices, but hedge price may be zero or invalid
                if self.market_prices[hedge_asset] <= 0:
                    # Try to estimate based on correlation and last known good value
                    self.logger.warning(f"FALLBACK: Estimating price for {hedge_asset} based on correlation with {primary_asset}")
                    
                    # Use approximate price ranges for estimation (can be refined with actual data)
                    if hedge_asset == "ETH-PERPETUAL" and primary_asset == "BTC-PERPETUAL":
                        # Typical BTC/ETH price ratio (approximately 16:1 as of 2025)
                        estimated_price = primary_price / 16.0
                        self.market_prices[hedge_asset] = estimated_price
                        self.logger.warning(f"FALLBACK PRICE for {hedge_asset}: {estimated_price} (estimated from {primary_asset})")
                    elif hedge_asset == "BTC-PERPETUAL" and primary_asset == "ETH-PERPETUAL":
                        # Inverse of above
                        estimated_price = primary_price * 16.0
                        self.market_prices[hedge_asset] = estimated_price
                        self.logger.warning(f"FALLBACK PRICE for {hedge_asset}: {estimated_price} (estimated from {primary_asset})")
            else:
                # First time seeing one of these assets
                self.logger.warning(f"Need both {primary_asset} and {hedge_asset} real-time prices for accurate hedging")
    
    def _fetch_price_from_exchange(self, asset: str) -> float:
        """
        Fetch price for an asset from the exchange client
        
        Args:
            asset: Asset symbol to fetch price for
            
        Returns:
            Current price or 0 if not available
        """
        if not self.execution.exchange_client:
            return 0.0
            
        try:
            # Different exchange clients have different methods to fetch price data
            # Here we handle a few common patterns
            exchange_client = self.execution.exchange_client
            
            # Try standard method first
            if hasattr(exchange_client, 'get_price') and callable(exchange_client.get_price):
                price = exchange_client.get_price(asset)
                if price > 0:
                    return price
                    
            # Try fetch_ticker if available
            if hasattr(exchange_client, 'fetch_ticker') and callable(exchange_client.fetch_ticker):
                try:
                    ticker = exchange_client.fetch_ticker(asset)
                    if ticker and 'last' in ticker and ticker['last'] > 0:
                        return float(ticker['last'])
                except Exception as e:
                    self.logger.debug(f"Error fetching ticker for {asset}: {e}")
                    
            # Try get_mark_price if available
            if hasattr(exchange_client, 'get_mark_price') and callable(exchange_client.get_mark_price):
                price = exchange_client.get_mark_price(asset)
                if price > 0:
                    return price
                    
            # Try get_instrument_data if available (Thalex specific)
            if hasattr(exchange_client, 'get_instrument_data') and callable(exchange_client.get_instrument_data):
                data = exchange_client.get_instrument_data(asset)
                if data and 'mark_price' in data:
                    return float(data['mark_price'])
                    
            # Try get_ticker if available
            if hasattr(exchange_client, 'get_ticker') and callable(exchange_client.get_ticker):
                ticker = exchange_client.get_ticker(asset)
                if ticker:
                    # Different exchanges use different keys for price
                    for key in ['mark_price', 'last', 'price', 'close']:
                        if key in ticker and ticker[key] > 0:
                            return float(ticker[key])
                            
            # Last resort - check if exchange client has market_prices attribute
            if hasattr(exchange_client, 'market_prices') and isinstance(exchange_client.market_prices, dict):
                if asset in exchange_client.market_prices and exchange_client.market_prices[asset] > 0:
                    return float(exchange_client.market_prices[asset])
                    
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error fetching price for {asset} from exchange: {e}")
            return 0.0


# Factory function to create hedge manager
def create_hedge_manager(
    config_path: Optional[str] = None,
    exchange_client: Any = None,
    strategy_type: str = "notional"
) -> HedgeManager:
    """
    Factory function to create a hedge manager
    
    Args:
        config_path: Path to configuration file
        exchange_client: Exchange client instance
        strategy_type: Hedge strategy type
        
    Returns:
        HedgeManager instance
    """
    return HedgeManager(
        config_path=config_path,
        exchange_client=exchange_client,
        strategy_type=strategy_type
    ) 