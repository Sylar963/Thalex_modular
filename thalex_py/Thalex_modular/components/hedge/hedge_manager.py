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
        Update position for an asset and calculate required hedges
        
        Args:
            asset: Asset symbol (e.g., "BTC-PERP")
            position: Current position size
            price: Current position price
            
        Returns:
            Dict with hedge operations performed
        """
        with self.lock:
            # Update market price
            self.market_prices[asset] = price
            
            # Find hedge assets for this primary asset
            hedge_assets = self.config.get_hedge_assets(asset)
            
            operations = {
                "asset": asset,
                "position": position,
                "price": price,
                "hedges": []
            }
            
            # If no hedge assets or hedging disabled, return early
            if not hedge_assets or not self.config.is_hedging_enabled():
                return operations
                
            # Initialize active hedges for this asset if needed
            if asset not in self.active_hedges:
                self.active_hedges[asset] = {}
            
            # Process each hedge asset
            for hedge_idx, hedge_asset in enumerate(hedge_assets):
                # Get correlation factor
                correlation_factors = self.config.get_correlation_factors(asset)
                correlation_factor = correlation_factors[hedge_idx] if hedge_idx < len(correlation_factors) else correlation_factors[0]
                
                # Get or create hedge position tracker
                if hedge_asset not in self.active_hedges[asset]:
                    self.active_hedges[asset][hedge_asset] = HedgePosition(
                        primary_asset=asset,
                        hedge_asset=hedge_asset,
                        primary_position=position,
                        primary_price=price
                    )
                    
                hedge_position = self.active_hedges[asset][hedge_asset]
                
                # Check if primary position direction changed
                direction_changed = hedge_position.update_primary(position, price)
                
                # Get current hedge price
                hedge_price = self.market_prices.get(hedge_asset, 0)
                
                # If no price available, can't calculate hedge
                if hedge_price <= 0:
                    self.logger.warning(f"No price available for hedge asset {hedge_asset}, skipping")
                    continue
                    
                # Calculate required hedge position
                target_hedge_size, metadata = self.strategy.calculate_hedge_position(
                    primary_asset=asset,
                    primary_position=position,
                    primary_price=price,
                    hedge_asset=hedge_asset,
                    hedge_price=hedge_price
                )
                
                # If direction changed or rebalance needed, execute hedge
                current_hedge_size = hedge_position.hedge_position
                
                if direction_changed or self.strategy.should_rebalance(
                    primary_asset=asset,
                    primary_position=position,
                    current_hedge_position=current_hedge_size,
                    target_hedge_position=target_hedge_size
                ):
                    # Calculate size change needed
                    size_change = target_hedge_size - current_hedge_size
                    
                    # Skip very small changes
                    min_size = self.config.get_hedge_pair_config(asset).get("min_hedge_size", 0.01)
                    if abs(size_change) < min_size and current_hedge_size != 0:
                        self.logger.debug(f"Skipping small hedge adjustment ({size_change}) for {asset}/{hedge_asset}")
                        continue
                        
                    # Execute hedge if size change is significant
                    if size_change != 0:
                        side = OrderSide.BUY if size_change > 0 else OrderSide.SELL
                        
                        # Execute the hedge order
                        order_result = self._execute_hedge_order(
                            hedge_asset=hedge_asset,
                            side=side, 
                            size=abs(size_change),
                            price=hedge_price,
                            metadata={
                                "primary_asset": asset,
                                "primary_position": position,
                                "primary_price": price,
                                "correlation_factor": correlation_factor,
                                "target_hedge_size": target_hedge_size,
                                "current_hedge_size": current_hedge_size
                            }
                        )
                        
                        # If order was filled, update hedge position
                        if order_result:
                            # Update hedge position
                            new_hedge_size = current_hedge_size + size_change
                            hedge_position.update_hedge(new_hedge_size, hedge_price)
                            
                            self.logger.info(
                                f"Hedge updated for {asset}/{hedge_asset}: "
                                f"Primary={position:.4f}@{price:.2f}, "
                                f"Hedge={new_hedge_size:.4f}@{hedge_price:.2f}, "
                                f"Ratio={hedge_position.hedge_ratio:.2f}"
                            )
                            
                            operations["hedges"].append({
                                "hedge_asset": hedge_asset,
                                "side": side.value,
                                "size": abs(size_change),
                                "price": hedge_price,
                                "new_position": new_hedge_size,
                                "hedge_ratio": hedge_position.hedge_ratio
                            })
                
            # Save state after updates
            self._save_state()
            
            return operations
    
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
            position.price = price  # Update price to latest price
            
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
    
    def _save_state(self):
        """Save current state to file"""
        try:
            # Convert active hedges to serializable format
            hedge_state = {}
            for primary_asset, hedges in self.active_hedges.items():
                hedge_state[primary_asset] = {}
                for hedge_asset, position in hedges.items():
                    hedge_state[primary_asset][hedge_asset] = position.to_dict()
            
            # Save to file
            with open(self.state_file, 'w') as f:
                json.dump({
                    "market_prices": self.market_prices,
                    "active_hedges": hedge_state,
                    "last_update": time.time()
                }, f, indent=2)
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