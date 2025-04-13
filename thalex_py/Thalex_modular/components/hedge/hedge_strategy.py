"""
Hedge strategy interface and implementations.
Define different strategies for hedging positions across assets.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time
import math
import logging
import json
from pathlib import Path

from .hedge_config import HedgeConfig
from ...thalex_logging import LoggerFactory


class HedgeStrategy(ABC):
    """Base abstract class for hedge strategies"""
    
    def __init__(self, config: HedgeConfig):
        """
        Initialize the hedge strategy with configuration
        
        Args:
            config: Hedge configuration
        """
        self.config = config
        # Add logger
        self.logger = LoggerFactory.configure_component_logger(
            "hedge_strategy",
            log_file="hedge_strategy.log"
        )
    
    @abstractmethod
    def calculate_hedge_position(
        self, 
        primary_asset: str, 
        primary_position: float,
        primary_price: float,
        hedge_asset: str,
        hedge_price: float
    ) -> Tuple[float, Dict]:
        """
        Calculate required hedge position size
        
        Args:
            primary_asset: Primary asset symbol
            primary_position: Current position size in primary asset
            primary_price: Current price of primary asset
            hedge_asset: Hedge asset symbol
            hedge_price: Current price of hedge asset
        
        Returns:
            Tuple of (required_hedge_position, metadata)
        """
        pass
    
    @abstractmethod
    def should_rebalance(
        self, 
        primary_asset: str,
        primary_position: float,
        current_hedge_position: float,
        target_hedge_position: float
    ) -> bool:
        """
        Determine if hedge position should be rebalanced
        
        Args:
            primary_asset: Primary asset symbol
            primary_position: Current position in primary asset
            current_hedge_position: Current hedge position
            target_hedge_position: Target hedge position
            
        Returns:
            True if rebalance is needed, False otherwise
        """
        pass


class NotionalValueHedgeStrategy(HedgeStrategy):
    """
    Hedge strategy based on notional value equivalence.
    Maintains equivalent notional exposure across assets.
    """
    
    def calculate_hedge_position(
        self, 
        primary_asset: str, 
        primary_position: float,
        primary_price: float,
        hedge_asset: str,
        hedge_price: float
    ) -> Tuple[float, Dict]:
        """
        Calculate required hedge position size based on notional value
        
        Args:
            primary_asset: Primary asset symbol (e.g., "BTC-PERP")
            primary_position: Current position size in primary asset
            primary_price: Current price of primary asset
            hedge_asset: Hedge asset symbol (e.g., "ETH-PERP")
            hedge_price: Current price of hedge asset
        
        Returns:
            Tuple of (required_hedge_position, metadata)
        """
        # If no primary position, no hedge needed
        if primary_position == 0:
            return 0, {"reason": "no_primary_position"}
            
        # Get correlation factor for this pair
        correlation_factors = self.config.get_correlation_factors(primary_asset)
        correlation_factor = correlation_factors[0]  # Default to first factor
        
        # Calculate monetary position (in USD)
        monetary_position = primary_position * primary_price
        
        # Get logger if available, otherwise create a simple logger
        logger = getattr(self, 'logger', None)
        if logger:
            logger.info(f"Monetary position: ${monetary_position:.2f} from {primary_position} {primary_asset} @ {primary_price}")
        
        # Hedge in opposite direction using equivalent cash value (adjusted by correlation)
        hedge_monetary_position = -monetary_position * correlation_factor
        
        # Convert to hedge asset units
        hedge_position = hedge_monetary_position / hedge_price
        
        # Log the hedge details
        if logger:
            logger.info(f"Hedge: ${hedge_monetary_position:.2f} as {hedge_position} {hedge_asset} @ {hedge_price}")
        
        # Apply max hedge ratio if configured
        max_hedge_ratio = self.config.get_hedge_settings().get("max_hedge_ratio", 1.0)
        if max_hedge_ratio < 1.0:
            hedge_position *= max_hedge_ratio
            hedge_monetary_position *= max_hedge_ratio
            
        # Apply minimum size threshold if needed
        min_hedge_size = self.config.get_hedge_pair_config(primary_asset).get("min_hedge_size", 0.01)
        if abs(hedge_position) < min_hedge_size:
            if hedge_position != 0:
                hedge_position = min_hedge_size * math.copysign(1, hedge_position)
                # Recalculate notional with adjusted position
                hedge_monetary_position = hedge_position * hedge_price
            
        return hedge_position, {
            "primary_monetary": monetary_position,
            "hedge_monetary": hedge_monetary_position,
            "correlation_factor": correlation_factor,
            "hedge_ratio": max_hedge_ratio
        }
    
    def should_rebalance(
        self, 
        primary_asset: str,
        primary_position: float,
        current_hedge_position: float,
        target_hedge_position: float
    ) -> bool:
        """
        Determine if hedge position should be rebalanced
        
        Args:
            primary_asset: Primary asset symbol
            primary_position: Current position in primary asset
            current_hedge_position: Current hedge position
            target_hedge_position: Target hedge position
            
        Returns:
            True if rebalance is needed, False otherwise
        """
        # If no primary position, close any hedge position
        if primary_position == 0:
            return current_hedge_position != 0
            
        # If no current hedge, always create one
        if current_hedge_position == 0:
            return target_hedge_position != 0
            
        # Calculate deviation percentage
        if target_hedge_position == 0:
            return current_hedge_position != 0
            
        deviation = abs(current_hedge_position - target_hedge_position) / abs(target_hedge_position)
        
        # Get deviation threshold from config
        threshold = self.config.get_hedge_settings().get("deviation_threshold", 0.05)
        
        # Rebalance if deviation exceeds threshold
        return deviation > threshold


class DeltaNeutralHedgeStrategy(HedgeStrategy):
    """
    Hedge strategy focused on delta neutrality across portfolio.
    Uses risk metrics to calculate appropriate hedge sizes.
    """
    
    def calculate_hedge_position(
        self, 
        primary_asset: str, 
        primary_position: float,
        primary_price: float,
        hedge_asset: str,
        hedge_price: float
    ) -> Tuple[float, Dict]:
        """
        Calculate required hedge position size based on delta neutrality
        
        Args:
            primary_asset: Primary asset symbol
            primary_position: Current position size in primary asset
            primary_price: Current price of primary asset
            hedge_asset: Hedge asset symbol
            hedge_price: Current price of hedge asset
        
        Returns:
            Tuple of (required_hedge_position, metadata)
        """
        # If no primary position, no hedge needed
        if primary_position == 0:
            return 0, {"reason": "no_primary_position"}
        
        # Get delta settings
        delta_settings = self.config.get_delta_settings()
        target_delta = delta_settings.get("portfolio_delta_target", 0.0)
        
        # Get correlation factor for this pair
        correlation_factors = self.config.get_correlation_factors(primary_asset)
        correlation_factor = correlation_factors[0]  # Default to first factor
        
        # Calculate delta of primary position (simplified)
        primary_delta = primary_position * primary_price
        
        # Calculate required hedge delta
        required_hedge_delta = -(primary_delta - target_delta)
        
        # Calculate hedge position size needed to achieve delta
        hedge_size = required_hedge_delta / (hedge_price * correlation_factor)
        
        # Apply max hedge ratio if configured
        max_hedge_ratio = self.config.get_hedge_settings().get("max_hedge_ratio", 1.0)
        if max_hedge_ratio < 1.0:
            hedge_size *= max_hedge_ratio
        
        # Apply minimum size threshold if needed
        min_hedge_size = self.config.get_hedge_pair_config(primary_asset).get("min_hedge_size", 0.01)
        if abs(hedge_size) < min_hedge_size and hedge_size != 0:
            hedge_size = min_hedge_size * math.copysign(1, hedge_size)
            
        return hedge_size, {
            "primary_delta": primary_delta,
            "hedge_delta": -hedge_size * hedge_price * correlation_factor,
            "correlation_factor": correlation_factor,
            "hedge_ratio": max_hedge_ratio
        }
    
    def should_rebalance(
        self, 
        primary_asset: str,
        primary_position: float,
        current_hedge_position: float,
        target_hedge_position: float
    ) -> bool:
        """
        Determine if hedge position should be rebalanced
        
        Args:
            primary_asset: Primary asset symbol
            primary_position: Current position in primary asset
            current_hedge_position: Current hedge position
            target_hedge_position: Target hedge position
            
        Returns:
            True if rebalance is needed, False otherwise
        """
        # If primary position is closed, close hedge
        if primary_position == 0:
            return current_hedge_position != 0
        
        # If no current hedge, create one
        if current_hedge_position == 0:
            return target_hedge_position != 0
        
        # Get delta threshold from config
        delta_settings = self.config.get_delta_settings()
        delta_threshold = delta_settings.get("delta_threshold", 0.01)
        
        # Calculate delta difference
        if target_hedge_position == 0:
            delta_diff = abs(current_hedge_position)
        else:
            delta_diff = abs(current_hedge_position - target_hedge_position) / abs(target_hedge_position)
        
        # Rebalance if delta difference exceeds threshold
        return delta_diff > delta_threshold


# Factory function to create appropriate strategy
def create_hedge_strategy(strategy_type: str, config: HedgeConfig) -> HedgeStrategy:
    """
    Factory function to create a hedge strategy
    
    Args:
        strategy_type: Type of strategy ("notional" or "delta_neutral")
        config: Hedge configuration
        
    Returns:
        HedgeStrategy implementation
    """
    if strategy_type == "delta_neutral":
        return DeltaNeutralHedgeStrategy(config)
    else:
        # Default to notional value strategy
        return NotionalValueHedgeStrategy(config) 