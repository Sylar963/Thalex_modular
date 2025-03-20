import logging
from typing import List, Optional, Tuple, Dict
import time
import thalex as th

from ..config.market_config import ORDERBOOK_CONFIG, TRADING_PARAMS, TECHNICAL_PARAMS
from ..models.data_models import Ticker, Quote
from .technical_analysis import TechnicalAnalysis
from .market_maker import MarketMaker

class QuoteManager:
    def __init__(self):
        self.tick: Optional[float] = None
        self.technical_analysis = TechnicalAnalysis()
        self.market_maker = MarketMaker()
        self.last_quote_time = 0.0
        self.quote_history: List[Dict] = []

    def set_tick_size(self, tick: float) -> None:
        """Set the tick size for price rounding"""
        if tick <= 0:
            raise ValueError("Tick size must be positive")
        self.tick = tick
        self.market_maker.set_tick_size(tick)

    def round_to_tick(self, value: float) -> float:
        """Round price to valid tick size"""
        if not self.tick:
            raise ValueError("Tick size not set")
        return self.tick * round(value / self.tick)

    def update_market_data(self, price: float, volume: Optional[float] = None):
        """Update market data in technical analysis"""
        self.technical_analysis.update_price(price, volume)

    def update_position(self, size: float, price: float):
        """Update position information in market maker"""
        self.market_maker.update_position(size, price)

    def make_quotes(
        self,
        ticker: Ticker,
        position_size: float,
        max_position: float,
        volatility: float
    ) -> List[List[th.SideQuote]]:
        """Generate quotes using advanced market making logic"""
        try:
            # Get market conditions
            market_conditions = self.technical_analysis.get_market_conditions(ticker.mark_price)
            
            # Update market maker state
            self.market_maker.update_market_conditions(
                volatility=volatility,
                market_impact=market_conditions["atr"] / ticker.mark_price
            )
            
            # Generate quotes
            bid_quotes, ask_quotes = self.market_maker.generate_quotes(
                mid_price=ticker.mark_price,
                market_conditions=market_conditions
            )
            
            # Convert to Thalex SideQuote format
            bids = [th.SideQuote(price=q.price, amount=q.amount) for q in bid_quotes]
            asks = [th.SideQuote(price=q.price, amount=q.amount) for q in ask_quotes]
            
            # Store quote history
            self.quote_history.append({
                "timestamp": time.time(),
                "mid_price": ticker.mark_price,
                "best_bid": bids[0].p if bids else None,
                "best_ask": asks[0].p if asks else None,
                "market_conditions": market_conditions
            })
            
            # Maintain history size
            if len(self.quote_history) > 1000:
                self.quote_history = self.quote_history[-1000:]
            
            return [bids, asks]
            
        except Exception as e:
            logging.error(f"Error making quotes: {str(e)}")
            return [[], []]

    def validate_quotes(self, quotes: List[List[th.SideQuote]]) -> bool:
        """Validate generated quotes with enhanced checks"""
        try:
            if not quotes or len(quotes) != 2:
                logging.error("Invalid quote structure")
                return False
                
            bids, asks = quotes
            if not bids or not asks:
                logging.error("Empty quotes")
                return False
                
            # Get current market conditions
            market_conditions = self.technical_analysis.get_market_conditions(
                (bids[0].p + asks[0].p) / 2
            )
            
            # Validate using market maker
            return self.market_maker.validate_quotes(
                [Quote(price=q.p, amount=q.a) for q in bids],
                [Quote(price=q.p, amount=q.a) for q in asks],
                market_conditions
            )
            
        except Exception as e:
            logging.error(f"Quote validation error: {str(e)}")
            return False

    def calculate_quote_imbalance(self, quotes: List[List[th.SideQuote]]) -> float:
        """Calculate quote imbalance with volume weighting"""
        try:
            if not quotes or len(quotes) != 2:
                return 0.0
                
            bids, asks = quotes
            if not bids or not asks:
                return 0.0
                
            # Calculate volume-weighted sizes
            bid_sizes = sum(q.a * (1.0 - i * 0.1) for i, q in enumerate(bids))
            ask_sizes = sum(q.a * (1.0 - i * 0.1) for i, q in enumerate(asks))
            
            total_size = bid_sizes + ask_sizes
            if total_size == 0:
                return 0.0
                
            return (bid_sizes - ask_sizes) / total_size
            
        except Exception as e:
            logging.error(f"Error calculating quote imbalance: {str(e)}")
            return 0.0

    def should_update_quotes(
        self,
        current_quotes: List[List[th.SideQuote]],
        ticker: Ticker
    ) -> bool:
        """Determine if quotes should be updated"""
        if not current_quotes or len(current_quotes) != 2:
            return True
            
        bids, asks = current_quotes
        if not bids or not asks:
            return True
            
        return self.market_maker.should_update_quotes(
            ([Quote(price=q.p, amount=q.a) for q in bids],
             [Quote(price=q.p, amount=q.a) for q in asks]),
            ticker.mark_price
        )

    def get_quote_analytics(self) -> Dict:
        """Get analytics about current quoting performance"""
        try:
            if not self.quote_history:
                return {}
                
            recent_quotes = self.quote_history[-100:]  # Last 100 quotes
            
            spreads = [
                q["best_ask"] - q["best_bid"]
                for q in recent_quotes
                if q["best_ask"] is not None and q["best_bid"] is not None
            ]
            
            mid_prices = [q["mid_price"] for q in recent_quotes]
            
            return {
                "avg_spread": sum(spreads) / len(spreads) if spreads else 0,
                "min_spread": min(spreads) if spreads else 0,
                "max_spread": max(spreads) if spreads else 0,
                "price_volatility": self.technical_analysis.calculate_volatility(mid_prices),
                "quote_count": len(recent_quotes),
                "market_conditions": recent_quotes[-1]["market_conditions"] if recent_quotes else {},
                "quote_performance": self.market_maker.quote_performance
            }
            
        except Exception as e:
            logging.error(f"Error getting quote analytics: {str(e)}")
            return {}
