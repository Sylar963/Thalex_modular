import asyncio
import logging
import time
from typing import Dict, List, Optional
from ..interfaces import TimeSyncManager, ExchangeGateway

logger = logging.getLogger(__name__)


class MultiVenueTimeSyncService(TimeSyncManager):
    """
    Modular engine that manages clock synchronization across multiple exchanges.
    Calculates offsets once every few minutes to ensure high-performance execution
    without constant REST overhead.
    """

    def __init__(
        self,
        exchanges: Optional[List[ExchangeGateway]] = None,
        sync_interval: int = 300,
    ):
        self.exchanges: Dict[str, ExchangeGateway] = (
            {ex.name: ex for ex in exchanges} if exchanges else {}
        )
        self.offsets: Dict[str, int] = {}  # exchange_name -> offset_ms
        self.sync_interval = sync_interval
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False

    def add_venue(self, gate: ExchangeGateway):
        """Dynamically add a venue to the sync manager."""
        self.exchanges[gate.name] = gate

    def get_timestamp(self, exchange: str) -> int:
        """Standardized method to get exchange-aligned timestamp in milliseconds."""
        offset = self.offsets.get(exchange, 0)
        return int(time.time() * 1000) + offset

    async def sync_all(self):
        """Proactively sync with all registered exchanges."""
        logger.info(
            f"TimeSync: Triggering global sync for {list(self.exchanges.keys())}"
        )
        tasks = [self._sync_venue(name) for name in self.exchanges.keys()]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _sync_venue(self, name: str):
        """Sync with a specific venue and update internal offset."""
        ex = self.exchanges.get(name)
        if not ex:
            return

        try:
            # Capturing local time just before/after to account for round-trip half-time
            t0 = int(time.time() * 1000)
            server_time = await ex.get_server_time()
            t1 = int(time.time() * 1000)

            # Simple NTP-style offset: server_time - (local_time_at_receive - round_trip/2)
            # For trading, usually just (server_time - local_time) is used but this is better.
            local_estimate = (t0 + t1) // 2
            # Offset = Server - Local
            # Added -11000ms extra buffer because local clock is consistently ~10s ahead of Bybit matching engine
            offset = server_time - local_estimate - 11000

            
            self.offsets[name] = offset




            logger.info(
                f"TimeSync: {name} offset matched: {offset}ms (Drift: {t1 - t0}ms RTT)"
            )
        except Exception as e:
            logger.warning(f"TimeSync: Failed to sync with {name}: {e}")

    async def start_sync_loop(self):
        """Starts the background heart-beat for time sync."""
        if self._running:
            return

        self._running = True
        self._sync_task = asyncio.create_task(self._run_loop())
        logger.info("TimeSync: Background sync loop started.")

    async def _run_loop(self):
        while self._running:
            await self.sync_all()
            await asyncio.sleep(self.sync_interval)

    def stop(self):
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
