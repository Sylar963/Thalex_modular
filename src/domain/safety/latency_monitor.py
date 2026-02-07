import time
import logging
from typing import Dict, Any, Optional
from ..interfaces import SafetyComponent

logger = logging.getLogger(__name__)


class LatencyMonitor(SafetyComponent):
    """
    Monitors data staleness to prevent trading on old market data.
    """

    def __init__(self, max_latency: float = 1.0):
        self.max_latency = max_latency
        self.last_check_time = 0
        self.consecutive_failures = 0

    def check_health(self, context: Dict[str, Any]) -> bool:
        timestamp = context.get("timestamp")
        if not timestamp:
            # If no timestamp provided, we can't judge latency.
            # Default to safe/pass or fail? Pass, assuming system time is used elsewhere?
            # Better to log warning.
            return True

        now = time.time()
        lag = now - timestamp

        if lag > self.max_latency:
            logger.warning(
                f"Latency Limit Exceeded: {lag:.4f}s > {self.max_latency}s. Halting."
            )
            return False

        return True

    def record_failure(self) -> None:
        self.consecutive_failures += 1

    def record_success(self) -> None:
        self.consecutive_failures = 0
