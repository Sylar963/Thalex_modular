import time
import logging
from typing import Dict, Any
from enum import Enum
from ..interfaces import SafetyComponent

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Tripped, no trading
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker(SafetyComponent):
    """
    Implements a classic Circuit Breaker pattern to halt trading
    after repeated failures or external kill signals.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        name: str = "Global",
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0

    def check_health(self, context: Dict[str, Any]) -> bool:
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                logger.info(
                    f"CircuitBreaker '{self.name}': Recovery timeout passed. Entering HALF_OPEN."
                )
                self.state = CircuitState.HALF_OPEN
                return True  # Allow one trial
            return False

        return True

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # If we fail in half-open, immediately reopen
            self.state = CircuitState.OPEN
            logger.critical(
                f"CircuitBreaker '{self.name}': Failed in HALF_OPEN. Re-opening circuit."
            )
        elif (
            self.state == CircuitState.CLOSED
            and self.failure_count >= self.failure_threshold
        ):
            self.state = CircuitState.OPEN
            logger.critical(
                f"CircuitBreaker '{self.name}': Tolerance exceeded ({self.failure_count}). TRIPPED."
            )

    def record_success(self) -> None:
        if self.state == CircuitState.HALF_OPEN:
            logger.info(
                f"CircuitBreaker '{self.name}': Recovery successful. Closing circuit."
            )
            self.state = CircuitState.CLOSED
            self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            # Gradually decay failure count on success?
            # For strict safety, maybe just reset or decrement.
            # Let's reset for simplicity in this version, effectively
            # requiring CONSECUTIVE failures to trip.
            self.failure_count = 0
