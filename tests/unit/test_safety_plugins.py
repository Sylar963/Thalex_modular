import pytest
import time
from unittest.mock import MagicMock
from src.domain.safety.latency_monitor import LatencyMonitor
from src.domain.safety.circuit_breaker import CircuitBreaker, CircuitState


class TestLatencyMonitor:
    def test_latency_check_pass(self):
        monitor = LatencyMonitor(max_latency=1.0)
        now = time.time()
        # timestamp is 0.5s ago (within 1.0s limit)
        context = {"timestamp": now - 0.5}
        assert monitor.check_health(context) is True

    def test_latency_check_fail(self):
        monitor = LatencyMonitor(max_latency=1.0)
        now = time.time()
        # timestamp is 1.5s ago (exceeds 1.0s limit)
        context = {"timestamp": now - 1.5}
        assert monitor.check_health(context) is False

    def test_no_timestamp_ignored(self):
        monitor = LatencyMonitor(max_latency=1.0)
        context = {}
        # Should default to True if no timestamp
        assert monitor.check_health(context) is True


class TestCircuitBreaker:
    def test_circuit_breaker_trip(self):
        # Threshold of 3
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
        assert cb.state == CircuitState.CLOSED

        # 2 failures -> still closed
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        # 3rd failure -> trips to OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_circuit_breaker_wont_trade_when_open(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.check_health({}) is False

    def test_circuit_breaker_recovery(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.2)

        # Check health -> transitions to HALF_OPEN
        assert cb.check_health({}) is True
        assert cb.state == CircuitState.HALF_OPEN

        # Success -> transitions to CLOSED
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_circuit_breaker_fail_in_half_open(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        cb.record_failure()
        time.sleep(0.2)
        cb.check_health({})  # Enters HALF_OPEN

        # Failure -> immediatley OPEN again
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
