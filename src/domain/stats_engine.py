import logging
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .entities.pnl import FillEffect

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AlphaMetrics:
    avg_adverse_selection_5s: float = 0.0
    avg_adverse_selection_30s: float = 0.0
    avg_edge: float = 0.0
    vamp_correlation: float = 0.0
    total_fills: int = 0


class StatsEngine:
    def __init__(self, intervals: List[int] = [5, 30, 60]):
        self.intervals = intervals
        self.fills_with_vamp: List[Dict[str, Any]] = []
        self.metrics = AlphaMetrics()

    def record_fill(
        self, fill: FillEffect, mid_price: float, vamp: Optional[float] = None
    ):
        self.fills_with_vamp.append(
            {
                "fill": fill,
                "mid_at_fill": mid_price,
                "vamp": vamp,
                "ts": fill.timestamp,
                "future_mids": {},
            }
        )
        self.metrics.total_fills += 1

    def update_future_mids(self, current_ts: float, current_mid: float):
        for data in self.fills_with_vamp:
            elapsed = current_ts - data["ts"]
            for interval in self.intervals:
                if interval not in data["future_mids"] and elapsed >= interval:
                    data["future_mids"][interval] = current_mid

    def calculate_alpha(self) -> AlphaMetrics:
        if not self.fills_with_vamp:
            return self.metrics

        adverse_5s = []
        adverse_30s = []
        edges = []

        for data in self.fills_with_vamp:
            fill = data["fill"]
            mid_at_fill = data["mid_at_fill"]
            side_sign = 1 if fill.side.lower() == "buy" else -1

            # Edge = (Mid - FillPrice) * Sign
            edge = (mid_at_fill - fill.price) * side_sign
            edges.append(edge)

            # Adverse Selection = (Mid_Future - Mid_At_Fill) * Sign
            # A positive value means price moved against us (adverse)
            if 5 in data["future_mids"]:
                adv = (data["future_mids"][5] - mid_at_fill) * (
                    -side_sign
                )  # Reversed for adverse selection
                adverse_5s.append(adv)
            if 30 in data["future_mids"]:
                adv = (data["future_mids"][30] - mid_at_fill) * (-side_sign)
                adverse_30s.append(adv)

        self.metrics.avg_edge = np.mean(edges) if edges else 0.0
        self.metrics.avg_adverse_selection_5s = (
            np.mean(adverse_5s) if adverse_5s else 0.0
        )
        self.metrics.avg_adverse_selection_30s = (
            np.mean(adverse_30s) if adverse_30s else 0.0
        )

        return self.metrics
