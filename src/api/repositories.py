import os
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional
from datetime import datetime


class MetricsRepository:
    def __init__(self):
        self.host = os.getenv("DATABASE_HOST", "localhost")
        self.name = os.getenv("DATABASE_NAME", "thalex_trading")
        self.user = os.getenv("DATABASE_USER", "postgres")
        self.password = os.getenv("DATABASE_PASSWORD", "password")
        self.port = os.getenv("DATABASE_PORT", "5433")

    def _get_conn(self):
        return psycopg2.connect(
            host=self.host,
            database=self.name,
            user=self.user,
            password=self.password,
            port=self.port,
        )

    def get_latest_metrics(self, limit: int = 100) -> List[Dict]:
        """Fetch latest market metrics."""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM options_live_metrics 
                    ORDER BY time DESC 
                    LIMIT %s
                    """,
                    (limit,),
                )
                return cur.fetchall()
        finally:
            conn.close()

    def get_simulation_report(self) -> Dict:
        """
        Placeholder: logic to aggregate simulation results.
        For now, returns a mock summary or reads from a latest run log table if available.
        """
        # TODO: Implement concrete simulation result storage so we can query it.
        # For now, we'll return a static structure.
        return {
            "status": "simulation_ended",
            "pnl": 1250.50,
            "sharpe": 1.45,
            "max_drawdown": -250.00,
            "trades": 45,
        }
