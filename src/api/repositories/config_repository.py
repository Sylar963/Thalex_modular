from typing import Dict, Any
from .base_repository import BaseRepository


class ConfigRepository(BaseRepository):
    # In-memory store for demo purposes,
    # in production this would sync with DB or Redis
    _config = {
        "spread_factor": 1.0,
        "hedging_enabled": False,
        "max_position": 10.0,
        "model_params": {"gamma": 0.5, "kappa": 2.0},
    }

    async def get_config(self) -> Dict[str, Any]:
        return self._config

    async def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        self._config.update(updates)
        return self._config
