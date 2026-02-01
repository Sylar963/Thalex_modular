from fastapi import APIRouter, Depends, Body
from typing import Dict, Any
from ...repositories import ConfigRepository
from ...dependencies import get_config_repo

router = APIRouter()


@router.get("", response_model=Dict[str, Any])
async def get_config(repo: ConfigRepository = Depends(get_config_repo)):
    """Get current bot parameters."""
    return await repo.get_config()


@router.patch("", response_model=Dict[str, Any])
async def update_config(
    updates: Dict[str, Any] = Body(...),
    repo: ConfigRepository = Depends(get_config_repo),
):
    """Update bot parameters."""
    return await repo.update_config(updates)
