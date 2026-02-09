from fastapi import APIRouter, HTTPException
from typing import Dict
from src.infrastructure.bot_control import get_bot_status, stop_bot, force_kill_bot

router = APIRouter()


@router.get("/status", response_model=Dict)
async def get_status():
    status = get_bot_status()
    return status.to_dict()


@router.post("/stop", response_model=Dict)
async def stop():
    result = stop_bot()
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@router.post("/kill", response_model=Dict)
async def kill():
    result = force_kill_bot()
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result
