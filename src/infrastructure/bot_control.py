import os
import signal
import time
import subprocess
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

PID_FILE = "/tmp/thalex_bot.pid"


class BotState(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


@dataclass
class BotStatus:
    state: BotState
    pid: Optional[int] = None
    uptime_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "pid": self.pid,
            "uptime_seconds": self.uptime_seconds,
        }


def _get_bot_pid() -> Optional[int]:
    try:
        if os.path.exists(PID_FILE):
            with open(PID_FILE, "r") as f:
                return int(f.read().strip())
    except (ValueError, IOError):
        pass
    return None


def _is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def get_bot_status() -> BotStatus:
    pid = _get_bot_pid()
    if pid and _is_process_running(pid):
        start_time = None
        try:
            stat_file = f"/proc/{pid}/stat"
            if os.path.exists(stat_file):
                with open(stat_file, "r") as f:
                    parts = f.read().split()
                    if len(parts) > 21:
                        boot_time = float(
                            open("/proc/stat").read().split("btime ")[1].split()[0]
                        )
                        start_ticks = int(parts[21])
                        hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
                        start_time = boot_time + (start_ticks / hz)
        except Exception:
            pass

        uptime = time.time() - start_time if start_time else None
        return BotStatus(state=BotState.RUNNING, pid=pid, uptime_seconds=uptime)

    if pid:
        try:
            os.unlink(PID_FILE)
        except OSError:
            pass

    return BotStatus(state=BotState.STOPPED)


def stop_bot() -> Dict[str, Any]:
    pid = _get_bot_pid()
    if not pid or not _is_process_running(pid):
        return {"success": False, "message": "Bot is not running", "pid": pid}

    try:
        os.kill(pid, signal.SIGTERM)
        logger.warning(f"KILL SWITCH: Sent SIGTERM to bot process {pid}")
        return {
            "success": True,
            "message": f"Stop signal sent to bot (PID: {pid})",
            "pid": pid,
        }
    except OSError as e:
        logger.error(f"Failed to send SIGTERM to {pid}: {e}")
        return {"success": False, "message": str(e), "pid": pid}


def force_kill_bot() -> Dict[str, Any]:
    pid = _get_bot_pid()
    if not pid or not _is_process_running(pid):
        return {"success": False, "message": "Bot is not running", "pid": pid}

    try:
        os.kill(pid, signal.SIGKILL)
        logger.warning(f"KILL SWITCH: Sent SIGKILL to bot process {pid}")
        return {
            "success": True,
            "message": f"Force kill signal sent to bot (PID: {pid})",
            "pid": pid,
        }
    except OSError as e:
        logger.error(f"Failed to send SIGKILL to {pid}: {e}")
        return {"success": False, "message": str(e), "pid": pid}
