"""API endpoints for running operations (daily workflow, backtest, etc.)."""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Body
from typing import Optional
from pydantic import BaseModel
import subprocess
from datetime import datetime

from src.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# Request models
class DailyRunRequest(BaseModel):
    """Request model for daily workflow."""

    date: Optional[str] = None
    skip_signals: bool = False
    skip_monitor: bool = False
    skip_close: bool = False
    config: str = "config.yaml"


class BacktestRequest(BaseModel):
    """Request model for backtest."""

    config: str = "config.yaml"


class GridSearchRequest(BaseModel):
    """Request model for grid search."""

    config: str = "config.yaml"


class PreprocessRequest(BaseModel):
    """Request model for preprocessing."""

    config: str = "config.yaml"
    force_full: bool = False


# Background task status storage (in-memory for now)
task_status = {}


def run_command_background(task_id: str, command: list):
    """Run command in background and update task status."""
    try:
        task_status[task_id] = {
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "command": " ".join(command),
        }

        result = subprocess.run(command, capture_output=True, text=True, timeout=3600)

        task_status[task_id].update(
            {
                "status": "completed" if result.returncode == 0 else "failed",
                "completed_at": datetime.now().isoformat(),
                "returncode": result.returncode,
                "stdout": result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout,  # Last 1000 chars
                "stderr": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr,
            }
        )

    except subprocess.TimeoutExpired:
        task_status[task_id].update(
            {
                "status": "timeout",
                "completed_at": datetime.now().isoformat(),
                "error": "Command timed out after 1 hour",
            }
        )
    except Exception as e:
        task_status[task_id].update(
            {
                "status": "error",
                "completed_at": datetime.now().isoformat(),
                "error": str(e),
            }
        )


@router.post("/daily")
async def run_daily_workflow(request: DailyRunRequest, background_tasks: BackgroundTasks):
    """
    Run the daily trading workflow (signals + monitoring + closing).

    This endpoint triggers the main daily runner with optional step skipping.
    Operation runs in the background and returns a task ID for status tracking.
    """
    try:
        # Build command
        command = ["python", "-m", "src.run", "daily", "--config", request.config]

        if request.date:
            command.extend(["--date", request.date])
        if request.skip_signals:
            command.append("--skip-signals")
        if request.skip_monitor:
            command.append("--skip-monitor")
        if request.skip_close:
            command.append("--skip-close")

        # Generate task ID
        task_id = f"daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Run in background
        background_tasks.add_task(run_command_background, task_id, command)

        return {
            "status": "started",
            "task_id": task_id,
            "message": "Daily workflow started in background",
            "command": " ".join(command),
        }

    except Exception as e:
        logger.error(f"Error starting daily workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest")
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Run historical backtest.

    Operation runs in the background and returns a task ID for status tracking.
    """
    try:
        command = ["python", "-m", "src.run", "backtest", "--config", request.config]

        task_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        background_tasks.add_task(run_command_background, task_id, command)

        return {
            "status": "started",
            "task_id": task_id,
            "message": "Backtest started in background",
            "command": " ".join(command),
        }

    except Exception as e:
        logger.error(f"Error starting backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gridsearch")
async def run_gridsearch(request: GridSearchRequest, background_tasks: BackgroundTasks):
    """
    Run parameter grid search optimization.

    This is a long-running operation that tests multiple parameter combinations.
    Operation runs in the background and returns a task ID for status tracking.
    """
    try:
        command = ["python", "-m", "src.run", "gridsearch", "--config", request.config]

        task_id = f"gridsearch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        background_tasks.add_task(run_command_background, task_id, command)

        return {
            "status": "started",
            "task_id": task_id,
            "message": "Grid search started in background",
            "command": " ".join(command),
            "warning": "This operation may take a long time (30+ minutes)",
        }

    except Exception as e:
        logger.error(f"Error starting grid search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preprocess")
async def run_preprocess(request: PreprocessRequest, background_tasks: BackgroundTasks):
    """
    Run data preprocessing (download and prepare data).

    Operation runs in the background and returns a task ID for status tracking.
    """
    try:
        command = ["python", "-m", "src.run", "preprocess", "--config", request.config]

        if request.force_full:
            command.append("--force-full")

        task_id = f"preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        background_tasks.add_task(run_command_background, task_id, command)

        return {
            "status": "started",
            "task_id": task_id,
            "message": "Preprocessing started in background",
            "command": " ".join(command),
        }

    except Exception as e:
        logger.error(f"Error starting preprocessing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Get status of a background task.

    Returns current status, output, and error information for the specified task.
    """
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    return {
        "task_id": task_id,
        **task_status[task_id],
    }


@router.get("/tasks")
async def list_tasks():
    """
    List all background tasks and their statuses.

    Returns summary of all tasks (running, completed, failed).
    """
    return {
        "status": "success",
        "total_tasks": len(task_status),
        "tasks": task_status,
    }
