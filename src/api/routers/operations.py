"""API endpoints for running operations (daily workflow, backtest, etc.)."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional
from pydantic import BaseModel
from datetime import datetime
import sys
import traceback

from src.core.logger import get_logger
from src.core.error_decorator import log_errors_to_file

logger = get_logger(__name__)
router = APIRouter()


class DailyRunRequest(BaseModel):
    date: Optional[str] = None
    skip_signals: bool = False
    skip_monitor: bool = False
    skip_close: bool = False
    config: str = "config.yaml"


class BacktestRequest(BaseModel):
    config: str = "config.yaml"


class GridSearchRequest(BaseModel):
    config: str = "config.yaml"


class PreprocessRequest(BaseModel):
    config: str = "config.yaml"
    force_full: bool = False


task_status = {}


def _format_kwargs(kwargs: dict) -> str:
    """Format kwargs for error logging."""
    if not kwargs:
        return "  (no kwargs)"

    formatted = []
    for key, value in kwargs.items():
        try:
            # Limit string length
            value_str = str(value)
            if len(value_str) > 200:
                value_str = value_str[:200] + f"... (truncated, length: {len(value_str)})"
            formatted.append(f"  {key}: {value_str}")
        except Exception:
            formatted.append(f"  {key}: <could not serialize>")

    return '\n'.join(formatted)


def run_task_background(task_id: str, func, **kwargs):
    """Run a task in the background with detailed error logging."""
    from pathlib import Path

    try:
        task_status[task_id] = {
            "status": "running",
            "started_at": datetime.now().isoformat(),
        }

        func(**kwargs)

        task_status[task_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
        })
    except Exception as e:
        # Get detailed error information
        error_msg = str(e)
        error_type = type(e).__name__
        stack_trace = traceback.format_exc()

        # Update task status
        task_status[task_id].update({
            "status": "error",
            "completed_at": datetime.now().isoformat(),
            "error": error_msg,
            "error_type": error_type,
        })

        # Log to console
        logger.error(f"Task {task_id} failed: {e}")

        # Log detailed error to file
        log_path = Path("logs/api_errors.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        log_entry = f"""
{'='*100}
TIMESTAMP:   {timestamp}
TASK ID:     {task_id}
FUNCTION:    {func.__module__}.{func.__name__}
ERROR TYPE:  {error_type}
ERROR MSG:   {error_msg}

KWARGS:
{_format_kwargs(kwargs)}

FULL STACK TRACE:
{stack_trace}
{'='*100}

"""

        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as log_error:
            logger.error(f"Failed to write error log: {log_error}")


@router.post("/daily")
@log_errors_to_file()
async def run_daily_workflow(request: DailyRunRequest, background_tasks: BackgroundTasks):
    from src.cli.daily_runner import main as daily_runner_main

    task_id = f"daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    sys.argv = ["daily_runner", "--config", request.config]
    if request.date:
        sys.argv.extend(["--date", request.date])
    if request.skip_signals:
        sys.argv.append("--skip-signals")
    if request.skip_monitor:
        sys.argv.append("--skip-monitor")
    if request.skip_close:
        sys.argv.append("--skip-close")

    background_tasks.add_task(run_task_background, task_id, daily_runner_main)

    return {
        "status": "started",
        "task_id": task_id,
        "message": "Daily workflow started",
    }


@router.post("/backtest")
@log_errors_to_file()
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    from src.cli.backtest import main as backtest_main

    task_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    sys.argv = ["backtest", "--config", request.config]

    background_tasks.add_task(run_task_background, task_id, backtest_main)

    return {
        "status": "started",
        "task_id": task_id,
        "message": "Backtest started",
    }


@router.post("/gridsearch")
@log_errors_to_file()
async def run_gridsearch(request: GridSearchRequest, background_tasks: BackgroundTasks):
    from src.cli.gridsearch import main as gridsearch_main

    task_id = f"gridsearch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    sys.argv = ["gridsearch", "--config", request.config]

    background_tasks.add_task(run_task_background, task_id, gridsearch_main)

    return {
        "status": "started",
        "task_id": task_id,
        "message": "Grid search started",
    }


@router.post("/preprocess")
@log_errors_to_file()
async def run_preprocess(request: PreprocessRequest, background_tasks: BackgroundTasks):
    from src.cli.preprocess import main as preprocess_main

    task_id = f"preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    sys.argv = ["preprocess", "--config", request.config]
    if request.force_full:
        sys.argv.append("--force-full")

    background_tasks.add_task(run_task_background, task_id, preprocess_main)

    return {
        "status": "started",
        "task_id": task_id,
        "message": "Preprocessing started",
    }


@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    return {
        "task_id": task_id,
        **task_status[task_id],
    }


@router.get("/tasks")
async def list_tasks():
    return {
        "status": "success",
        "total_tasks": len(task_status),
        "tasks": task_status,
    }
