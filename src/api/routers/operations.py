"""API endpoints for running operations (daily workflow, backtest, etc.)."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional
from pydantic import BaseModel
from datetime import datetime
import sys

from src.core.logger import get_logger

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


def run_task_background(task_id: str, func, **kwargs):
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
        task_status[task_id].update({
            "status": "error",
            "completed_at": datetime.now().isoformat(),
            "error": str(e),
        })
        logger.error(f"Task {task_id} failed: {e}")


@router.post("/daily")
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
