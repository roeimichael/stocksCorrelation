"""API endpoints for trade history."""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional

from api.services.trades_service import TradesService
from src.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
trades_service = TradesService()


@router.get("/")
async def get_trade_history(
    start_date: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD)"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    min_pnl: Optional[float] = Query(None, description="Minimum P&L filter"),
    max_pnl: Optional[float] = Query(None, description="Maximum P&L filter"),
):
    """
    Get trade history with optional filters.

    Returns comprehensive trade history with performance statistics.
    """
    try:
        trade_data = trades_service.get_trade_history(
            start_date=start_date,
            end_date=end_date,
            symbol=symbol,
            min_pnl=min_pnl,
            max_pnl=max_pnl,
        )

        return {
            "status": "success",
            "data": trade_data,
        }
    except Exception as e:
        logger.error(f"Error fetching trade history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recent")
async def get_recent_trades(
    n: int = Query(20, description="Number of recent trades to return", ge=1, le=100),
):
    """
    Get N most recent trades.

    Returns latest trades sorted by exit date descending.
    """
    try:
        recent_trades = trades_service.get_recent_trades(n=n)

        return {
            "status": "success",
            "count": len(recent_trades),
            "data": recent_trades,
        }
    except Exception as e:
        logger.error(f"Error fetching recent trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/by-symbol")
async def get_performance_by_symbol():
    """
    Get performance statistics grouped by symbol.

    Returns aggregate metrics for each symbol traded.
    """
    try:
        performance_data = trades_service.get_performance_by_symbol()

        return {
            "status": "success",
            "count": len(performance_data),
            "data": performance_data,
        }
    except Exception as e:
        logger.error(f"Error fetching performance by symbol: {e}")
        raise HTTPException(status_code=500, detail=str(e))
