"""API endpoints for position management."""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional

from api.services.positions_service import PositionsService
from src.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
positions_service = PositionsService()


@router.get("/")
async def get_positions(
    sort_by: Optional[str] = Query(None, description="Sort field: pnl, pnl_pct, days_held, symbol, confidence"),
    filter_alert: Optional[str] = Query(None, description="Filter by alert level: GREEN, YELLOW, RED"),
):
    """
    Get all open positions with optional sorting and filtering.

    Returns organized view of current investments with P&L, alerts, and metrics.
    """
    try:
        positions_data = positions_service.get_all_positions(sort_by=sort_by, filter_alert=filter_alert)
        return {
            "status": "success",
            "data": positions_data,
        }
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}")
async def get_position_by_symbol(symbol: str):
    """
    Get detailed information for a specific position.

    Includes full analog history and detailed metrics.
    """
    try:
        position = positions_service.get_position_by_symbol(symbol.upper())

        if position is None:
            raise HTTPException(status_code=404, detail=f"Position not found for symbol: {symbol}")

        return {
            "status": "success",
            "data": position,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching position for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary/stats")
async def get_positions_summary():
    """
    Get summary statistics for all positions.

    Returns high-level portfolio metrics without individual position details.
    """
    try:
        positions_data = positions_service.get_all_positions()

        # Return summary only (without individual positions)
        summary = {
            "total_positions": positions_data["total_positions"],
            "total_notional": positions_data["total_notional"],
            "total_pnl": positions_data["total_pnl"],
            "total_pnl_pct": positions_data["total_pnl_pct"],
            "alert_breakdown": positions_data["alert_breakdown"],
        }

        return {
            "status": "success",
            "data": summary,
        }
    except Exception as e:
        logger.error(f"Error fetching positions summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
