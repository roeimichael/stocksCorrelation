"""API endpoints for trading signals."""

from fastapi import APIRouter, Query, HTTPException, Path

from src.api.services.signals_service import SignalsService
from src.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
signals_service = SignalsService()


@router.get("/latest")
async def get_latest_signals():
    """
    Get the most recent trading signals.

    Returns latest signal generation with all signals and summary statistics.
    """
    try:
        signals_data = signals_service.get_latest_signals()

        return signals_data
    except Exception as e:
        logger.error(f"Error fetching latest signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/date/{date}")
async def get_signals_by_date(
    date: str = Path(..., description="Date in YYYY-MM-DD format"),
):
    """
    Get signals for a specific date.

    Returns all signals generated on the specified date.
    """
    try:
        signals_data = signals_service.get_signals_by_date(date=date)

        return signals_data
    except Exception as e:
        logger.error(f"Error fetching signals for date {date}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{symbol}")
async def get_signal_history(
    symbol: str = Path(..., description="Stock ticker symbol"),
    days: int = Query(30, description="Number of days to look back", ge=1, le=365),
):
    """
    Get signal history for a specific symbol.

    Returns chronological signal history for the symbol over the specified period.
    """
    try:
        history = signals_service.get_signal_history(symbol=symbol, days=days)

        return {
            "status": "success",
            "symbol": symbol.upper(),
            "lookback_days": days,
            "count": len(history),
            "data": history,
        }
    except Exception as e:
        logger.error(f"Error fetching signal history for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
