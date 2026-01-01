"""API endpoints for correlation analysis."""

from fastapi import APIRouter, Query, HTTPException, Body
from typing import Optional, List
import json

from src.api.services.correlations_service import CorrelationsService
from src.api.services.positions_service import PositionsService
from src.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
correlations_service = CorrelationsService()
positions_service = PositionsService()


@router.get("/matrix")
async def get_correlation_matrix(
    symbols: Optional[str] = Query(None, description="Comma-separated list of symbols (e.g., 'AAPL,MSFT,GOOGL')"),
    lookback_days: int = Query(60, description="Number of days to look back", ge=10, le=365),
):
    """
    Get correlation matrix for specified symbols.

    If no symbols provided, uses current portfolio positions.
    Returns correlation matrix with top positive/negative pairs.
    """
    try:
        # Parse symbols if provided
        symbols_list = None
        if symbols:
            symbols_list = [s.strip().upper() for s in symbols.split(",")]

        corr_data = correlations_service.get_correlation_matrix(
            symbols=symbols_list, lookback_days=lookback_days
        )

        return {
            "status": "success",
            "data": corr_data,
        }
    except Exception as e:
        logger.error(f"Error computing correlation matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio")
async def get_portfolio_correlation(
    lookback_days: int = Query(60, description="Number of days to look back", ge=10, le=365),
):
    """
    Analyze correlation for current portfolio positions.

    Returns correlation matrix, diversification score, concentration risk,
    and suggested hedge candidates.
    """
    try:
        # Get current portfolio symbols
        positions_data = positions_service.get_all_positions()
        portfolio_symbols = [p["symbol"] for p in positions_data.get("positions", [])]

        if not portfolio_symbols:
            return {
                "status": "success",
                "data": {
                    "portfolio_symbols": [],
                    "correlation_matrix": {},
                    "diversification_score": 0.0,
                    "concentration_risk": 0.0,
                    "suggested_hedges": [],
                },
                "message": "No open positions in portfolio",
            }

        portfolio_corr = correlations_service.get_portfolio_correlation(
            positions_symbols=portfolio_symbols, lookback_days=lookback_days
        )

        return {
            "status": "success",
            "data": portfolio_corr,
        }
    except Exception as e:
        logger.error(f"Error analyzing portfolio correlation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pairs/{symbol1}/{symbol2}")
async def get_pair_correlation(
    symbol1: str,
    symbol2: str,
    lookback_days: int = Query(60, description="Number of days to look back", ge=10, le=365),
):
    """
    Get correlation between two specific symbols.

    Returns correlation coefficient and lookback period.
    """
    try:
        # Get correlation matrix for these two symbols
        corr_data = correlations_service.get_correlation_matrix(
            symbols=[symbol1.upper(), symbol2.upper()], lookback_days=lookback_days
        )

        if len(corr_data.get("symbols", [])) < 2:
            raise HTTPException(status_code=404, detail="One or both symbols not found in data")

        # Extract correlation value
        correlation = corr_data["matrix"][0][1]  # Off-diagonal element

        return {
            "status": "success",
            "data": {
                "symbol1": symbol1.upper(),
                "symbol2": symbol2.upper(),
                "correlation": correlation,
                "lookback_days": lookback_days,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing pair correlation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
