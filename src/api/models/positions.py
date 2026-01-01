"""Pydantic models for positions."""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date


class Position(BaseModel):
    """Individual position model."""

    symbol: str = Field(..., description="Stock ticker symbol")
    entry_date: str = Field(..., description="Position entry date")
    entry_price: float = Field(..., description="Entry price")
    current_price: Optional[float] = Field(None, description="Current market price")
    shares: float = Field(..., description="Number of shares")
    notional: float = Field(..., description="Position notional value")
    pnl: Optional[float] = Field(None, description="Unrealized P&L")
    pnl_pct: Optional[float] = Field(None, description="Unrealized P&L percentage")
    signal: str = Field(..., description="Original signal (UP/DOWN)")
    confidence: float = Field(..., description="Signal confidence (0-1)")
    alert_level: Optional[str] = Field(None, description="Alert level (GREEN/YELLOW/RED)")
    days_held: Optional[int] = Field(None, description="Days position has been held")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "entry_date": "2025-12-15",
                "entry_price": 195.50,
                "current_price": 198.75,
                "shares": 50,
                "notional": 9775.0,
                "pnl": 162.50,
                "pnl_pct": 1.66,
                "signal": "UP",
                "confidence": 0.75,
                "alert_level": "GREEN",
                "days_held": 3,
            }
        }


class PositionsSummary(BaseModel):
    """Summary of all positions."""

    total_positions: int = Field(..., description="Total number of open positions")
    total_notional: float = Field(..., description="Total notional value")
    total_pnl: float = Field(..., description="Total unrealized P&L")
    total_pnl_pct: float = Field(..., description="Total unrealized P&L percentage")
    positions: List[Position] = Field(..., description="List of positions")
    alert_breakdown: dict = Field(..., description="Count by alert level")

    class Config:
        json_schema_extra = {
            "example": {
                "total_positions": 8,
                "total_notional": 80000.0,
                "total_pnl": 1250.50,
                "total_pnl_pct": 1.56,
                "positions": [],
                "alert_breakdown": {"GREEN": 5, "YELLOW": 2, "RED": 1},
            }
        }


class PositionResponse(BaseModel):
    """Response wrapper for positions."""

    status: str = Field("success", description="Response status")
    data: PositionsSummary = Field(..., description="Positions data")
