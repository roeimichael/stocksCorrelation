"""Pydantic models for trade history."""

from pydantic import BaseModel, Field
from typing import List, Optional


class Trade(BaseModel):
    """Individual trade model."""

    symbol: str = Field(..., description="Stock ticker symbol")
    entry_date: str = Field(..., description="Entry date")
    exit_date: str = Field(..., description="Exit date")
    entry_price: float = Field(..., description="Entry price")
    exit_price: float = Field(..., description="Exit price")
    shares: float = Field(..., description="Number of shares")
    pnl: float = Field(..., description="Realized P&L")
    pnl_pct: float = Field(..., description="Realized P&L percentage")
    signal: str = Field(..., description="Original signal (UP/DOWN)")
    exit_reason: str = Field(..., description="Reason for exit")
    confidence: Optional[float] = Field(None, description="Signal confidence")
    days_held: int = Field(..., description="Days held")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "entry_date": "2025-12-10",
                "exit_date": "2025-12-15",
                "entry_price": 195.50,
                "exit_price": 198.75,
                "shares": 50,
                "pnl": 162.50,
                "pnl_pct": 1.66,
                "signal": "UP",
                "exit_reason": "TARGET_HIT",
                "confidence": 0.75,
                "days_held": 5,
            }
        }


class TradeHistory(BaseModel):
    """Trade history with summary statistics."""

    total_trades: int = Field(..., description="Total number of trades")
    winning_trades: int = Field(..., description="Number of winning trades")
    losing_trades: int = Field(..., description="Number of losing trades")
    win_rate: float = Field(..., description="Win rate percentage")
    total_pnl: float = Field(..., description="Total realized P&L")
    avg_win: float = Field(..., description="Average winning trade")
    avg_loss: float = Field(..., description="Average losing trade")
    profit_factor: float = Field(..., description="Profit factor (gross profit / gross loss)")
    trades: List[Trade] = Field(..., description="List of trades")

    class Config:
        json_schema_extra = {
            "example": {
                "total_trades": 100,
                "winning_trades": 65,
                "losing_trades": 35,
                "win_rate": 65.0,
                "total_pnl": 12500.0,
                "avg_win": 350.0,
                "avg_loss": -180.0,
                "profit_factor": 1.94,
                "trades": [],
            }
        }
