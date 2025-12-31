"""Pydantic models for trading signals."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class Signal(BaseModel):
    """Individual trading signal."""

    symbol: str = Field(..., description="Stock ticker symbol")
    signal: str = Field(..., description="Signal direction (UP/DOWN/ABSTAIN)")
    confidence: float = Field(..., description="Signal confidence (0-1)")
    p_up: float = Field(..., description="Probability of up move")
    p_down: float = Field(..., description="Probability of down move")
    n_analogs: int = Field(..., description="Number of analogs used")
    date: str = Field(..., description="Signal date")
    analogs: Optional[List[Dict]] = Field(None, description="Analog patterns used")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "signal": "UP",
                "confidence": 0.75,
                "p_up": 0.75,
                "p_down": 0.25,
                "n_analogs": 25,
                "date": "2025-12-31",
                "analogs": [],
            }
        }


class SignalResponse(BaseModel):
    """Response wrapper for signals."""

    status: str = Field("success", description="Response status")
    date: str = Field(..., description="Signal generation date")
    total_signals: int = Field(..., description="Total number of signals")
    up_signals: int = Field(..., description="Number of UP signals")
    down_signals: int = Field(..., description="Number of DOWN signals")
    abstain_signals: int = Field(..., description="Number of ABSTAIN signals")
    signals: List[Signal] = Field(..., description="List of signals")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "date": "2025-12-31",
                "total_signals": 50,
                "up_signals": 12,
                "down_signals": 8,
                "abstain_signals": 30,
                "signals": [],
            }
        }
