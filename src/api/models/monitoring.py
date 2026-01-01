"""Pydantic models for monitoring and alerts."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class MonitoringMetrics(BaseModel):
    """Position monitoring metrics."""

    symbol: str = Field(..., description="Stock ticker symbol")
    similarity_retention: Optional[float] = Field(None, description="Similarity retention (0-1)")
    directional_concordance: Optional[float] = Field(None, description="Directional concordance (0-1)")
    correlation_decay: Optional[float] = Field(None, description="Correlation decay")
    pattern_deviation_z: Optional[float] = Field(None, description="Pattern deviation Z-score")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "similarity_retention": 0.85,
                "directional_concordance": 0.72,
                "correlation_decay": -0.15,
                "pattern_deviation_z": 0.5,
            }
        }


class Alert(BaseModel):
    """Position alert model."""

    symbol: str = Field(..., description="Stock ticker symbol")
    alert_level: str = Field(..., description="Alert level (GREEN/YELLOW/RED/ERROR)")
    date: str = Field(..., description="Alert date")
    metrics: Optional[MonitoringMetrics] = Field(None, description="Monitoring metrics")
    message: str = Field(..., description="Alert message")
    recommended_action: Optional[str] = Field(None, description="Recommended action")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "alert_level": "YELLOW",
                "date": "2025-12-31",
                "metrics": None,
                "message": "Similarity retention below threshold",
                "recommended_action": "Consider reducing position by 50%",
            }
        }


class AlertsSummary(BaseModel):
    """Summary of all alerts."""

    date: str = Field(..., description="Alert date")
    total_alerts: int = Field(..., description="Total number of alerts")
    green: int = Field(0, description="Number of GREEN alerts")
    yellow: int = Field(0, description="Number of YELLOW alerts")
    red: int = Field(0, description="Number of RED alerts")
    error: int = Field(0, description="Number of ERROR alerts")
    alerts: List[Alert] = Field(..., description="List of alerts")

    class Config:
        json_schema_extra = {
            "example": {
                "date": "2025-12-31",
                "total_alerts": 10,
                "green": 6,
                "yellow": 3,
                "red": 1,
                "error": 0,
                "alerts": [],
            }
        }
