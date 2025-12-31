"""Pydantic models for correlation analysis."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class CorrelationPair(BaseModel):
    """Correlation between two symbols."""

    symbol1: str = Field(..., description="First symbol")
    symbol2: str = Field(..., description="Second symbol")
    correlation: float = Field(..., description="Correlation coefficient (-1 to 1)")
    lookback_days: int = Field(..., description="Lookback period in days")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol1": "AAPL",
                "symbol2": "MSFT",
                "correlation": 0.85,
                "lookback_days": 60,
            }
        }


class CorrelationMatrix(BaseModel):
    """Correlation matrix for multiple symbols."""

    symbols: List[str] = Field(..., description="List of symbols")
    matrix: List[List[float]] = Field(..., description="Correlation matrix (NxN)")
    lookback_days: int = Field(..., description="Lookback period in days")
    top_positive: List[CorrelationPair] = Field(..., description="Top positively correlated pairs")
    top_negative: List[CorrelationPair] = Field(..., description="Top negatively correlated pairs")

    class Config:
        json_schema_extra = {
            "example": {
                "symbols": ["AAPL", "MSFT", "GOOGL"],
                "matrix": [[1.0, 0.85, 0.72], [0.85, 1.0, 0.78], [0.72, 0.78, 1.0]],
                "lookback_days": 60,
                "top_positive": [],
                "top_negative": [],
            }
        }


class PortfolioCorrelation(BaseModel):
    """Correlation analysis for current portfolio."""

    portfolio_symbols: List[str] = Field(..., description="Symbols in current portfolio")
    correlation_matrix: CorrelationMatrix = Field(..., description="Correlation matrix")
    diversification_score: float = Field(..., description="Portfolio diversification score (0-1)")
    concentration_risk: float = Field(..., description="Concentration risk score (0-1)")
    suggested_hedges: List[Dict] = Field(..., description="Suggested hedge symbols")

    class Config:
        json_schema_extra = {
            "example": {
                "portfolio_symbols": ["AAPL", "MSFT", "GOOGL"],
                "correlation_matrix": {},
                "diversification_score": 0.65,
                "concentration_risk": 0.35,
                "suggested_hedges": [{"symbol": "SPY", "correlation": -0.25}],
            }
        }
