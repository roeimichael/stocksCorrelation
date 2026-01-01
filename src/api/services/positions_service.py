"""Service for managing positions."""

import json
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime, date
import pandas as pd

from src.core.logger import get_logger
from src.core.constants import Paths

logger = get_logger(__name__)


class PositionsService:
    """Service for managing and querying positions."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.positions_file = Paths.POSITIONS_STATE
        self.ledger_file = Paths.PORTFOLIO_LEDGER

    def get_all_positions(self, sort_by: Optional[str] = None, filter_alert: Optional[str] = None) -> Dict:
        """
        Get all open positions with optional sorting and filtering.

        Args:
            sort_by: Sort field (pnl, pnl_pct, days_held, symbol, confidence)
            filter_alert: Filter by alert level (GREEN, YELLOW, RED)

        Returns:
            Dictionary with positions summary
        """
        try:
            # Load positions state
            if not self.positions_file.exists():
                logger.warning(f"Positions file not found: {self.positions_file}")
                return self._empty_response()

            with open(self.positions_file, "r") as f:
                positions_data = json.load(f)

            if not positions_data:
                return self._empty_response()

            # Convert to list of dicts
            positions = []
            for symbol, pos in positions_data.items():
                position_dict = {
                    "symbol": symbol,
                    "entry_date": pos.get("entry_date", ""),
                    "entry_price": pos.get("entry_price", 0.0),
                    "current_price": pos.get("current_price"),
                    "shares": pos.get("shares", 0.0),
                    "notional": pos.get("notional", 0.0),
                    "pnl": pos.get("pnl", 0.0),
                    "pnl_pct": pos.get("pnl_pct", 0.0),
                    "signal": pos.get("signal", ""),
                    "confidence": pos.get("confidence", 0.0),
                    "alert_level": pos.get("alert_level", "UNKNOWN"),
                    "days_held": self._calculate_days_held(pos.get("entry_date")),
                }
                positions.append(position_dict)

            # Filter by alert level if specified
            if filter_alert:
                positions = [p for p in positions if p["alert_level"] == filter_alert.upper()]

            # Sort if specified
            if sort_by and sort_by in ["pnl", "pnl_pct", "days_held", "symbol", "confidence"]:
                reverse = sort_by != "symbol"  # Ascending for symbol, descending for others
                positions.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse)

            # Calculate summary statistics
            total_notional = sum(p["notional"] for p in positions)
            total_pnl = sum(p["pnl"] for p in positions if p["pnl"] is not None)
            total_pnl_pct = (total_pnl / total_notional * 100) if total_notional > 0 else 0.0

            # Alert breakdown
            alert_breakdown = {}
            for p in positions:
                level = p["alert_level"]
                alert_breakdown[level] = alert_breakdown.get(level, 0) + 1

            return {
                "total_positions": len(positions),
                "total_notional": total_notional,
                "total_pnl": total_pnl,
                "total_pnl_pct": total_pnl_pct,
                "positions": positions,
                "alert_breakdown": alert_breakdown,
            }

        except Exception as e:
            logger.error(f"Error loading positions: {e}")
            return self._empty_response()

    def get_position_by_symbol(self, symbol: str) -> Optional[Dict]:
        """Get a specific position by symbol."""
        try:
            if not self.positions_file.exists():
                return None

            with open(self.positions_file, "r") as f:
                positions_data = json.load(f)

            if symbol not in positions_data:
                return None

            pos = positions_data[symbol]
            return {
                "symbol": symbol,
                "entry_date": pos.get("entry_date", ""),
                "entry_price": pos.get("entry_price", 0.0),
                "current_price": pos.get("current_price"),
                "shares": pos.get("shares", 0.0),
                "notional": pos.get("notional", 0.0),
                "pnl": pos.get("pnl", 0.0),
                "pnl_pct": pos.get("pnl_pct", 0.0),
                "signal": pos.get("signal", ""),
                "confidence": pos.get("confidence", 0.0),
                "alert_level": pos.get("alert_level", "UNKNOWN"),
                "days_held": self._calculate_days_held(pos.get("entry_date")),
                "analogs": pos.get("analogs", []),
            }

        except Exception as e:
            logger.error(f"Error loading position for {symbol}: {e}")
            return None

    def _calculate_days_held(self, entry_date: Optional[str]) -> int:
        """Calculate days held from entry date."""
        if not entry_date:
            return 0
        try:
            entry = datetime.strptime(entry_date, "%Y-%m-%d").date()
            today = date.today()
            return (today - entry).days
        except Exception:
            return 0

    def _empty_response(self) -> Dict:
        """Return empty response structure."""
        return {
            "total_positions": 0,
            "total_notional": 0.0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "positions": [],
            "alert_breakdown": {},
        }
