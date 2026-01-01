"""Service for trade history management."""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from src.core.logger import get_logger
from src.core.constants import Paths

logger = get_logger(__name__)


class TradesService:
    """Service for managing and analyzing trade history."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.ledger_file = Paths.PORTFOLIO_LEDGER

    def get_trade_history(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbol: Optional[str] = None,
        min_pnl: Optional[float] = None,
        max_pnl: Optional[float] = None,
    ) -> Dict:
        """
        Get trade history with optional filters.

        Args:
            start_date: Filter trades after this date (YYYY-MM-DD)
            end_date: Filter trades before this date (YYYY-MM-DD)
            symbol: Filter by symbol
            min_pnl: Minimum P&L filter
            max_pnl: Maximum P&L filter

        Returns:
            Dictionary with trade history and statistics
        """
        try:
            if not self.ledger_file.exists():
                logger.warning(f"Ledger file not found: {self.ledger_file}")
                return self._empty_response()

            # Load ledger
            df = pd.read_csv(self.ledger_file)

            if df.empty:
                return self._empty_response()

            # Apply filters
            if start_date:
                df = df[df["exit_date"] >= start_date]
            if end_date:
                df = df[df["exit_date"] <= end_date]
            if symbol:
                df = df[df["symbol"] == symbol.upper()]
            if min_pnl is not None:
                df = df[df["pnl"] >= min_pnl]
            if max_pnl is not None:
                df = df[df["pnl"] <= max_pnl]

            if df.empty:
                return self._empty_response()

            # Calculate statistics
            total_trades = len(df)
            winning_trades = len(df[df["pnl"] > 0])
            losing_trades = len(df[df["pnl"] < 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

            total_pnl = df["pnl"].sum()
            avg_win = df[df["pnl"] > 0]["pnl"].mean() if winning_trades > 0 else 0.0
            avg_loss = df[df["pnl"] < 0]["pnl"].mean() if losing_trades > 0 else 0.0

            # Profit factor
            gross_profit = df[df["pnl"] > 0]["pnl"].sum()
            gross_loss = abs(df[df["pnl"] < 0]["pnl"].sum())
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

            # Convert to list of dicts
            trades = df.to_dict("records")

            # Calculate days held if not in data
            for trade in trades:
                if "days_held" not in trade or pd.isna(trade.get("days_held")):
                    trade["days_held"] = self._calculate_days_held(
                        trade.get("entry_date"), trade.get("exit_date")
                    )

            return {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": round(win_rate, 2),
                "total_pnl": round(total_pnl, 2),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "profit_factor": round(profit_factor, 2),
                "trades": trades,
            }

        except Exception as e:
            logger.error(f"Error loading trade history: {e}")
            return self._empty_response()

    def get_recent_trades(self, n: int = 20) -> List[Dict]:
        """Get N most recent trades."""
        try:
            trade_history = self.get_trade_history()
            trades = trade_history.get("trades", [])

            # Sort by exit date descending
            trades.sort(key=lambda x: x.get("exit_date", ""), reverse=True)

            return trades[:n]

        except Exception as e:
            logger.error(f"Error loading recent trades: {e}")
            return []

    def get_performance_by_symbol(self) -> List[Dict]:
        """Get performance statistics grouped by symbol."""
        try:
            if not self.ledger_file.exists():
                return []

            df = pd.read_csv(self.ledger_file)
            if df.empty:
                return []

            # Group by symbol
            grouped = df.groupby("symbol").agg(
                {
                    "pnl": ["count", "sum", "mean"],
                    "entry_date": "first",
                    "exit_date": "last",
                }
            )

            results = []
            for symbol in grouped.index:
                row = grouped.loc[symbol]
                total_trades = int(row["pnl"]["count"])
                total_pnl = float(row["pnl"]["sum"])
                avg_pnl = float(row["pnl"]["mean"])

                symbol_df = df[df["symbol"] == symbol]
                wins = len(symbol_df[symbol_df["pnl"] > 0])
                win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

                results.append(
                    {
                        "symbol": symbol,
                        "total_trades": total_trades,
                        "total_pnl": round(total_pnl, 2),
                        "avg_pnl": round(avg_pnl, 2),
                        "win_rate": round(win_rate, 2),
                    }
                )

            # Sort by total P&L descending
            results.sort(key=lambda x: x["total_pnl"], reverse=True)

            return results

        except Exception as e:
            logger.error(f"Error calculating performance by symbol: {e}")
            return []

    def _calculate_days_held(self, entry_date: Optional[str], exit_date: Optional[str]) -> int:
        """Calculate days held between entry and exit."""
        if not entry_date or not exit_date:
            return 0
        try:
            entry = datetime.strptime(entry_date, "%Y-%m-%d").date()
            exit = datetime.strptime(exit_date, "%Y-%m-%d").date()
            return (exit - entry).days
        except Exception:
            return 0

    def _empty_response(self) -> Dict:
        """Return empty trade history response."""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "trades": [],
        }
