"""Service for trading signals management."""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import glob

from src.core.logger import get_logger
from src.core.constants import Paths

logger = get_logger(__name__)


class SignalsService:
    """Service for managing and querying trading signals."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.signals_dir = Paths.RESULTS_LIVE

    def get_latest_signals(self) -> Dict:
        """Get the most recent signals file."""
        try:
            # Find latest signals file
            signal_files = list(self.signals_dir.glob("signals_*.csv"))
            if not signal_files:
                logger.warning(f"No signal files found in {self.signals_dir}")
                return self._empty_response()

            # Sort by filename (date) descending
            signal_files.sort(reverse=True)
            latest_file = signal_files[0]

            # Extract date from filename
            date_str = latest_file.stem.replace("signals_", "")

            # Load signals
            df = pd.read_csv(latest_file)

            if df.empty:
                return self._empty_response(date_str)

            # Count signals by type
            up_signals = len(df[df["signal"] == "UP"])
            down_signals = len(df[df["signal"] == "DOWN"])
            abstain_signals = len(df[df["signal"] == "ABSTAIN"])

            # Convert to list of dicts
            signals = df.to_dict("records")

            # Parse analogs if stored as JSON string
            for signal in signals:
                if "analogs" in signal and isinstance(signal["analogs"], str):
                    try:
                        import json
                        signal["analogs"] = json.loads(signal["analogs"])
                    except Exception:
                        signal["analogs"] = []

            return {
                "status": "success",
                "date": date_str,
                "total_signals": len(signals),
                "up_signals": up_signals,
                "down_signals": down_signals,
                "abstain_signals": abstain_signals,
                "signals": signals,
            }

        except Exception as e:
            logger.error(f"Error loading latest signals: {e}")
            return self._empty_response()

    def get_signals_by_date(self, date: str) -> Dict:
        """Get signals for a specific date."""
        try:
            signal_file = self.signals_dir / f"signals_{date}.csv"

            if not signal_file.exists():
                logger.warning(f"Signal file not found for date {date}")
                return self._empty_response(date)

            # Load signals
            df = pd.read_csv(signal_file)

            if df.empty:
                return self._empty_response(date)

            # Count signals by type
            up_signals = len(df[df["signal"] == "UP"])
            down_signals = len(df[df["signal"] == "DOWN"])
            abstain_signals = len(df[df["signal"] == "ABSTAIN"])

            # Convert to list of dicts
            signals = df.to_dict("records")

            # Parse analogs if stored as JSON string
            for signal in signals:
                if "analogs" in signal and isinstance(signal["analogs"], str):
                    try:
                        import json
                        signal["analogs"] = json.loads(signal["analogs"])
                    except Exception:
                        signal["analogs"] = []

            return {
                "status": "success",
                "date": date,
                "total_signals": len(signals),
                "up_signals": up_signals,
                "down_signals": down_signals,
                "abstain_signals": abstain_signals,
                "signals": signals,
            }

        except Exception as e:
            logger.error(f"Error loading signals for date {date}: {e}")
            return self._empty_response(date)

    def get_signal_history(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get signal history for a specific symbol over the last N days."""
        try:
            # Find all signal files
            signal_files = sorted(list(self.signals_dir.glob("signals_*.csv")), reverse=True)

            # Limit to last N days
            signal_files = signal_files[:days]

            history = []
            for signal_file in signal_files:
                try:
                    df = pd.read_csv(signal_file)
                    symbol_signals = df[df["symbol"] == symbol.upper()]

                    if not symbol_signals.empty:
                        date_str = signal_file.stem.replace("signals_", "")
                        for _, row in symbol_signals.iterrows():
                            history.append(
                                {
                                    "date": date_str,
                                    "symbol": symbol,
                                    "signal": row.get("signal", ""),
                                    "confidence": row.get("confidence", 0.0),
                                    "p_up": row.get("p_up", 0.0),
                                    "n_analogs": row.get("n_analogs", 0),
                                }
                            )
                except Exception as e:
                    logger.warning(f"Error reading signal file {signal_file}: {e}")
                    continue

            return history

        except Exception as e:
            logger.error(f"Error loading signal history for {symbol}: {e}")
            return []

    def _empty_response(self, date: str = "") -> Dict:
        """Return empty signals response."""
        return {
            "status": "success",
            "date": date,
            "total_signals": 0,
            "up_signals": 0,
            "down_signals": 0,
            "abstain_signals": 0,
            "signals": [],
        }
