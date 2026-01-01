"""Service for position monitoring and alerts."""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from src.core.logger import get_logger
from src.core.constants import Paths

logger = get_logger(__name__)


class MonitoringService:
    """Service for monitoring positions and generating alerts."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.alerts_dir = Paths.RESULTS_LIVE

    def get_latest_alerts(self) -> Dict:
        """Get the most recent alerts file."""
        try:
            # Find latest alerts file
            alert_files = list(self.alerts_dir.glob("alerts_*.csv"))
            if not alert_files:
                logger.warning(f"No alert files found in {self.alerts_dir}")
                return self._empty_response()

            # Sort by filename (date) descending
            alert_files.sort(reverse=True)
            latest_file = alert_files[0]

            # Extract date from filename
            date_str = latest_file.stem.replace("alerts_", "")

            # Load alerts
            df = pd.read_csv(latest_file)

            if df.empty:
                return self._empty_response(date_str)

            # Count alerts by level
            green = len(df[df["alert_level"] == "GREEN"])
            yellow = len(df[df["alert_level"] == "YELLOW"])
            red = len(df[df["alert_level"] == "RED"])
            error = len(df[df["alert_level"] == "ERROR"])

            # Convert to list of dicts
            alerts = []
            for _, row in df.iterrows():
                alert = {
                    "symbol": row.get("symbol", ""),
                    "alert_level": row.get("alert_level", "UNKNOWN"),
                    "date": date_str,
                    "message": row.get("message", ""),
                    "recommended_action": row.get("recommended_action", ""),
                }

                # Add metrics if available
                metrics = {}
                if "similarity_retention" in row and pd.notna(row["similarity_retention"]):
                    metrics["similarity_retention"] = float(row["similarity_retention"])
                if "directional_concordance" in row and pd.notna(row["directional_concordance"]):
                    metrics["directional_concordance"] = float(row["directional_concordance"])
                if "correlation_decay" in row and pd.notna(row["correlation_decay"]):
                    metrics["correlation_decay"] = float(row["correlation_decay"])
                if "pattern_deviation_z" in row and pd.notna(row["pattern_deviation_z"]):
                    metrics["pattern_deviation_z"] = float(row["pattern_deviation_z"])

                if metrics:
                    alert["metrics"] = metrics

                alerts.append(alert)

            return {
                "date": date_str,
                "total_alerts": len(alerts),
                "green": green,
                "yellow": yellow,
                "red": red,
                "error": error,
                "alerts": alerts,
            }

        except Exception as e:
            logger.error(f"Error loading latest alerts: {e}")
            return self._empty_response()

    def get_alerts_by_date(self, date: str) -> Dict:
        """Get alerts for a specific date."""
        try:
            alert_file = self.alerts_dir / f"alerts_{date}.csv"

            if not alert_file.exists():
                logger.warning(f"Alert file not found for date {date}")
                return self._empty_response(date)

            # Load alerts
            df = pd.read_csv(alert_file)

            if df.empty:
                return self._empty_response(date)

            # Count alerts by level
            green = len(df[df["alert_level"] == "GREEN"])
            yellow = len(df[df["alert_level"] == "YELLOW"])
            red = len(df[df["alert_level"] == "RED"])
            error = len(df[df["alert_level"] == "ERROR"])

            # Convert to list of dicts
            alerts = df.to_dict("records")

            return {
                "date": date,
                "total_alerts": len(alerts),
                "green": green,
                "yellow": yellow,
                "red": red,
                "error": error,
                "alerts": alerts,
            }

        except Exception as e:
            logger.error(f"Error loading alerts for date {date}: {e}")
            return self._empty_response(date)

    def get_alert_history(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get alert history for a specific symbol over the last N days."""
        try:
            # Find all alert files
            alert_files = sorted(list(self.alerts_dir.glob("alerts_*.csv")), reverse=True)

            # Limit to last N days
            alert_files = alert_files[:days]

            history = []
            for alert_file in alert_files:
                try:
                    df = pd.read_csv(alert_file)
                    symbol_alerts = df[df["symbol"] == symbol.upper()]

                    if not symbol_alerts.empty:
                        date_str = alert_file.stem.replace("alerts_", "")
                        for _, row in symbol_alerts.iterrows():
                            history.append(
                                {
                                    "date": date_str,
                                    "symbol": symbol,
                                    "alert_level": row.get("alert_level", "UNKNOWN"),
                                    "message": row.get("message", ""),
                                }
                            )
                except Exception as e:
                    logger.warning(f"Error reading alert file {alert_file}: {e}")
                    continue

            return history

        except Exception as e:
            logger.error(f"Error loading alert history for {symbol}: {e}")
            return []

    def _empty_response(self, date: str = "") -> Dict:
        """Return empty alerts response."""
        return {
            "date": date,
            "total_alerts": 0,
            "green": 0,
            "yellow": 0,
            "red": 0,
            "error": 0,
            "alerts": [],
        }
