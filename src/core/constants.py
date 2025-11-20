"""Centralized constants and paths for the sp500-analogs project.

This module contains all hardcoded values, paths, and configuration constants
used throughout the application. All modules should import from here rather
than embedding magic values directly in code.
"""
from pathlib import Path
from typing import Final


class Paths:
    """Centralized file system paths."""

    # Project root
    ROOT: Final[Path] = Path(__file__).parent.parent.parent

    # Data directories
    DATA_ROOT: Final[Path] = ROOT / 'data'
    DATA_RAW: Final[Path] = DATA_ROOT / 'raw'
    DATA_PROCESSED: Final[Path] = DATA_ROOT / 'processed'
    DATA_LIVE_INGEST: Final[Path] = DATA_ROOT / 'live_ingest'
    DATA_PAPER_TRADING: Final[Path] = DATA_ROOT / 'paper_trading'

    # Results directories
    RESULTS_ROOT: Final[Path] = ROOT / 'results'
    RESULTS_BACKTESTS: Final[Path] = RESULTS_ROOT / 'backtests'
    RESULTS_EXPERIMENTS: Final[Path] = RESULTS_ROOT / 'experiments'
    RESULTS_LIVE: Final[Path] = RESULTS_ROOT / 'live'
    RESULTS_PAPER_TRADING: Final[Path] = RESULTS_ROOT / 'paper_trading'
    RESULTS_PLOTS: Final[Path] = RESULTS_ROOT / 'plots'

    # Specific data files
    RETURNS_FILE: Final[Path] = DATA_PROCESSED / 'returns.parquet'
    WINDOWS_FILE: Final[Path] = DATA_PROCESSED / 'windows.parquet'
    PRICES_FILE: Final[Path] = DATA_PROCESSED / 'prices_clean.parquet'

    # State files
    PORTFOLIO_STATE: Final[Path] = DATA_PAPER_TRADING / 'portfolio_state.json'
    POSITIONS_STATE: Final[Path] = RESULTS_LIVE / 'positions_state.json'
    PORTFOLIO_LEDGER: Final[Path] = RESULTS_LIVE / 'portfolio_ledger.csv'


class TradingConstants:
    """Trading-related constants and defaults."""

    # Calendar constants
    TRADING_DAYS_PER_YEAR: Final[int] = 252
    BUSINESS_DAYS_PER_MONTH: Final[int] = 21

    # Default values
    DEFAULT_INITIAL_CAPITAL: Final[float] = 100000.0
    DEFAULT_POSITION_SIZE: Final[float] = 10000.0
    DEFAULT_MIN_HISTORY_DAYS: Final[int] = 250

    # Precision and numerical stability
    EPSILON: Final[float] = 1e-8
    VOL_EPSILON: Final[float] = 1e-6
    DECIMAL_PRECISION: Final[int] = 2

    # Estimation constants
    EXPECTED_RETURN_PER_TRADE: Final[float] = 0.01  # 1%


class MonitoringConstants:
    """Position monitoring and drift detection constants."""

    # Correlation monitoring
    CORR_WINDOW_DAYS: Final[int] = 20
    CORR_LOOKBACK_DAYS: Final[int] = 3

    # Pattern deviation monitoring
    DEVIATION_WINDOW_DAYS: Final[int] = 30

    # Drift alert thresholds - Red (critical)
    SR_FLOOR_RED: Final[float] = 0.3
    DC_FLOOR_RED: Final[float] = 0.4
    CD_DROP_ALERT_RED: Final[float] = -0.3
    PDS_Z_ALERT_RED: Final[float] = 3.0

    # Drift alert thresholds - Yellow (warning)
    SR_FLOOR_YELLOW: Final[float] = 0.5
    DC_FLOOR_YELLOW: Final[float] = 0.5
    CD_DROP_ALERT_YELLOW: Final[float] = -0.2
    PDS_Z_ALERT_YELLOW: Final[float] = 2.0


class HedgingConstants:
    """Hedging and risk management constants."""

    # Hedge basket configuration
    DEFAULT_BASKET_SIZE: Final[int] = 10
    DEFAULT_MIN_NEG_SIM: Final[float] = -0.1
    DEFAULT_TARGET_RATIO: Final[float] = 1.0

    # Volatility and rebalancing
    VOL_WINDOW_DAYS: Final[int] = 20
    REBALANCE_DAYS: Final[int] = 5


class PlotConstants:
    """Plotting and visualization constants."""

    DEFAULT_DPI: Final[int] = 150
    DEFAULT_FIGURE_SIZE: Final[tuple[int, int]] = (12, 8)
    COLORMAP: Final[str] = 'RdYlGn'


class FileFormats:
    """File format constants and templates."""

    # Timestamp formats
    TIMESTAMP_FORMAT: Final[str] = '%Y%m%d_%H%M%S'
    DATE_FORMAT: Final[str] = '%Y-%m-%d'

    # File name templates
    SIGNALS_FILE_TEMPLATE: Final[str] = 'signals_{date}.csv'
    ALERTS_FILE_TEMPLATE: Final[str] = 'alerts_{date}.csv'
    POSITIONS_FILE_TEMPLATE: Final[str] = 'open_positions_{date}.csv'
    REPORT_FILE_TEMPLATE: Final[str] = 'paper_trading_report_{timestamp}.xlsx'


class ValidationConstants:
    """Data validation and quality control constants."""

    # Retry configuration
    DEFAULT_MAX_RETRIES: Final[int] = 3
    RETRY_BACKOFF_BASE: Final[int] = 2  # Exponential backoff base (2^n seconds)

    # Data quality thresholds
    MAX_MISSING_PCT: Final[float] = 0.10  # 10%
    FORWARD_FILL_LIMIT: Final[int] = 1


class LoggingConstants:
    """Logging configuration constants."""

    # Progress logging intervals
    PROGRESS_LOG_INTERVAL: Final[int] = 50  # Log every N iterations

    # Log levels (as strings for consistency)
    LEVEL_DEBUG: Final[str] = 'DEBUG'
    LEVEL_INFO: Final[str] = 'INFO'
    LEVEL_WARNING: Final[str] = 'WARNING'
    LEVEL_ERROR: Final[str] = 'ERROR'


# Convenience exports for common use cases
TRADING_DAYS_PER_YEAR = TradingConstants.TRADING_DAYS_PER_YEAR
EPSILON = TradingConstants.EPSILON
