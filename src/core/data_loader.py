"""Centralized data loading utilities to eliminate code duplication.

This module provides reusable functions for loading common data files,
reducing duplication across scripts and modules.
"""
from pathlib import Path

import pandas as pd

from src.core.constants import Paths
from src.core.logger import get_logger


logger = get_logger(__name__)


def load_returns() -> pd.DataFrame:
    """Load processed returns data.

    Returns:
        DataFrame with returns (index=dates, columns=symbols)

    Raises:
        FileNotFoundError: If returns file does not exist
    """
    if not Paths.RETURNS_FILE.exists():
        raise FileNotFoundError(f"Returns file not found: {Paths.RETURNS_FILE}")

    logger.debug(f"Loading returns from {Paths.RETURNS_FILE}")
    returns_df = pd.read_parquet(Paths.RETURNS_FILE)
    logger.info(f"Loaded {len(returns_df)} days of returns for {len(returns_df.columns)} symbols")

    return returns_df


def load_windows() -> pd.DataFrame:
    """Load processed windows bank data.

    Returns:
        DataFrame with windows [symbol, start_date, end_date, features, label]

    Raises:
        FileNotFoundError: If windows file does not exist
    """
    if not Paths.WINDOWS_FILE.exists():
        raise FileNotFoundError(f"Windows file not found: {Paths.WINDOWS_FILE}")

    logger.debug(f"Loading windows from {Paths.WINDOWS_FILE}")
    windows_df = pd.read_parquet(Paths.WINDOWS_FILE)
    logger.info(f"Loaded {len(windows_df)} windows")

    return windows_df


def load_returns_and_windows() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both returns and windows data.

    Returns:
        Tuple of (returns_df, windows_df)

    Raises:
        FileNotFoundError: If either file does not exist
    """
    returns_df = load_returns()
    windows_df = load_windows()

    return returns_df, windows_df


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, create if necessary.

    Args:
        path: Directory path to ensure

    Returns:
        The path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_path(category: str, filename: str) -> Path:
    """Construct output path for a given category.

    Args:
        category: Category name (e.g., 'backtests', 'experiments', 'live')
        filename: Output filename

    Returns:
        Full path to output file
    """
    base_dir = Paths.RESULTS_ROOT / category
    ensure_directory(base_dir)
    return base_dir / filename
