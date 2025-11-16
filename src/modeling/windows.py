"""Window construction and normalization for pattern matching."""
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.core.logger import get_logger


logger = get_logger(__name__)


def normalize_window(vec: np.ndarray, method: str) -> np.ndarray:
    """
    Normalize a window of returns.

    Args:
        vec: Array of returns (length X)
        method: Normalization method - "zscore", "rank", or "vol"

    Returns:
        Normalized array of same length

    Methods:
        - zscore: (x - mean) / std with epsilon guard
        - rank: rankdata normalized to [-0.5, 0.5]
        - vol: x / (std + epsilon)

    Notes:
        - Uses eps=1e-8 to prevent division by zero
        - For rank: converts to percentiles then centers around 0
    """
    vec = np.asarray(vec, dtype=np.float64)
    eps = 1e-8

    if method == 'zscore':
        mean = np.mean(vec)
        std = np.std(vec)
        if std < eps:
            # If no variance, return zero-centered values
            return vec - mean
        return (vec - mean) / (std + eps)

    if method == 'rank':
        # Rank transform to [0, 1] then shift to [-0.5, 0.5]
        ranks = rankdata(vec, method='average')  # 1-based ranks
        normalized_ranks = (ranks - 1) / (len(vec) - 1) if len(vec) > 1 else np.array([0.5])
        return normalized_ranks - 0.5

    if method == 'vol':
        # Scale by volatility (standard deviation)
        std = np.std(vec)
        return vec / (std + eps)

    raise ValueError(f"Unknown normalization method: {method}. Use 'zscore', 'rank', or 'vol'.")


def build_windows(returns_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Build rolling windows from returns DataFrame.

    Args:
        returns_df: Wide DataFrame with shape (days, symbols) containing daily returns
        cfg: Configuration dictionary with 'windows' section

    Returns:
        DataFrame with columns:
            - symbol: str
            - start_date: pd.Timestamp (first day of window)
            - end_date: pd.Timestamp (last day of window)
            - features: np.ndarray of length X (normalized returns)
            - label: int (1 if next-day return > 0, else 0)
                     Missing if no next day available

    Process:
        1. For each symbol column in returns_df
        2. Create rolling windows of length X = cfg.windows.length
        3. Normalize each window using cfg.windows.normalization method
        4. Label with next-day direction (1=up, 0=down)
        5. Store all windows in DataFrame

    Notes:
        - Minimum history required: cfg.windows.min_history_days (default 250)
        - Last window per symbol has no label (next day unavailable)
        - Windows start after min_history_days of data
    """
    window_length = cfg['windows']['length']
    normalization = cfg['windows']['normalization']
    min_history = cfg['windows'].get('min_history_days', 250)

    logger.info(f"Building windows: length={window_length}, normalization={normalization}, min_history={min_history}")

    if len(returns_df) < min_history:
        raise ValueError(f"Insufficient data: {len(returns_df)} < {min_history} days")

    windows_list = []
    dates = returns_df.index
    symbols = returns_df.columns

    for symbol in symbols:
        symbol_returns = returns_df[symbol].values

        # Start creating windows after min_history
        start_idx = max(window_length, min_history)

        for end_idx in range(start_idx, len(symbol_returns) + 1):
            # Window spans [end_idx - window_length : end_idx]
            window_start_idx = end_idx - window_length
            window_returns = symbol_returns[window_start_idx:end_idx]

            # Skip if any NaN in window
            if np.isnan(window_returns).any():
                continue

            # Normalize window
            try:
                features = normalize_window(window_returns, method=normalization)
            except Exception as e:
                logger.warning(f"Failed to normalize window for {symbol} at {dates[end_idx-1]}: {e}")
                continue

            # Get dates
            start_date = dates[window_start_idx]
            end_date = dates[end_idx - 1]

            # Compute label (next-day direction)
            # end_idx is one past the last window element, so next day is at end_idx
            if end_idx < len(symbol_returns):
                next_return = symbol_returns[end_idx]
                if np.isnan(next_return):
                    label = -1  # Missing next-day return
                else:
                    label = 1 if next_return > 0 else 0
            else:
                # No next day available
                label = -1

            windows_list.append({
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'features': features,
                'label': label
            })

    # Convert to DataFrame
    windows_df = pd.DataFrame(windows_list)

    logger.info(f"Built {len(windows_df)} windows for {len(symbols)} symbols")

    # Log label distribution
    if len(windows_df) > 0:
        label_counts = windows_df['label'].value_counts()
        logger.info(f"Label distribution: {label_counts.to_dict()}")

    # Save to parquet
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'windows.parquet'

    windows_df.to_parquet(output_file)
    logger.info(f"Saved windows to {output_file}")

    return windows_df
