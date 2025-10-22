"""Window construction and management."""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from src.core.logger import get_logger

logger = get_logger()


@dataclass
class Window:
    """Container for a single window with metadata."""
    symbol: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    features: np.ndarray  # Normalized returns vector of length X
    label: int  # Next-day direction: 1 (up), 0 (down), -1 (missing)
    next_day_return: float  # Actual next-day return


def normalize_window(
    returns: np.ndarray,
    method: str = 'zscore'
) -> np.ndarray:
    """
    Normalize a single window of returns.

    Args:
        returns: Array of returns (length X)
        method: Normalization method ('zscore', 'rank', 'vol')

    Returns:
        Normalized returns
    """
    if len(returns) == 0:
        return returns

    if method == 'zscore':
        mean = returns.mean()
        std = returns.std()
        if std > 0:
            return (returns - mean) / std
        else:
            return returns - mean

    elif method == 'rank':
        # Rank normalize to [0, 1]
        return pd.Series(returns).rank(pct=True).values

    elif method == 'vol':
        # Scale by volatility
        std = returns.std()
        if std > 0:
            return returns / std
        else:
            return returns

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def build_windows(
    returns: pd.DataFrame,
    window_length: int,
    normalization: str = 'zscore',
    min_history: int = 250
) -> List[Window]:
    """
    Build all rolling windows from returns data.

    Args:
        returns: DataFrame with returns (dates x tickers)
        window_length: Length of each window (X)
        normalization: Normalization method
        min_history: Minimum days of history required before creating windows

    Returns:
        List of Window objects
    """
    logger.info(f"Building windows: length={window_length}, norm={normalization}")

    windows = []
    dates = returns.index
    tickers = returns.columns

    for ticker in tickers:
        ticker_returns = returns[ticker].values
        ticker_dates = dates

        # Start creating windows after min_history
        start_idx = max(window_length, min_history)

        for end_idx in range(start_idx, len(ticker_returns)):
            # Window spans [end_idx - window_length, end_idx - 1]
            window_start_idx = end_idx - window_length
            window_returns = ticker_returns[window_start_idx:end_idx]

            # Skip if any NaN in window
            if np.isnan(window_returns).any():
                continue

            # Normalize window
            features = normalize_window(window_returns, method=normalization)

            # Get next-day return and label
            if end_idx < len(ticker_returns):
                next_return = ticker_returns[end_idx]
                if np.isnan(next_return):
                    label = -1  # Missing
                else:
                    label = 1 if next_return > 0 else 0
            else:
                # No next day available (most recent window)
                next_return = np.nan
                label = -1

            window = Window(
                symbol=ticker,
                start_date=ticker_dates[window_start_idx],
                end_date=ticker_dates[end_idx - 1],
                features=features,
                label=label,
                next_day_return=next_return
            )

            windows.append(window)

    logger.info(f"Built {len(windows)} windows across {len(tickers)} tickers")

    return windows


def filter_windows_by_date(
    windows: List[Window],
    max_date: pd.Timestamp
) -> List[Window]:
    """
    Filter windows to only include those ending on or before max_date.
    This prevents look-ahead bias.

    Args:
        windows: List of windows
        max_date: Maximum end date (inclusive)

    Returns:
        Filtered list of windows
    """
    filtered = [w for w in windows if w.end_date <= max_date]
    logger.info(f"Filtered to {len(filtered)}/{len(windows)} windows ending <= {max_date.date()}")
    return filtered


def get_latest_window(
    returns: pd.DataFrame,
    ticker: str,
    window_length: int,
    normalization: str = 'zscore',
    as_of_date: Optional[pd.Timestamp] = None
) -> Optional[Window]:
    """
    Get the most recent window for a ticker.

    Args:
        returns: DataFrame with returns
        ticker: Ticker symbol
        window_length: Window length
        normalization: Normalization method
        as_of_date: Date to get window as of (default: latest available)

    Returns:
        Window object or None if insufficient data
    """
    if ticker not in returns.columns:
        logger.warning(f"Ticker {ticker} not in returns data")
        return None

    ticker_returns = returns[ticker]

    if as_of_date:
        ticker_returns = ticker_returns[ticker_returns.index <= as_of_date]

    if len(ticker_returns) < window_length:
        logger.warning(f"Insufficient data for {ticker}: {len(ticker_returns)} < {window_length}")
        return None

    # Get last window_length returns
    window_returns = ticker_returns.iloc[-window_length:].values
    dates = ticker_returns.iloc[-window_length:].index

    # Skip if any NaN
    if np.isnan(window_returns).any():
        logger.warning(f"NaN values in latest window for {ticker}")
        return None

    # Normalize
    features = normalize_window(window_returns, method=normalization)

    window = Window(
        symbol=ticker,
        start_date=dates[0],
        end_date=dates[-1],
        features=features,
        label=-1,  # Unknown (future)
        next_day_return=np.nan
    )

    return window


def windows_to_dataframe(windows: List[Window]) -> pd.DataFrame:
    """
    Convert list of windows to DataFrame for analysis.

    Args:
        windows: List of Window objects

    Returns:
        DataFrame with window metadata
    """
    data = []
    for w in windows:
        data.append({
            'symbol': w.symbol,
            'start_date': w.start_date,
            'end_date': w.end_date,
            'label': w.label,
            'next_day_return': w.next_day_return
        })

    df = pd.DataFrame(data)
    return df
