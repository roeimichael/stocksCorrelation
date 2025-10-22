"""Data preprocessing and cleaning."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from src.core.logger import get_logger

logger = get_logger()


def clean_price_data(
    prices: pd.DataFrame,
    max_missing_pct: float = 0.10,
    forward_fill_limit: int = 5
) -> pd.DataFrame:
    """
    Clean price data by handling missing values and removing bad tickers.

    Args:
        prices: DataFrame with prices (dates x tickers)
        max_missing_pct: Maximum allowed percentage of missing data per ticker
        forward_fill_limit: Maximum number of days to forward fill

    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Cleaning price data: {prices.shape}")

    # Remove tickers with too much missing data
    missing_pct = prices.isna().sum() / len(prices)
    bad_tickers = missing_pct[missing_pct > max_missing_pct].index.tolist()

    if bad_tickers:
        logger.warning(f"Removing {len(bad_tickers)} tickers with >{max_missing_pct*100}% missing data")
        prices = prices.drop(columns=bad_tickers)

    # Forward fill small gaps (max forward_fill_limit days)
    prices = prices.ffill(limit=forward_fill_limit)

    # Drop rows with any remaining NaN values
    initial_rows = len(prices)
    prices = prices.dropna()
    dropped_rows = initial_rows - len(prices)

    if dropped_rows > 0:
        logger.info(f"Dropped {dropped_rows} rows with missing data")

    logger.info(f"Cleaned data shape: {prices.shape}")
    return prices


def compute_returns(
    prices: pd.DataFrame,
    method: str = 'simple'
) -> pd.DataFrame:
    """
    Compute returns from prices.

    Args:
        prices: DataFrame with prices (dates x tickers)
        method: 'simple' for (P_t/P_{t-1} - 1) or 'log' for log returns

    Returns:
        DataFrame with returns
    """
    logger.info(f"Computing {method} returns")

    if method == 'simple':
        returns = prices.pct_change()
    elif method == 'log':
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unknown return method: {method}")

    # Drop first row (NaN)
    returns = returns.iloc[1:]

    logger.info(f"Returns shape: {returns.shape}")
    return returns


def align_trading_calendar(
    returns: pd.DataFrame,
    min_tickers_per_day: int = 10
) -> pd.DataFrame:
    """
    Ensure consistent trading calendar across all tickers.

    Args:
        returns: DataFrame with returns (dates x tickers)
        min_tickers_per_day: Minimum number of tickers with data to keep a date

    Returns:
        DataFrame with aligned calendar
    """
    logger.info("Aligning trading calendar")

    # Count valid tickers per day
    valid_per_day = returns.notna().sum(axis=1)

    # Keep only days with enough valid tickers
    valid_days = valid_per_day >= min_tickers_per_day
    returns = returns[valid_days]

    logger.info(f"Calendar aligned: {len(returns)} trading days")
    return returns


def normalize_returns(
    returns: pd.DataFrame,
    method: str = 'zscore',
    window: Optional[int] = None
) -> pd.DataFrame:
    """
    Normalize returns (usually done per-window, but can be global).

    Args:
        returns: DataFrame with returns
        method: Normalization method ('zscore', 'rank', 'vol')
        window: Rolling window for normalization (None = global)

    Returns:
        Normalized returns
    """
    logger.info(f"Normalizing returns: method={method}, window={window}")

    if method == 'zscore':
        if window:
            # Rolling z-score
            mean = returns.rolling(window).mean()
            std = returns.rolling(window).std()
            normalized = (returns - mean) / std
        else:
            # Global z-score per ticker
            normalized = (returns - returns.mean()) / returns.std()

    elif method == 'rank':
        if window:
            # Rolling rank
            normalized = returns.rolling(window).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )
        else:
            # Global rank per ticker
            normalized = returns.rank(pct=True)

    elif method == 'vol':
        if window:
            # Scale by rolling volatility
            vol = returns.rolling(window).std()
            normalized = returns / vol
        else:
            # Scale by global volatility per ticker
            normalized = returns / returns.std()

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized


def prepare_data(
    prices: pd.DataFrame,
    config: dict,
    save_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete data preparation pipeline.

    Args:
        prices: Raw price DataFrame
        config: Configuration dictionary
        save_path: Optional path to save processed data

    Returns:
        Tuple of (cleaned prices, returns)
    """
    logger.info("Running data preparation pipeline")

    # Clean prices
    prices_clean = clean_price_data(prices)

    # Compute returns
    returns = compute_returns(prices_clean, method='simple')

    # Align calendar
    returns = align_trading_calendar(returns)

    # Ensure we have enough history
    min_history = config['windows'].get('min_history_days', 250)
    if len(returns) < min_history:
        raise ValueError(f"Insufficient history: {len(returns)} < {min_history} days")

    logger.info(f"Prepared data: {returns.shape[0]} days, {returns.shape[1]} tickers")

    # Save processed data
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        prices_clean.to_parquet(save_dir / 'prices_clean.parquet')
        returns.to_parquet(save_dir / 'returns.parquet')
        logger.info(f"Saved processed data to {save_path}")

    return prices_clean, returns


def load_processed_data(data_dir: str = 'data/processed') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load previously processed data.

    Args:
        data_dir: Directory containing processed data

    Returns:
        Tuple of (prices, returns)
    """
    data_path = Path(data_dir)

    prices = pd.read_parquet(data_path / 'prices_clean.parquet')
    returns = pd.read_parquet(data_path / 'returns.parquet')

    logger.info(f"Loaded processed data: {returns.shape[0]} days, {returns.shape[1]} tickers")

    return prices, returns
