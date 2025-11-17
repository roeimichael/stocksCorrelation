"""Data preprocessing and cleaning with trading calendar alignment."""
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from src.core.logger import get_logger


logger = get_logger(__name__)


def prepare_returns(symbols: list[str], cfg: dict) -> pd.DataFrame:
    """
    Read raw parquet files, align on NYSE trading calendar, and compute returns.

    Args:
        symbols: List of ticker symbols
        cfg: Configuration dictionary with 'data' section

    Returns:
        Wide DataFrame with shape (days, tickers) containing daily returns

    Process:
        1. Load per-symbol parquet files from data/raw/<symbol>.parquet
        2. Align all data to NYSE trading calendar (pandas_market_calendars)
        3. Forward-fill small gaps (max 1 day)
        4. Compute close-to-close daily returns: (Close_t / Close_{t-1}) - 1
        5. Save to data/processed/returns.parquet
        6. Return the wide DataFrame

    Features:
        - Handles missing data by forward-filling up to 1 day
        - Aligns all symbols to common NYSE trading calendar
        - Removes symbols with insufficient data
        - Logs data quality stats
    """
    raw_dir = Path('data/raw')
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)

    start_date = cfg['data']['start']
    end_date = cfg['data']['end']

    logger.info(f"Preparing returns for {len(symbols)} symbols")

    # Get NYSE trading calendar
    logger.debug("Loading NYSE trading calendar")
    nyse = mcal.get_calendar('NYSE')
    trading_days = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_dates = pd.DatetimeIndex(trading_days.index.date)

    logger.info(f"NYSE trading calendar: {len(trading_dates)} days from {trading_dates[0]} to {trading_dates[-1]}")

    # Load all symbol data and extract Adj Close
    price_dict = {}
    skipped = []

    for symbol in symbols:
        symbol_file = raw_dir / f"{symbol}.parquet"

        if not symbol_file.exists():
            logger.warning(f"Missing file for {symbol}, skipping")
            skipped.append(symbol)
            continue

        try:
            data = pd.read_parquet(symbol_file)

            # Use Adj Close for returns calculation (already adjusted for splits/dividends)
            if 'Adj Close' not in data.columns:
                logger.warning(f"{symbol}: No 'Adj Close' column, using 'Close'")
                prices = data['Close']
            else:
                prices = data['Adj Close']

            # Ensure datetime index
            if not isinstance(prices.index, pd.DatetimeIndex):
                prices.index = pd.to_datetime(prices.index)

            # Convert to date only (remove time component)
            prices.index = pd.DatetimeIndex(prices.index.date)

            price_dict[symbol] = prices

        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")
            skipped.append(symbol)

    if skipped:
        logger.warning(f"Skipped {len(skipped)}/{len(symbols)} symbols due to errors")

    if not price_dict:
        raise ValueError("No valid symbol data loaded")

    # Combine into wide DataFrame
    logger.debug(f"Combining {len(price_dict)} symbols into wide DataFrame")
    prices = pd.DataFrame(price_dict)

    # Reindex to NYSE trading calendar
    logger.debug("Aligning to NYSE trading calendar")
    prices = prices.reindex(trading_dates)

    # Forward-fill gaps
    forward_fill_limit = cfg['data'].get('forward_fill_limit', 1)
    logger.debug(f"Forward-filling gaps (max {forward_fill_limit} day)")
    prices_filled = prices.ffill(limit=forward_fill_limit)

    # Check data quality before computing returns
    missing_before = prices.isna().sum().sum()
    missing_after = prices_filled.isna().sum().sum()
    logger.info(f"Missing values: {missing_before} before ffill, {missing_after} after ffill (max 1 day)")

    # Drop columns (symbols) with too many NaNs
    max_missing_pct = cfg['data'].get('max_missing_pct', 0.10)
    missing_pct = prices_filled.isna().sum() / len(prices_filled)
    bad_symbols = missing_pct[missing_pct > max_missing_pct].index.tolist()

    if bad_symbols:
        logger.warning(f"Dropping {len(bad_symbols)} symbols with >{max_missing_pct*100}% missing data: {bad_symbols}")
        prices_filled = prices_filled.drop(columns=bad_symbols)

    # Compute daily returns: (P_t / P_{t-1}) - 1
    logger.debug("Computing daily returns")
    returns = prices_filled.pct_change()

    # Drop first row (NaN from pct_change)
    returns = returns.iloc[1:]

    # Check for any remaining NaNs
    remaining_nans = returns.isna().sum().sum()
    if remaining_nans > 0:
        logger.warning(f"{remaining_nans} NaN values remain in returns")

        # Drop rows with any NaN (conservative approach for clean data)
        initial_rows = len(returns)
        returns = returns.dropna(how='any')
        dropped_rows = initial_rows - len(returns)

        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows containing NaN values")

    # Final data quality report
    logger.info(f"Returns DataFrame shape: {returns.shape} ({returns.shape[0]} days, {returns.shape[1]} symbols)")
    logger.info(f"Date range: {returns.index[0]} to {returns.index[-1]}")

    # Summary statistics
    logger.debug(f"Returns summary: mean={returns.mean().mean():.6f}, std={returns.std().mean():.6f}")

    # Save to parquet
    output_file = processed_dir / 'returns.parquet'
    returns.to_parquet(output_file)
    logger.info(f"Saved returns to {output_file}")

    return returns
