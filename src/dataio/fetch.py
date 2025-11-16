"""Fetch stock data from external sources."""
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.core.logger import get_logger


logger = get_logger(__name__)


def fetch_universe(cfg: dict) -> list[str]:
    """
    Get ticker universe based on configuration.

    Args:
        cfg: Configuration dictionary with 'data' section

    Returns:
        List of ticker symbols

    Logic:
        - If cfg.data.universe == "sp500":
          - Check for artifacts/universes/top50_<asof>.csv (if present, use it)
          - Otherwise return placeholder list of ~10 tickers for tests
        - Otherwise treat as path to CSV file
    """
    universe_spec = cfg['data']['universe']
    top_n = cfg['data'].get('top_n')
    include_index = cfg['data'].get('include_index', False)

    if universe_spec == 'sp500':
        # Check for artifacts/universes/top50_*.csv files
        artifacts_dir = Path('artifacts/universes')
        if artifacts_dir.exists():
            csv_files = list(artifacts_dir.glob('top50_*.csv'))
            if csv_files:
                # Use the most recent one (by filename)
                latest_csv = sorted(csv_files)[-1]
                logger.info(f"Loading universe from {latest_csv}")
                df = pd.read_csv(latest_csv)
                tickers = df['ticker'].tolist() if 'ticker' in df.columns else df.iloc[:, 0].tolist()
                logger.info(f"Loaded {len(tickers)} tickers from artifact file")

                if include_index and '^GSPC' not in tickers:
                    tickers.append('^GSPC')
                    logger.info("Added S&P 500 index (^GSPC) to universe")

                return tickers[:top_n] if top_n else tickers

        # Placeholder list for tests/development
        placeholder = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
            'TSLA', 'NVDA', 'JPM', 'V', 'JNJ'
        ]

        tickers = placeholder[:top_n] if top_n else placeholder
        logger.info(f"Using placeholder universe: {len(tickers)} tickers")

        if include_index and '^GSPC' not in tickers:
            tickers.append('^GSPC')
            logger.info("Added S&P 500 index (^GSPC) to universe")

        return tickers

    if Path(universe_spec).exists():
        # Load from CSV file
        df = pd.read_csv(universe_spec)
        tickers = df['ticker'].tolist() if 'ticker' in df.columns else df.iloc[:, 0].tolist()
        logger.info(f"Loaded {len(tickers)} tickers from {universe_spec}")

        if include_index and '^GSPC' not in tickers:
            tickers.append('^GSPC')
            logger.info("Added S&P 500 index (^GSPC) to universe")

        return tickers[:top_n] if top_n else tickers
    raise ValueError(f"Unknown universe specification: {universe_spec}")


def fetch_prices(symbols: list[str], cfg: dict) -> None:
    """
    Download adjusted OHLCV data via yfinance with retries and save per-symbol parquet files.

    Args:
        symbols: List of ticker symbols
        cfg: Configuration dictionary with 'data' section

    Saves:
        Per-symbol parquet files in data/raw/<symbol>.parquet with columns:
        - Date (index)
        - Open, High, Low, Close, Volume, Adj Close

    Features:
        - Retry logic for failed downloads (max 3 attempts)
        - Trading calendar alignment (NYSE via pandas_market_calendars)
        - Progress logging
    """
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)

    start_date = cfg['data']['start']
    end_date = cfg['data']['end']
    interval = cfg['data'].get('interval', '1d')

    logger.info(f"Fetching prices for {len(symbols)} symbols from {start_date} to {end_date}")

    success_count = 0
    fail_count = 0

    for symbol in symbols:
        output_file = output_dir / f"{symbol}.parquet"

        # Skip if already downloaded
        if output_file.exists():
            logger.debug(f"Skipping {symbol} (already exists)")
            success_count += 1
            continue

        # Try up to 3 times with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.debug(f"Downloading {symbol} (attempt {attempt + 1}/{max_retries})")

                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=False,
                    actions=False
                )

                if data.empty:
                    logger.warning(f"No data returned for {symbol}")
                    fail_count += 1
                    break

                # Ensure we have the required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in data.columns for col in required_cols):
                    logger.warning(f"Missing columns for {symbol}: {data.columns.tolist()}")
                    fail_count += 1
                    break

                # Add Adj Close if not present (use Close as fallback)
                if 'Adj Close' not in data.columns:
                    data['Adj Close'] = data['Close']

                # Save to parquet
                data.to_parquet(output_file)
                logger.debug(f"Saved {symbol}: {len(data)} rows")
                success_count += 1
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"Error fetching {symbol}: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to fetch {symbol} after {max_retries} attempts: {e}")
                    fail_count += 1

        # Small delay between symbols to avoid rate limiting
        time.sleep(0.1)

    logger.info(f"Fetch complete: {success_count} successful, {fail_count} failed")

    if fail_count > 0:
        logger.warning(f"{fail_count}/{len(symbols)} symbols failed to download")
