"""Fetch stock data from external sources."""
import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import List, Optional
from src.core.logger import get_logger

logger = get_logger()


def get_sp500_tickers(top_n: Optional[int] = None, cache_file: str = 'data/sp500_tickers.txt') -> List[str]:
    """
    Get S&P 500 ticker symbols from Wikipedia or cached file.

    Args:
        top_n: If specified, return only top N tickers
        cache_file: Path to cache file for storing tickers

    Returns:
        List of ticker symbols
    """
    cache_path = Path(cache_file)

    # Try to load from cache file first
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(tickers)} S&P 500 tickers from cache file: {cache_file}")

            if top_n:
                tickers = tickers[:top_n]
                logger.info(f"Using top {top_n} tickers")

            return tickers
        except Exception as e:
            logger.warning(f"Error reading cache file: {e}. Will fetch from Wikipedia.")

    # Fetch from Wikipedia if cache doesn't exist or failed to load
    logger.info("Fetching S&P 500 tickers from Wikipedia...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    try:
        # Add headers to avoid 403 Forbidden
        import requests
        from io import StringIO

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()

        # Clean tickers (remove any special characters that yfinance doesn't like)
        tickers = [ticker.replace('.', '-') for ticker in tickers]

        logger.info(f"Fetched {len(tickers)} S&P 500 tickers from Wikipedia")

        # Save to cache file
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            for ticker in tickers:
                f.write(f"{ticker}\n")

        logger.info(f"Saved {len(tickers)} tickers to cache file: {cache_file}")

        if top_n:
            tickers = tickers[:top_n]
            logger.info(f"Using top {top_n} tickers")

        return tickers

    except Exception as e:
        logger.error(f"Error fetching S&P 500 tickers from Wikipedia: {e}")

        # Fallback to a small set of major tickers
        fallback = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT']
        logger.warning(f"Using fallback tickers: {fallback}")
        return fallback[:top_n] if top_n else fallback


def download_stock_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = '1d',
    output_dir: str = 'data/raw'
) -> pd.DataFrame:
    """
    Download OHLCV data for given tickers.

    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval (default: 1d)
        output_dir: Directory to save raw data

    Returns:
        DataFrame with adjusted close prices, indexed by date with tickers as columns
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}")

    # Download data for all tickers at once
    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
            auto_adjust=False,
            group_by='ticker'
        )

        if len(tickers) == 1:
            # yfinance returns different structure for single ticker
            ticker = tickers[0]
            adj_close = data['Adj Close'].to_frame(ticker)
        else:
            # Extract adjusted close prices
            try:
                adj_close = data.xs('Adj Close', axis=1, level=1)
            except KeyError:
                # Handle case where data structure is different (flat columns)
                adj_close_cols = [col for col in data.columns if 'Adj Close' in str(col)]
                if adj_close_cols:
                    adj_close = data[adj_close_cols]
                    # Clean column names
                    adj_close.columns = [col[0] if isinstance(col, tuple) else col.replace('Adj Close', '').strip() for col in adj_close.columns]
                else:
                    # Fallback: use Close if Adj Close not available
                    logger.warning("Adj Close not found, using Close prices")
                    try:
                        adj_close = data.xs('Close', axis=1, level=1)
                    except:
                        adj_close = data[[col for col in data.columns if 'Close' in str(col)]]

        # Save raw data
        output_file = output_path / 'adj_close_prices.parquet'
        adj_close.to_parquet(output_file)
        logger.info(f"Saved adjusted close prices to {output_file}")

        # Log summary
        logger.info(f"Downloaded {len(adj_close)} days of data")
        logger.info(f"Tickers with data: {adj_close.notna().any().sum()}/{len(tickers)}")

        return adj_close

    except Exception as e:
        logger.error(f"Error downloading stock data: {e}")
        raise


def load_price_data(file_path: str = 'data/raw/adj_close_prices.parquet') -> pd.DataFrame:
    """
    Load previously downloaded price data.

    Args:
        file_path: Path to parquet file

    Returns:
        DataFrame with adjusted close prices
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Price data not found: {file_path}")

    df = pd.read_parquet(path)
    logger.info(f"Loaded price data: {df.shape[0]} days, {df.shape[1]} tickers")
    return df


def get_universe(config: dict) -> List[str]:
    """
    Get ticker universe based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of ticker symbols
    """
    universe_spec = config['data']['universe']
    top_n = config['data'].get('top_n')
    include_index = config['data'].get('include_index', False)

    if universe_spec == 'sp500':
        tickers = get_sp500_tickers(top_n)

        # Add S&P 500 index if requested
        if include_index:
            tickers.append('^GSPC')
            logger.info("Added S&P 500 index (^GSPC) to universe")

        return tickers
    elif Path(universe_spec).exists():
        # Load from CSV file
        df = pd.read_csv(universe_spec)
        tickers = df['ticker'].tolist() if 'ticker' in df.columns else df.iloc[:, 0].tolist()
        logger.info(f"Loaded {len(tickers)} tickers from {universe_spec}")

        # Add S&P 500 index if requested
        if include_index and '^GSPC' not in tickers:
            tickers.append('^GSPC')
            logger.info("Added S&P 500 index (^GSPC) to universe")

        return tickers[:top_n] if top_n else tickers
    else:
        raise ValueError(f"Unknown universe specification: {universe_spec}")
