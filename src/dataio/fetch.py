"""Fetch stock data from external sources."""
import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import List, Optional
from src.core.logger import get_logger

logger = get_logger()


def get_sp500_tickers(top_n: Optional[int] = None) -> List[str]:
    """
    Get S&P 500 ticker symbols.

    Args:
        top_n: If specified, return only top N tickers by market cap

    Returns:
        List of ticker symbols
    """
    # Fetch S&P 500 list from Wikipedia
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        tables = pd.read_html(url)
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()

        # Clean tickers (remove any special characters that yfinance doesn't like)
        tickers = [ticker.replace('.', '-') for ticker in tickers]

        logger.info(f"Fetched {len(tickers)} S&P 500 tickers")

        if top_n:
            # For top_n, we'll just take the first N from the list
            # In a production system, you'd fetch market cap data and sort
            tickers = tickers[:top_n]
            logger.info(f"Using top {top_n} tickers")

        return tickers
    except Exception as e:
        logger.error(f"Error fetching S&P 500 tickers: {e}")
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
            group_by='ticker'
        )

        if len(tickers) == 1:
            # yfinance returns different structure for single ticker
            ticker = tickers[0]
            adj_close = data['Adj Close'].to_frame(ticker)
        else:
            # Extract adjusted close prices
            adj_close = data.xs('Adj Close', axis=1, level=1)

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

    if universe_spec == 'sp500':
        return get_sp500_tickers(top_n)
    elif Path(universe_spec).exists():
        # Load from CSV file
        df = pd.read_csv(universe_spec)
        tickers = df['ticker'].tolist() if 'ticker' in df.columns else df.iloc[:, 0].tolist()
        logger.info(f"Loaded {len(tickers)} tickers from {universe_spec}")
        return tickers[:top_n] if top_n else tickers
    else:
        raise ValueError(f"Unknown universe specification: {universe_spec}")
