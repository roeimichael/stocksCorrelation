"""Live data ingestion: append new bars to existing raw data."""
from pathlib import Path

import pandas as pd

from src.core.logger import get_logger


logger = get_logger(__name__)


def append_new_bars(live_dir: str = 'data/live_ingest', raw_dir: str = 'data/raw') -> list[str]:
    """
    Merge any CSV/Parquet files dropped into live_ingest directory into raw data per symbol.

    Args:
        live_dir: Directory containing new data files to ingest
        raw_dir: Directory containing existing raw data files

    Returns:
        List of symbols that were updated

    Process:
        1. Scan live_ingest directory for CSV or Parquet files
        2. For each file, identify the symbol (from filename or column)
        3. Load existing raw data for that symbol (if exists)
        4. Append new bars (deduplicating by date)
        5. Save updated data back to raw directory
        6. Move/delete ingested file

    File naming conventions:
        - CSV/Parquet files should be named: <SYMBOL>.csv or <SYMBOL>.parquet
        - OR contain a 'Symbol' column in the data

    Expected columns in ingested files:
        - Date (or index): datetime
        - Open, High, Low, Close, Volume (required)
        - Adj Close (optional, will use Close if missing)
    """
    live_path = Path(live_dir)
    raw_path = Path(raw_dir)

    # Ensure directories exist
    live_path.mkdir(parents=True, exist_ok=True)
    raw_path.mkdir(parents=True, exist_ok=True)

    updated_symbols = []

    # Scan for CSV and Parquet files
    ingest_files = list(live_path.glob('*.csv')) + list(live_path.glob('*.parquet'))

    if not ingest_files:
        logger.info(f"No files to ingest in {live_dir}")
        return updated_symbols

    logger.info(f"Found {len(ingest_files)} files to ingest")

    for file_path in ingest_files:
        try:
            # Determine symbol from filename
            symbol = file_path.stem.upper()

            logger.info(f"Processing {file_path.name} for symbol {symbol}")

            # Load new data
            if file_path.suffix == '.csv':
                new_data = pd.read_csv(file_path)
            else:
                new_data = pd.read_parquet(file_path)

            # Ensure Date is index or convert it
            if 'Date' in new_data.columns:
                new_data = new_data.set_index('Date')

            # Convert index to datetime if needed
            if not isinstance(new_data.index, pd.DatetimeIndex):
                new_data.index = pd.to_datetime(new_data.index)

            # Validate required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in new_data.columns]

            if missing_cols:
                logger.error(f"Missing required columns for {symbol}: {missing_cols}")
                continue

            # Add Adj Close if not present
            if 'Adj Close' not in new_data.columns:
                logger.debug(f"Adding Adj Close column for {symbol} (using Close)")
                new_data['Adj Close'] = new_data['Close']

            # Load existing raw data if exists
            raw_file = raw_path / f"{symbol}.parquet"

            if raw_file.exists():
                logger.debug(f"Loading existing data for {symbol}")
                existing_data = pd.read_parquet(raw_file)

                # Combine and deduplicate
                combined = pd.concat([existing_data, new_data])

                # Remove duplicates, keeping last occurrence (newer data)
                combined = combined[~combined.index.duplicated(keep='last')]

                # Sort by date
                combined = combined.sort_index()

                logger.info(f"Updated {symbol}: {len(existing_data)} -> {len(combined)} bars ({len(new_data)} new)")
            else:
                logger.info(f"Creating new file for {symbol} with {len(new_data)} bars")
                combined = new_data.sort_index()

            # Save to raw directory
            combined.to_parquet(raw_file)
            logger.info(f"Saved {symbol} to {raw_file}")

            updated_symbols.append(symbol)

            # Archive or delete the ingested file
            archive_dir = live_path / 'processed'
            archive_dir.mkdir(exist_ok=True)

            archive_path = archive_dir / file_path.name
            file_path.rename(archive_path)
            logger.debug(f"Archived {file_path.name} to {archive_path}")

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")
            continue

    logger.info(f"Ingestion complete: {len(updated_symbols)} symbols updated")

    return updated_symbols
