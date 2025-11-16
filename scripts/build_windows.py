#!/usr/bin/env python
"""Build windows bank from returns data."""
import sys
from pathlib import Path


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import yaml

from src.core.logger import get_logger
from src.modeling.windows import build_windows


logger = get_logger(__name__)


def main():
    """Build windows bank."""
    logger.info("=" * 60)
    logger.info("BUILDING WINDOWS BANK")
    logger.info("=" * 60)

    # Load config
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)

    # Load returns
    logger.info("Loading returns...")
    returns_df = pd.read_parquet('data/processed/returns.parquet')
    logger.info(f"Loaded {len(returns_df)} days, {len(returns_df.columns)} symbols")
    logger.info(f"Date range: {returns_df.index.min()} to {returns_df.index.max()}")

    # Apply light mode if enabled
    light_mode = cfg.get('light_mode', {})
    if light_mode.get('enabled', False):
        logger.info("=" * 60)
        logger.info("LIGHT MODE ENABLED - Using reduced dataset")
        logger.info("=" * 60)

        # Get top N stocks
        top_n = light_mode.get('top_n_stocks', 50)
        selected_symbols = returns_df.columns[:top_n].tolist()

        logger.info(f"Selected top {len(selected_symbols)} stocks")

        # Filter returns to selected symbols
        returns_df = returns_df[selected_symbols]

        logger.info(f"Filtered to {len(returns_df.columns)} symbols")
        logger.info("=" * 60)

    # Build windows
    logger.info("Building windows...")
    windows_df = build_windows(returns_df, cfg)

    logger.info(f"Built {len(windows_df)} windows")
    logger.info(f"Symbols: {windows_df['symbol'].nunique()}")
    logger.info(f"Date range: {windows_df['start_date'].min()} to {windows_df['end_date'].max()}")

    # Save
    output_path = 'data/processed/windows.parquet'
    windows_df.to_parquet(output_path)
    logger.info(f"Saved windows to {output_path}")

    logger.info("=" * 60)
    logger.info("WINDOWS BANK BUILD COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
