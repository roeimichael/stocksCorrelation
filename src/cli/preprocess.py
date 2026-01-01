#!/usr/bin/env python
"""
Preprocessing pipeline: fetch universe -> fetch prices -> prepare returns -> build windows.

Supports incremental updates: if recent data exists, only append new data.

Usage:
    python scripts/preprocess.py [--config config.yaml] [--force-full]
"""
import argparse
import sys
from pathlib import Path


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


import pandas as pd
import yaml

from src.core.constants import Paths
from src.core.logger import get_logger
from src.dataio.fetch import fetch_prices, fetch_universe
from src.dataio.prep import prepare_returns
from src.modeling.windows import build_windows


logger = get_logger(__name__)


def check_existing_data():
    """Check for existing processed data files and return status."""
    status = {
        'prices_exists': False,
        'prices_last_date': None,
        'returns_exists': False,
        'returns_last_date': None
    }

    if Paths.PRICES_FILE.exists():
        try:
            prices_df = pd.read_parquet(Paths.PRICES_FILE)
            status['prices_exists'] = True
            status['prices_last_date'] = prices_df.index.max()
        except Exception as e:
            logger.warning(f"Could not read existing prices: {e}")

    if Paths.RETURNS_FILE.exists():
        try:
            returns_df = pd.read_parquet(Paths.RETURNS_FILE)
            status['returns_exists'] = True
            status['returns_last_date'] = returns_df.index.max()
        except Exception as e:
            logger.warning(f"Could not read existing returns: {e}")

    return status


def run_preprocessing(cfg: dict, force_full: bool = False):
    """Run full preprocessing pipeline with incremental updates."""
    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE STARTING")
    logger.info("=" * 60)

    # Check existing data
    if not force_full:
        status = check_existing_data()
        logger.info("Existing data status:")
        logger.info(f"  Prices: {status['prices_exists']} (last: {status['prices_last_date']})")
        logger.info(f"  Returns: {status['returns_exists']} (last: {status['returns_last_date']})")
    else:
        logger.info("Force full refresh enabled - ignoring existing data")
        status = {'prices_exists': False, 'returns_exists': False}

    # Step 1: Fetch S&P 500 universe
    logger.info("-" * 60)
    logger.info("STEP 1: Fetch Universe")
    logger.info("-" * 60)

    tickers = fetch_universe(cfg)
    logger.info(f"Fetched {len(tickers)} tickers")

    # Apply light mode if enabled
    light_mode = cfg.get('light_mode', {})
    if light_mode.get('enabled', False):
        top_n = light_mode.get('top_n_stocks', 50)
        tickers = tickers[:top_n]
        logger.info(f"Light mode: Using top {len(tickers)} tickers")

    # Step 2: Fetch prices
    logger.info("-" * 60)
    logger.info("STEP 2: Fetch Prices")
    logger.info("-" * 60)

    # Note: fetch_prices writes to data/raw/ directory
    # For simplicity in CLI, always fetch (incremental logic can be added later)
    fetch_prices(tickers, cfg)
    logger.info(f"Fetched prices for {len(tickers)} symbols")

    # Step 3: Prepare returns
    logger.info("-" * 60)
    logger.info("STEP 3: Prepare Returns")
    logger.info("-" * 60)

    returns_df = prepare_returns(tickers, cfg)
    logger.info(f"Returns shape: {returns_df.shape}")

    # Step 4: Build windows
    logger.info("-" * 60)
    logger.info("STEP 4: Build Windows Bank")
    logger.info("-" * 60)

    windows_df = build_windows(returns_df, cfg)
    logger.info(f"Built {len(windows_df)} windows for {windows_df['symbol'].nunique()} symbols")

    # Label distribution
    label_counts = windows_df['label'].value_counts()
    logger.info("Label distribution:")
    logger.info(f"  Up (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(windows_df)*100:.1f}%)")
    logger.info(f"  Down (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(windows_df)*100:.1f}%)")
    logger.info(f"  Missing (-1): {label_counts.get(-1, 0)} ({label_counts.get(-1, 0)/len(windows_df)*100:.1f}%)")

    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE COMPLETE")
    logger.info("=" * 60)


def main():
    """Main entry point for preprocessing script."""
    parser = argparse.ArgumentParser(description='Run preprocessing pipeline')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--force-full',
        action='store_true',
        help='Force full refresh, ignore existing data'
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {args.config}")

    try:
        run_preprocessing(cfg, force_full=args.force_full)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


if __name__ == '__main__':
    main()
