"""
Preprocessing script: Fetch data, clean, compute returns, build windows.

Usage:
    python scripts/preprocess.py [--config configs/default.yaml]
"""
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import load_config
from src.core.logger import setup_logger
from src.dataio.fetch import get_universe, download_stock_data
from src.dataio.prep import prepare_data
from src.modeling.windows import build_windows
from src.evals.correlation_matrix import analyze_correlations
import pickle

logger = setup_logger()


def main():
    parser = argparse.ArgumentParser(description="Preprocess stock data")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download of price data'
    )
    parser.add_argument(
        '--skip-correlation',
        action='store_true',
        help='Skip correlation analysis'
    )

    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Get universe
    logger.info("Getting ticker universe")
    tickers = get_universe(config)
    logger.info(f"Universe: {len(tickers)} tickers")

    # Download or load price data
    price_file = Path('data/raw/adj_close_prices.parquet')

    if args.force_download or not price_file.exists():
        logger.info("Downloading price data")
        prices = download_stock_data(
            tickers,
            config['data']['start'],
            config['data']['end'],
            config['data']['interval']
        )
    else:
        logger.info("Loading existing price data")
        from src.dataio.fetch import load_price_data
        prices = load_price_data()

    # Prepare data
    logger.info("Preparing data (cleaning, computing returns)")
    prices_clean, returns = prepare_data(
        prices,
        config,
        save_path='data/processed'
    )

    # Correlation analysis (optional)
    if not args.skip_correlation:
        logger.info("Running correlation analysis")
        corr_stats = analyze_correlations(returns)
        logger.info(f"Correlation analysis: {corr_stats}")

    # Build windows
    logger.info("Building windows")
    windows = build_windows(
        returns,
        window_length=config['windows']['length'],
        normalization=config['windows']['normalization'],
        min_history=config['windows']['min_history_days']
    )

    # Save windows
    windows_file = Path('data/processed/windows.pkl')
    with open(windows_file, 'wb') as f:
        pickle.dump(windows, f)
    logger.info(f"Saved {len(windows)} windows to {windows_file}")

    # Summary statistics
    from src.modeling.windows import windows_to_dataframe
    windows_df = windows_to_dataframe(windows)

    logger.info(f"\nWindows summary:")
    logger.info(f"  Total windows: {len(windows)}")
    logger.info(f"  Date range: {windows_df['start_date'].min()} to {windows_df['end_date'].max()}")
    logger.info(f"  Unique symbols: {windows_df['symbol'].nunique()}")
    logger.info(f"  Windows per symbol (avg): {len(windows) / windows_df['symbol'].nunique():.1f}")

    # Label distribution
    label_counts = windows_df['label'].value_counts()
    logger.info(f"\nLabel distribution:")
    logger.info(f"  Up (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(windows)*100:.1f}%)")
    logger.info(f"  Down (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(windows)*100:.1f}%)")
    logger.info(f"  Missing (-1): {label_counts.get(-1, 0)} ({label_counts.get(-1, 0)/len(windows)*100:.1f}%)")

    logger.info("\nPreprocessing complete!")


if __name__ == '__main__':
    main()
