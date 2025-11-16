"""
Live signal generation script: Generate signals for today's trading.

Usage:
    python scripts/live.py [--config configs/default.yaml]
"""
import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import load_config
from src.core.logger import setup_logger
from src.dataio.prep import load_processed_data
from src.modeling.similarity import find_top_analogs
from src.modeling.vote import generate_signal
from src.modeling.windows import filter_windows_by_date, get_latest_window


logger = setup_logger()


def main():
    parser = argparse.ArgumentParser(description="Generate live signals")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--as-of-date',
        type=str,
        default=None,
        help='Generate signals as of this date (YYYY-MM-DD), default: latest'
    )

    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Load processed data
    logger.info("Loading processed data")
    prices, returns = load_processed_data()

    # Load windows
    windows_file = Path('data/processed/windows.pkl')
    if not windows_file.exists():
        logger.error("Windows file not found. Run preprocess.py first.")
        return

    with open(windows_file, 'rb') as f:
        windows = pickle.load(f)
    logger.info(f"Loaded {len(windows)} windows")

    # Determine as-of date
    if args.as_of_date:
        as_of_date = pd.Timestamp(args.as_of_date)
    else:
        as_of_date = returns.index[-1]

    logger.info(f"Generating signals as of {as_of_date.date()}")

    # Get candidate pool (all windows before as_of_date)
    candidate_pool = filter_windows_by_date(windows, as_of_date - pd.Timedelta(days=1))
    candidate_pool = [w for w in candidate_pool if w.label != -1]
    logger.info(f"Candidate pool: {len(candidate_pool)} windows")

    # Get latest window for each ticker
    tickers = returns.columns.tolist()
    logger.info(f"Generating signals for {len(tickers)} tickers")

    signals_data = []

    for ticker in tickers:
        # Get latest window
        target_window = get_latest_window(
            returns,
            ticker,
            window_length=config['windows']['length'],
            normalization=config['windows']['normalization'],
            as_of_date=as_of_date
        )

        if not target_window:
            continue

        # Find analogs
        analogs = find_top_analogs(
            target_window,
            candidate_pool,
            top_k=config['similarity']['top_k'],
            metric=config['similarity']['metric'],
            min_similarity=config['similarity']['min_sim']
        )

        # Generate signal
        signal = generate_signal(analogs, config)

        # Store results
        signals_data.append({
            'ticker': ticker,
            'date': as_of_date,
            'signal': signal.direction,
            'confidence': signal.confidence,
            'num_analogs': signal.num_analogs,
            'avg_similarity': signal.avg_similarity,
            'window_start': target_window.start_date,
            'window_end': target_window.end_date
        })

    # Convert to DataFrame
    signals_df = pd.DataFrame(signals_data)

    # Sort by confidence
    signals_df = signals_df.sort_values('confidence', ascending=False)

    # Filter to active signals only
    active_signals = signals_df[signals_df['signal'] != 'ABSTAIN'].copy()

    logger.info("\nGenerated signals:")
    logger.info(f"  Total: {len(signals_df)}")
    logger.info(f"  UP: {len(active_signals[active_signals['signal'] == 'UP'])}")
    logger.info(f"  DOWN: {len(active_signals[active_signals['signal'] == 'DOWN'])}")
    logger.info(f"  ABSTAIN: {len(signals_df[signals_df['signal'] == 'ABSTAIN'])}")

    # Save signals
    output_dir = Path(config['live']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all signals
    all_signals_file = output_dir / f"signals_{as_of_date.strftime('%Y%m%d')}.csv"
    signals_df.to_csv(all_signals_file, index=False)
    logger.info(f"\nSaved all signals to {all_signals_file}")

    # Save active signals (for trading)
    if len(active_signals) > 0:
        active_signals_file = output_dir / f"active_signals_{as_of_date.strftime('%Y%m%d')}.csv"
        active_signals.to_csv(active_signals_file, index=False)
        logger.info(f"Saved active signals to {active_signals_file}")

        # Print top signals
        max_positions = config['backtest']['max_positions']
        top_signals = active_signals.head(max_positions)

        logger.info(f"\nTop {len(top_signals)} signals for trading:")
        logger.info("\n" + top_signals[['ticker', 'signal', 'confidence', 'num_analogs']].to_string(index=False))

    logger.info("\nLive signal generation complete!")


if __name__ == '__main__':
    main()
