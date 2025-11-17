#!/usr/bin/env python
"""
Live (paper-trading) daily pipeline.

Steps:
1. Append new bars from data/live_ingest/ to data/raw/
2. Prepare returns for updated symbols (or all)
3. Rebuild windows for recent dates only
4. Generate signals for today's date (or last available date)
5. Write results/live/signals_YYYY-MM-DD.csv
6. Maintain results/live/positions_state.json

Usage:
    python scripts/live_daily.py [--config config.yaml] [--date YYYY-MM-DD]
"""
import argparse
import json
import sys
from pathlib import Path


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from src.core.logger import get_logger
from src.dataio.live_append import append_new_bars
from src.dataio.prep import prepare_returns
from src.modeling.similarity import rank_analogs
from src.modeling.vote import vote
from src.modeling.windows import build_windows, normalize_window


logger = get_logger(__name__)


def load_positions_state(state_file: Path) -> dict:
    """
    Load positions state from JSON file.

    Returns:
        Dictionary with structure:
        {
            "open_positions": [
                {
                    "symbol": "AAPL",
                    "entry_date": "2024-01-15",
                    "side": "UP",
                    "p_up": 0.75,
                    "confidence": 0.25,
                    "analogs": [
                        {"symbol": "MSFT", "end_date": "2024-01-10", "sim": 0.85, "label": 1},
                        ...
                    ]
                },
                ...
            ]
        }
    """
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    else:
        return {"open_positions": []}


def save_positions_state(state: dict, state_file: Path):
    """Save positions state to JSON file."""
    state_file.parent.mkdir(parents=True, exist_ok=True)

    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

    logger.info(f"Saved positions state to {state_file}")


def generate_live_signals(
    returns_df: pd.DataFrame,
    windows_bank: pd.DataFrame,
    cfg: dict,
    signal_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Generate trading signals for all symbols on a given date with analog details.

    Args:
        returns_df: DataFrame with returns (index=dates, columns=symbols)
        windows_bank: DataFrame with all windows
        cfg: Configuration dictionary
        signal_date: Date for which to generate signals

    Returns:
        DataFrame with columns [symbol, p_up, signal, confidence, analogs]
        where analogs is a list of dicts with analog details
    """
    window_length = cfg['windows']['length']
    normalization = cfg['windows']['normalization']
    similarity_metric = cfg['similarity']['metric']
    top_k = cfg['similarity']['top_k']
    min_sim = cfg['similarity'].get('min_sim', 0.0)
    vote_scheme = cfg['vote']['scheme']
    vote_threshold = cfg['vote']['threshold']
    abstain_if_below_k = cfg['vote']['abstain_if_below_k']

    # Cutoff date: use signal_date-1 to prevent look-ahead bias
    cutoff_date = signal_date - pd.Timedelta(days=1)

    logger.info(f"Generating signals for {signal_date.date()}, cutoff_date={cutoff_date.date()}")

    signals_list = []

    # For each symbol in returns
    for symbol in returns_df.columns:
        # Get returns up to cutoff_date
        symbol_returns = returns_df.loc[:cutoff_date, symbol]

        # Need at least window_length returns
        if len(symbol_returns) < window_length:
            continue

        # Skip if recent returns have NaN
        recent_returns = symbol_returns.iloc[-window_length:]
        if recent_returns.isna().any():
            continue

        # Form target window (last X returns up to cutoff_date)
        target_window = recent_returns.values

        # Normalize target window
        try:
            epsilon = cfg['windows'].get('epsilon', 1e-8)
            target_vec = normalize_window(target_window, method=normalization, epsilon=epsilon)
        except Exception as e:
            logger.warning(f"Failed to normalize target window for {symbol}: {e}")
            continue

        # Rank analogs with cutoff_date (no look-ahead)
        analogs_df = rank_analogs(
            target_vec=target_vec,
            bank_df=windows_bank,
            cutoff_date=cutoff_date,
            metric=similarity_metric,
            top_k=top_k,
            min_sim=min_sim,
            exclude_symbol=symbol  # Don't match to own history
        )

        # Vote on analogs
        vote_result = vote(
            analogs_df=analogs_df,
            scheme=vote_scheme,
            threshold=vote_threshold,
            abstain_if_below_k=abstain_if_below_k
        )

        # Convert analogs to list of dicts for JSON serialization
        analogs_list = []
        for _, analog_row in analogs_df.iterrows():
            analogs_list.append({
                'symbol': analog_row['symbol'],
                'end_date': analog_row['end_date'].strftime('%Y-%m-%d'),
                'sim': float(analog_row['sim']),
                'label': int(analog_row['label'])
            })

        # Add to signals
        signals_list.append({
            'symbol': symbol,
            'p_up': vote_result['p_up'],
            'signal': vote_result['signal'],
            'confidence': vote_result['confidence'],
            'analogs': analogs_list
        })

    # Convert to DataFrame
    signals_df = pd.DataFrame(signals_list)

    if len(signals_df) > 0:
        # Log summary
        signal_counts = signals_df['signal'].value_counts().to_dict()
        logger.info(f"Generated {len(signals_df)} signals: {signal_counts}")

    return signals_df


def run_live_daily(cfg: dict, signal_date: str = None):
    """
    Run daily live trading pipeline.

    Args:
        cfg: Configuration dictionary
        signal_date: Optional date string (YYYY-MM-DD) for signal generation.
                     If None, uses last available date in returns data.
    """
    logger.info("=" * 60)
    logger.info("LIVE DAILY PIPELINE STARTING")
    logger.info("=" * 60)

    # Step 1: Append new bars
    logger.info("-" * 60)
    logger.info("STEP 1: Append New Bars")
    logger.info("-" * 60)

    updated_symbols = append_new_bars(
        live_dir='data/live_ingest',
        raw_dir='data/raw'
    )

    if len(updated_symbols) > 0:
        logger.info(f"Updated {len(updated_symbols)} symbols: {updated_symbols}")
    else:
        logger.info("No new bars to ingest")

    # Step 2: Prepare returns
    logger.info("-" * 60)
    logger.info("STEP 2: Prepare Returns")
    logger.info("-" * 60)

    # Get all symbols from config or use updated symbols
    from src.dataio.fetch import fetch_universe
    all_symbols = fetch_universe(cfg)

    returns_df = prepare_returns(all_symbols, cfg)
    logger.info(f"Returns shape: {returns_df.shape}")

    # Step 3: Build windows
    logger.info("-" * 60)
    logger.info("STEP 3: Build Windows")
    logger.info("-" * 60)

    windows_df = build_windows(returns_df, cfg)
    logger.info(f"Built {len(windows_df)} windows for {windows_df['symbol'].nunique()} symbols")

    # Step 4: Generate signals
    logger.info("-" * 60)
    logger.info("STEP 4: Generate Signals")
    logger.info("-" * 60)

    # Determine signal date
    if signal_date:
        signal_ts = pd.Timestamp(signal_date)
        logger.info(f"Using specified signal date: {signal_ts.date()}")
    else:
        # Use last available date in returns
        signal_ts = returns_df.index[-1]
        logger.info(f"Using last available date: {signal_ts.date()}")

    signals_df = generate_live_signals(returns_df, windows_df, cfg, signal_ts)

    logger.info(f"Generated {len(signals_df)} signals")

    # Step 5: Write signals CSV
    logger.info("-" * 60)
    logger.info("STEP 5: Save Signals")
    logger.info("-" * 60)

    output_dir = Path('results/live')
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = signal_ts.strftime('%Y-%m-%d')
    signals_file = output_dir / f'signals_{date_str}.csv'

    # Separate analogs for CSV (save as JSON string)
    signals_for_csv = signals_df.copy()
    if 'analogs' in signals_for_csv.columns:
        signals_for_csv['analogs'] = signals_for_csv['analogs'].apply(json.dumps)

    signals_for_csv.to_csv(signals_file, index=False)
    logger.info(f"Saved signals to {signals_file}")

    # Step 6: Update positions state
    logger.info("-" * 60)
    logger.info("STEP 6: Update Positions State")
    logger.info("-" * 60)

    state_file = output_dir / 'positions_state.json'
    state = load_positions_state(state_file)

    # Add new positions for UP/DOWN signals (filter out ABSTAIN)
    active_signals = signals_df[signals_df['signal'] != 'ABSTAIN']

    new_positions = []
    for _, signal_row in active_signals.iterrows():
        position = {
            'symbol': signal_row['symbol'],
            'entry_date': date_str,
            'side': signal_row['signal'],
            'p_up': float(signal_row['p_up']),
            'confidence': float(signal_row['confidence']),
            'analogs': signal_row['analogs']
        }
        new_positions.append(position)

    # For simplicity, replace open positions with today's signals
    # In production, would need logic to track entry/exit
    state['open_positions'] = new_positions

    save_positions_state(state, state_file)

    logger.info(f"Updated positions state: {len(new_positions)} open positions")

    logger.info("=" * 60)
    logger.info("LIVE DAILY PIPELINE COMPLETE")
    logger.info("=" * 60)

    # Print summary
    if len(active_signals) > 0:
        logger.info("\nActive signals summary:")
        logger.info(f"  UP signals: {len(active_signals[active_signals['signal'] == 'UP'])}")
        logger.info(f"  DOWN signals: {len(active_signals[active_signals['signal'] == 'DOWN'])}")
        logger.info(f"  Avg confidence: {active_signals['confidence'].mean():.2%}")


def main():
    """Main entry point for live daily script."""
    parser = argparse.ArgumentParser(description='Run live daily trading pipeline')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Signal date (YYYY-MM-DD). If not provided, uses last available date.'
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {args.config}")

    try:
        run_live_daily(cfg, signal_date=args.date)
    except Exception as e:
        logger.error(f"Live daily pipeline failed: {e}")
        raise


if __name__ == '__main__':
    main()
