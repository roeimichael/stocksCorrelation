#!/usr/bin/env python
"""
Daily trading monitoring script.

For each open position in positions_state.json:
1. Compute drift metrics (similarity retention, directional concordance, etc.)
2. Classify alert level (GREEN/YELLOW/RED)
3. Write results/live/alerts_YYYY-MM-DD.csv
4. Update positions_state.json with metrics and alert level

Usage:
    python scripts/monitor.py [--config config.yaml] [--date YYYY-MM-DD]
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
from src.trading.monitor import (
    classify_alert,
    correlation_decay,
    directional_concordance,
    pattern_deviation_z,
    similarity_retention,
)


logger = get_logger(__name__)


def load_positions_state(state_file: Path) -> dict:
    """Load positions state from JSON file."""
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


def monitor_positions(cfg: dict, monitor_date: str = None):
    """
    Monitor all open positions and generate alerts.

    Args:
        cfg: Configuration dictionary
        monitor_date: Optional date string (YYYY-MM-DD) for monitoring.
                      If None, uses last available date in returns data.
    """
    logger.info("=" * 60)
    logger.info("POSITION MONITORING STARTING")
    logger.info("=" * 60)

    # Load positions state
    state_file = Path('results/live/positions_state.json')
    state = load_positions_state(state_file)

    open_positions = state.get('open_positions', [])

    if not open_positions:
        logger.info("No open positions to monitor")
        logger.info("=" * 60)
        return

    logger.info(f"Monitoring {len(open_positions)} open positions")

    # Load returns and windows data
    logger.info("Loading data...")
    returns_df = pd.read_parquet('data/processed/returns.parquet')
    windows_bank = pd.read_parquet('data/processed/windows.parquet')

    logger.info(f"Loaded {len(returns_df)} days of returns, {len(windows_bank)} windows")

    # Determine monitoring date
    if monitor_date:
        today = pd.Timestamp(monitor_date)
        logger.info(f"Using specified date: {today.date()}")
    else:
        # Use last available date in returns
        today = returns_df.index[-1]
        logger.info(f"Using last available date: {today.date()}")

    # Monitor each position
    alerts = []

    for i, position in enumerate(open_positions):
        symbol = position['symbol']
        entry_date = position['entry_date']
        side = position['side']

        logger.info(f"\n[{i+1}/{len(open_positions)}] Monitoring {symbol} (entered {entry_date}, side={side})")

        try:
            # Compute metrics
            logger.debug("  Computing similarity retention...")
            sr = similarity_retention(position, today, returns_df, windows_bank, cfg)

            logger.debug("  Computing directional concordance...")
            dc = directional_concordance(position, today, returns_df, cfg)

            logger.debug("  Computing correlation decay...")
            corr_today, delta_corr = correlation_decay(position, today, returns_df, cfg)

            logger.debug("  Computing pattern deviation...")
            pdz = pattern_deviation_z(position, today, returns_df, cfg)

            # Classify alert
            alert_level = classify_alert(sr, dc, delta_corr, pdz, cfg)

            logger.info("  Metrics:")
            logger.info(f"    Similarity Retention:      {sr:.3f}")
            logger.info(f"    Directional Concordance:   {dc:.3f}")
            logger.info(f"    Correlation Today:         {corr_today:.3f}")
            logger.info(f"    Correlation Delta (3d):    {delta_corr:.3f}")
            logger.info(f"    Pattern Deviation Z:       {pdz:.3f}")
            logger.info(f"  Alert Level: {alert_level}")

            # Store alert
            alert = {
                'date': today.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'entry_date': entry_date,
                'side': side,
                'similarity_retention': sr,
                'directional_concordance': dc,
                'correlation_today': corr_today,
                'correlation_delta_3d': delta_corr,
                'pattern_deviation_z': pdz,
                'alert_level': alert_level
            }
            alerts.append(alert)

            # Update position with metrics and alert
            position['last_monitor_date'] = today.strftime('%Y-%m-%d')
            position['metrics'] = {
                'similarity_retention': sr,
                'directional_concordance': dc,
                'correlation_today': corr_today,
                'correlation_delta_3d': delta_corr,
                'pattern_deviation_z': pdz
            }
            position['alert_level'] = alert_level

        except Exception as e:
            logger.error(f"  Failed to monitor {symbol}: {e}")
            # Store failed alert
            alert = {
                'date': today.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'entry_date': entry_date,
                'side': side,
                'similarity_retention': 0.0,
                'directional_concordance': 0.0,
                'correlation_today': 0.0,
                'correlation_delta_3d': 0.0,
                'pattern_deviation_z': 0.0,
                'alert_level': 'ERROR'
            }
            alerts.append(alert)
            position['alert_level'] = 'ERROR'

    # Save alerts CSV
    date_str = today.strftime('%Y-%m-%d')
    alerts_file = Path(f'results/live/alerts_{date_str}.csv')
    alerts_file.parent.mkdir(parents=True, exist_ok=True)

    alerts_df = pd.DataFrame(alerts)
    alerts_df.to_csv(alerts_file, index=False)

    logger.info(f"\nSaved alerts to {alerts_file}")

    # Update positions state
    save_positions_state(state, state_file)

    # Summary statistics
    if len(alerts_df) > 0:
        alert_counts = alerts_df['alert_level'].value_counts().to_dict()
        logger.info("\nAlert Summary:")
        for level in ['GREEN', 'YELLOW', 'RED', 'ERROR']:
            count = alert_counts.get(level, 0)
            if count > 0:
                logger.info(f"  {level}: {count}")

    logger.info("=" * 60)
    logger.info("POSITION MONITORING COMPLETE")
    logger.info("=" * 60)


def main():
    """Main entry point for monitoring script."""
    parser = argparse.ArgumentParser(description='Monitor open trading positions')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Monitoring date (YYYY-MM-DD). If not provided, uses last available date.'
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {args.config}")

    try:
        monitor_positions(cfg, monitor_date=args.date)
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        raise


if __name__ == '__main__':
    main()
