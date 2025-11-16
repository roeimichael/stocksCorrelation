#!/usr/bin/env python
"""Run backtest with configuration."""
import sys
from pathlib import Path


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml

from src.core.logger import get_logger
from src.trading.engine import run_backtest


logger = get_logger(__name__)


def load_config(config_path: str = 'config.yaml') -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")

    return cfg


def main():
    """Main entry point for backtest script."""
    import argparse

    parser = argparse.ArgumentParser(description='Run backtest')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("BACKTEST STARTING")
    logger.info("=" * 60)

    try:
        # Load configuration
        cfg = load_config(args.config)

        # Run backtest
        summary = run_backtest(cfg)

        # Print summary
        logger.info("=" * 60)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total Return:  ${summary['total_return']:>12,.2f}")
        logger.info(f"Sharpe Ratio:  {summary['sharpe']:>12.2f}")
        logger.info(f"Max Drawdown:  {summary['max_dd']:>12.2%}")
        logger.info(f"Hit Rate:      {summary['hit_rate']:>12.2%}")
        logger.info(f"# Trades:      {summary['n_trades']:>12}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


if __name__ == '__main__':
    main()
