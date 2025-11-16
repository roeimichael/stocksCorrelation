#!/usr/bin/env python
"""
Grid search script: Sweep parameters (window_length, similarity.metric, similarity.top_k, vote.threshold).

For each combination, run a short backtest window and collect summary metrics.

Usage:
    python scripts/gridsearch.py [--config config.yaml]
"""
import argparse
import sys
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.core.logger import get_logger
from src.trading.engine import run_backtest


logger = get_logger(__name__)


def run_grid_search(base_cfg: dict):
    """
    Run grid search over parameter combinations.

    Args:
        base_cfg: Base configuration dictionary

    Returns:
        DataFrame with grid search results
    """
    logger.info("=" * 60)
    logger.info("GRID SEARCH STARTING")
    logger.info("=" * 60)

    # Define parameter grid
    param_grid = {
        'windows.length': [5, 10, 15],
        'similarity.metric': ['pearson', 'spearman', 'cosine'],
        'similarity.top_k': [10, 25, 50],
        'vote.threshold': [0.60, 0.65, 0.70, 0.75]
    }

    logger.info("Parameter grid:")
    for key, values in param_grid.items():
        logger.info(f"  {key}: {values}")

    total_combinations = np.prod([len(v) for v in param_grid.values()])
    logger.info(f"Total combinations: {int(total_combinations)}")

    # Prepare results storage
    results = []
    combination_idx = 0

    # Grid search
    for window_length, metric, top_k, threshold in product(
        param_grid['windows.length'],
        param_grid['similarity.metric'],
        param_grid['similarity.top_k'],
        param_grid['vote.threshold']
    ):
        combination_idx += 1

        logger.info(f"\n[{combination_idx}/{int(total_combinations)}] Testing: "
                    f"window={window_length}, metric={metric}, top_k={top_k}, threshold={threshold}")

        # Create config for this combination
        cfg = deepcopy(base_cfg)
        cfg['windows']['length'] = window_length
        cfg['similarity']['metric'] = metric
        cfg['similarity']['top_k'] = top_k
        cfg['vote']['threshold'] = threshold

        try:
            # Run backtest
            summary = run_backtest(cfg)

            # Store results
            result = {
                'window_length': window_length,
                'metric': metric,
                'top_k': top_k,
                'threshold': threshold,
                'total_return': summary['total_return'],
                'sharpe': summary['sharpe'],
                'max_dd': summary['max_dd'],
                'hit_rate': summary['hit_rate'],
                'n_trades': summary['n_trades']
            }
            results.append(result)

            logger.info(f"  Result: sharpe={summary['sharpe']:.2f}, hit_rate={summary['hit_rate']:.2%}, n_trades={summary['n_trades']}")

        except Exception as e:
            logger.error(f"  Failed: {e}")
            # Store failed result
            result = {
                'window_length': window_length,
                'metric': metric,
                'top_k': top_k,
                'threshold': threshold,
                'total_return': 0.0,
                'sharpe': 0.0,
                'max_dd': 0.0,
                'hit_rate': 0.0,
                'n_trades': 0
            }
            results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    return results_df


def main():
    """Main entry point for grid search script."""
    parser = argparse.ArgumentParser(description="Grid search for optimal parameters")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {args.config}")

    try:
        # Run grid search
        results_df = run_grid_search(cfg)

        # Sort by sharpe ratio (primary) and hit_rate (secondary)
        results_df = results_df.sort_values(['sharpe', 'hit_rate'], ascending=[False, False])

        # Save results
        output_dir = Path('results/experiments')
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'gridsearch_{timestamp}.csv'
        results_df.to_csv(output_file, index=False)

        logger.info("=" * 60)
        logger.info(f"Saved grid search results to {output_file}")
        logger.info("=" * 60)

        # Print top 5
        logger.info("\nTop 5 parameter combinations by Sharpe ratio:")
        logger.info("\n" + results_df.head(5).to_string(index=False))

        # Print best configuration
        if len(results_df) > 0:
            best = results_df.iloc[0]
            logger.info("\n" + "=" * 60)
            logger.info("BEST CONFIGURATION:")
            logger.info("=" * 60)
            logger.info(f"  Window length: {int(best['window_length'])}")
            logger.info(f"  Similarity metric: {best['metric']}")
            logger.info(f"  Top K: {int(best['top_k'])}")
            logger.info(f"  Vote threshold: {best['threshold']:.2f}")
            logger.info("\n  Performance:")
            logger.info(f"    Total Return:  ${best['total_return']:>12,.2f}")
            logger.info(f"    Sharpe Ratio:  {best['sharpe']:>12.2f}")
            logger.info(f"    Max Drawdown:  {best['max_dd']:>12.2%}")
            logger.info(f"    Hit Rate:      {best['hit_rate']:>12.2%}")
            logger.info(f"    # Trades:      {int(best['n_trades']):>12}")
            logger.info("=" * 60)

        logger.info("\nGrid search complete!")

    except Exception as e:
        logger.error(f"Grid search failed: {e}")
        raise


if __name__ == '__main__':
    main()
