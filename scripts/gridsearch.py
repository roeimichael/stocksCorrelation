"""
Grid search script: Sweep X/Y/Z parameters to find optimal combinations.

Usage:
    python scripts/gridsearch.py [--config configs/default.yaml]
"""
import argparse
from pathlib import Path
import sys
import pickle
import pandas as pd
import numpy as np
from itertools import product

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import load_config
from src.core.logger import setup_logger
from src.dataio.prep import load_processed_data
from src.modeling.windows import build_windows, filter_windows_by_date
from src.modeling.similarity import find_top_analogs
from src.modeling.vote import generate_signal
from src.evals.metrics import compute_hit_rate, compute_precision_recall

logger = setup_logger()


def evaluate_params(
    windows,
    test_windows,
    window_length,
    top_k,
    threshold,
    metric,
    vote_scheme,
    min_analogs=10
):
    """
    Evaluate a single parameter combination.

    Returns hit rate and other metrics.
    """
    predictions = []
    actuals = []

    for target_window in test_windows:
        # Get candidate pool (before target end date)
        candidate_pool = [
            w for w in windows
            if w.end_date < target_window.end_date and w.label != -1
        ]

        if len(candidate_pool) < top_k:
            continue

        # Find analogs
        analogs = find_top_analogs(
            target_window,
            candidate_pool,
            top_k=top_k,
            metric=metric,
            min_similarity=-1.0
        )

        # Generate signal
        config_temp = {
            'vote': {
                'scheme': vote_scheme,
                'threshold': threshold,
                'abstain_if_below_k': min_analogs
            }
        }

        signal = generate_signal(analogs, config_temp)

        # Only evaluate non-abstain signals with valid labels
        if signal.direction != 'ABSTAIN' and target_window.label != -1:
            pred = 1 if signal.direction == 'UP' else 0
            predictions.append(pred)
            actuals.append(target_window.label)

    # Compute metrics
    if len(predictions) == 0:
        return {
            'hit_rate': 0.0,
            'num_signals': 0,
            'precision_up': 0.0,
            'recall_up': 0.0,
            'precision_down': 0.0,
            'recall_down': 0.0
        }

    hit_rate = compute_hit_rate(predictions, actuals)
    pr_metrics = compute_precision_recall(predictions, actuals)

    return {
        'hit_rate': hit_rate,
        'num_signals': len(predictions),
        **pr_metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Grid search for optimal parameters")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.3,
        help='Fraction of data to use for testing'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Load processed data
    logger.info("Loading processed data")
    prices, returns = load_processed_data()

    # Define parameter grid
    param_grid = {
        'window_length': [5, 10, 15, 20],
        'top_k': [10, 25, 50, 100],
        'threshold': [0.60, 0.65, 0.70, 0.75, 0.80],
        'metric': ['pearson', 'spearman', 'cosine'],
        'vote_scheme': ['majority', 'similarity_weighted']
    }

    logger.info("Parameter grid:")
    for key, values in param_grid.items():
        logger.info(f"  {key}: {values}")

    total_combinations = np.prod([len(v) for v in param_grid.values()])
    logger.info(f"Total combinations: {total_combinations}")

    # Split data
    split_idx = int(len(returns) * (1 - args.test_split))
    test_start_date = returns.index[split_idx]
    logger.info(f"Test period starts: {test_start_date.date()}")

    results = []

    # Grid search
    combination_idx = 0

    for window_length, top_k, threshold, metric, vote_scheme in product(
        param_grid['window_length'],
        param_grid['top_k'],
        param_grid['threshold'],
        param_grid['metric'],
        param_grid['vote_scheme']
    ):
        combination_idx += 1

        logger.info(f"\n[{combination_idx}/{total_combinations}] Testing: "
                    f"window={window_length}, top_k={top_k}, threshold={threshold}, "
                    f"metric={metric}, vote={vote_scheme}")

        # Build windows for this window length
        windows = build_windows(
            returns,
            window_length=window_length,
            normalization=config['windows']['normalization'],
            min_history=config['windows']['min_history_days']
        )

        # Split windows
        train_windows = filter_windows_by_date(windows, test_start_date - pd.Timedelta(days=1))
        test_windows = [w for w in windows if w.end_date >= test_start_date and w.label != -1]

        logger.info(f"  Train windows: {len(train_windows)}, Test windows: {len(test_windows)}")

        # Evaluate
        metrics = evaluate_params(
            train_windows,
            test_windows,
            window_length,
            top_k,
            threshold,
            metric,
            vote_scheme
        )

        logger.info(f"  Hit rate: {metrics['hit_rate']:.3f}, Signals: {metrics['num_signals']}")

        # Store results
        result = {
            'window_length': window_length,
            'top_k': top_k,
            'threshold': threshold,
            'metric': metric,
            'vote_scheme': vote_scheme,
            **metrics
        }
        results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by hit rate
    results_df = results_df.sort_values('hit_rate', ascending=False)

    # Save results
    output_dir = Path('results/experiments')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'gridsearch_results.csv'
    results_df.to_csv(output_file, index=False)
    logger.info(f"\nSaved grid search results to {output_file}")

    # Print top 10
    logger.info("\nTop 10 parameter combinations by hit rate:")
    logger.info("\n" + results_df.head(10).to_string())

    # Print best configuration
    best = results_df.iloc[0]
    logger.info("\nBest configuration:")
    logger.info(f"  Window length: {int(best['window_length'])}")
    logger.info(f"  Top K: {int(best['top_k'])}")
    logger.info(f"  Threshold: {best['threshold']:.2f}")
    logger.info(f"  Metric: {best['metric']}")
    logger.info(f"  Vote scheme: {best['vote_scheme']}")
    logger.info(f"  Hit rate: {best['hit_rate']:.3f}")
    logger.info(f"  Num signals: {int(best['num_signals'])}")

    logger.info("\nGrid search complete!")


if __name__ == '__main__':
    main()
