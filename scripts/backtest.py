"""
Backtest script: Walk-forward backtest with configured parameters.

Usage:
    python scripts/backtest.py [--config configs/default.yaml]
"""
import argparse
from pathlib import Path
import sys
import pickle
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import load_config
from src.core.logger import setup_logger
from src.dataio.prep import load_processed_data
from src.trading.engine import walk_forward_backtest
from src.evals.metrics import print_backtest_summary
import matplotlib.pyplot as plt

logger = setup_logger()


def plot_equity_curve(equity_series, output_path='results/backtests/equity_curve.png'):
    """Plot equity curve."""
    plt.figure(figsize=(12, 6))

    plt.plot(equity_series.index, equity_series.values, linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.title('Backtest Equity Curve')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved equity curve plot to {output_path}")
    plt.close()


def plot_drawdown(equity_series, output_path='results/backtests/drawdown.png'):
    """Plot drawdown curve."""
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max

    plt.figure(figsize=(12, 6))

    plt.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    plt.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.title('Drawdown')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved drawdown plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run backtest")
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
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=100000.0,
        help='Initial capital'
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

    # Determine test start date
    split_idx = int(len(returns) * (1 - args.test_split))
    test_start_date = returns.index[split_idx]
    logger.info(f"Test period: {test_start_date.date()} to {returns.index[-1].date()}")

    # Run backtest
    logger.info("Running walk-forward backtest...")
    trades_df, equity_series, metrics = walk_forward_backtest(
        windows,
        returns,
        prices,
        config,
        test_start_date,
        initial_capital=args.initial_capital
    )

    # Print summary
    print_backtest_summary(metrics)

    # Plot equity curve
    if len(equity_series) > 0:
        plot_equity_curve(equity_series)
        plot_drawdown(equity_series)

    # Save summary metrics
    metrics_df = pd.DataFrame([metrics])
    output_dir = Path('results/backtests')
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = output_dir / 'metrics.csv'
    metrics_df.to_csv(metrics_file, index=False)
    logger.info(f"Saved metrics to {metrics_file}")

    logger.info("\nBacktest complete!")


if __name__ == '__main__':
    main()
