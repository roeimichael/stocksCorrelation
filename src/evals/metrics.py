"""Performance metrics for backtesting."""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from src.core.logger import get_logger

logger = get_logger()


def compute_hit_rate(
    predictions: List[int],
    actuals: List[int]
) -> float:
    """
    Compute hit rate (accuracy) of predictions.

    Args:
        predictions: List of predicted directions (1=up, 0=down)
        actuals: List of actual directions (1=up, 0=down)

    Returns:
        Hit rate in [0, 1]
    """
    if len(predictions) != len(actuals):
        raise ValueError("Predictions and actuals must have same length")

    if len(predictions) == 0:
        return 0.0

    correct = sum(p == a for p, a in zip(predictions, actuals))
    return correct / len(predictions)


def compute_precision_recall(
    predictions: List[int],
    actuals: List[int]
) -> Dict[str, float]:
    """
    Compute precision and recall for up and down predictions.

    Args:
        predictions: List of predicted directions (1=up, 0=down)
        actuals: List of actual directions (1=up, 0=down)

    Returns:
        Dictionary with precision and recall metrics
    """
    if len(predictions) != len(actuals):
        raise ValueError("Predictions and actuals must have same length")

    # True positives, false positives, etc.
    tp_up = sum(p == 1 and a == 1 for p, a in zip(predictions, actuals))
    fp_up = sum(p == 1 and a == 0 for p, a in zip(predictions, actuals))
    fn_up = sum(p == 0 and a == 1 for p, a in zip(predictions, actuals))

    tp_down = sum(p == 0 and a == 0 for p, a in zip(predictions, actuals))
    fp_down = sum(p == 0 and a == 1 for p, a in zip(predictions, actuals))
    fn_down = sum(p == 1 and a == 0 for p, a in zip(predictions, actuals))

    # Precision and recall
    precision_up = tp_up / (tp_up + fp_up) if (tp_up + fp_up) > 0 else 0.0
    recall_up = tp_up / (tp_up + fn_up) if (tp_up + fn_up) > 0 else 0.0

    precision_down = tp_down / (tp_down + fp_down) if (tp_down + fp_down) > 0 else 0.0
    recall_down = tp_down / (tp_down + fn_down) if (tp_down + fn_down) > 0 else 0.0

    return {
        'precision_up': precision_up,
        'recall_up': recall_up,
        'precision_down': precision_down,
        'recall_down': recall_down
    }


def compute_returns_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Compute return-based performance metrics.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 0)

    Returns:
        Dictionary with metrics
    """
    if len(returns) == 0:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'annualized_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'num_trades': 0
        }

    # Total return
    total_return = (1 + returns).prod() - 1

    # Annualized return (assuming daily returns)
    num_days = len(returns)
    annualized_return = (1 + total_return) ** (252 / num_days) - 1

    # Annualized volatility
    annualized_vol = returns.std() * np.sqrt(252)

    # Sharpe ratio
    if annualized_vol > 0:
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'annualized_volatility': float(annualized_vol),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'num_trades': len(returns)
    }


def compute_pnl_metrics(
    equity_curve: pd.Series,
    initial_capital: float = 100000.0
) -> Dict[str, float]:
    """
    Compute PnL-based metrics from equity curve.

    Args:
        equity_curve: Series of equity values over time
        initial_capital: Starting capital

    Returns:
        Dictionary with PnL metrics
    """
    if len(equity_curve) == 0:
        return {
            'final_equity': initial_capital,
            'total_pnl': 0.0,
            'pnl_pct': 0.0,
            'max_equity': initial_capital,
            'min_equity': initial_capital,
            'max_drawdown_pct': 0.0
        }

    final_equity = equity_curve.iloc[-1]
    total_pnl = final_equity - initial_capital
    pnl_pct = total_pnl / initial_capital

    # Max and min equity
    max_equity = equity_curve.max()
    min_equity = equity_curve.min()

    # Max drawdown percentage
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown_pct = drawdown.min()

    return {
        'final_equity': float(final_equity),
        'total_pnl': float(total_pnl),
        'pnl_pct': float(pnl_pct),
        'max_equity': float(max_equity),
        'min_equity': float(min_equity),
        'max_drawdown_pct': float(max_drawdown_pct)
    }


def compute_trade_statistics(
    trades_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Compute statistics from trade log.

    Args:
        trades_df: DataFrame with trade records (must have 'pnl' column)

    Returns:
        Dictionary with trade statistics
    """
    if len(trades_df) == 0 or 'pnl' not in trades_df.columns:
        return {
            'num_trades': 0,
            'num_wins': 0,
            'num_losses': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'avg_pnl': 0.0
        }

    pnls = trades_df['pnl']
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    num_trades = len(trades_df)
    num_wins = len(wins)
    num_losses = len(losses)

    win_rate = num_wins / num_trades if num_trades > 0 else 0.0
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0

    # Profit factor
    total_wins = wins.sum() if len(wins) > 0 else 0.0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 0.0
    profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

    avg_pnl = pnls.mean()

    return {
        'num_trades': int(num_trades),
        'num_wins': int(num_wins),
        'num_losses': int(num_losses),
        'win_rate': float(win_rate),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'profit_factor': float(profit_factor),
        'avg_pnl': float(avg_pnl)
    }


def print_backtest_summary(metrics: Dict[str, float]) -> None:
    """
    Print formatted backtest summary.

    Args:
        metrics: Dictionary with all metrics
    """
    logger.info("=" * 60)
    logger.info("BACKTEST SUMMARY")
    logger.info("=" * 60)

    # Returns metrics
    if 'annualized_return' in metrics:
        logger.info("\nReturn Metrics:")
        logger.info(f"  Total Return:        {metrics.get('total_return', 0) * 100:>8.2f}%")
        logger.info(f"  Annualized Return:   {metrics.get('annualized_return', 0) * 100:>8.2f}%")
        logger.info(f"  Annualized Vol:      {metrics.get('annualized_volatility', 0) * 100:>8.2f}%")
        logger.info(f"  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):>8.2f}")
        logger.info(f"  Max Drawdown:        {metrics.get('max_drawdown', 0) * 100:>8.2f}%")

    # PnL metrics
    if 'total_pnl' in metrics:
        logger.info("\nPnL Metrics:")
        logger.info(f"  Total PnL:           ${metrics.get('total_pnl', 0):>12,.2f}")
        logger.info(f"  PnL %:               {metrics.get('pnl_pct', 0) * 100:>8.2f}%")
        logger.info(f"  Final Equity:        ${metrics.get('final_equity', 0):>12,.2f}")

    # Trade statistics
    if 'num_trades' in metrics:
        logger.info("\nTrade Statistics:")
        logger.info(f"  Number of Trades:    {metrics.get('num_trades', 0):>8}")
        logger.info(f"  Wins / Losses:       {metrics.get('num_wins', 0):>4} / {metrics.get('num_losses', 0):<4}")
        logger.info(f"  Win Rate:            {metrics.get('win_rate', 0) * 100:>8.2f}%")
        logger.info(f"  Avg Win:             ${metrics.get('avg_win', 0):>10,.2f}")
        logger.info(f"  Avg Loss:            ${metrics.get('avg_loss', 0):>10,.2f}")
        logger.info(f"  Profit Factor:       {metrics.get('profit_factor', 0):>8.2f}")

    # Direction metrics
    if 'hit_rate' in metrics:
        logger.info("\nDirection Metrics:")
        logger.info(f"  Hit Rate:            {metrics.get('hit_rate', 0) * 100:>8.2f}%")
        logger.info(f"  Precision (up):      {metrics.get('precision_up', 0) * 100:>8.2f}%")
        logger.info(f"  Recall (up):         {metrics.get('recall_up', 0) * 100:>8.2f}%")
        logger.info(f"  Precision (down):    {metrics.get('precision_down', 0) * 100:>8.2f}%")
        logger.info(f"  Recall (down):       {metrics.get('recall_down', 0) * 100:>8.2f}%")

    logger.info("=" * 60)
