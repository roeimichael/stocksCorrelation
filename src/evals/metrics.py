"""Performance metrics for backtesting."""
from typing import Any

import numpy as np
import pandas as pd

from src.core.logger import get_logger


logger = get_logger(__name__)


def equity_from_trades(
    trades_df: pd.DataFrame,
    costs_bps: float,
    slippage_bps: float
) -> pd.DataFrame:
    """
    Compute equity curve from trades DataFrame with transaction costs.

    Args:
        trades_df: DataFrame with columns [date, pnl, ...]
                   pnl = realized PnL per trade (before costs)
        costs_bps: Transaction costs in basis points (e.g., 5.0 = 0.05%)
        slippage_bps: Slippage in basis points (e.g., 2.0 = 0.02%)

    Returns:
        DataFrame with columns [date, equity]
            - date: Trade date (index)
            - equity: Cumulative equity after costs

    Process:
        1. For each trade, compute net PnL:
           net_pnl = pnl - (costs + slippage)
           where costs = costs_bps/10000 * abs(notional)
        2. Cumsum net PnL to get equity curve
        3. Return DataFrame indexed by date

    Notes:
        - Assumes trades_df has 'date' and 'pnl' columns
        - pnl should be gross PnL (before costs)
        - If 'notional' column exists, uses it for cost calculation
        - Otherwise, assumes notional = abs(pnl) / 0.01 (rough estimate)
        - Equity starts at 0 (cumulative PnL)

    Example:
        >>> trades = pd.DataFrame({
        ...     'date': pd.date_range('2024-01-01', periods=3),
        ...     'pnl': [100, -50, 150],
        ...     'notional': [10000, 10000, 10000]
        ... })
        >>> equity_df = equity_from_trades(trades, costs_bps=5.0, slippage_bps=2.0)
        >>> # costs per trade = 10000 * 0.0007 = 7.0
        >>> # net_pnl = [100-7, -50-7, 150-7] = [93, -57, 143]
        >>> # equity = cumsum([93, -57, 143]) = [93, 36, 179]
    """
    if len(trades_df) == 0:
        logger.warning("Empty trades DataFrame, returning empty equity")
        return pd.DataFrame(columns=['equity'])

    if 'date' not in trades_df.columns:
        raise ValueError("trades_df must have 'date' column")

    if 'pnl' not in trades_df.columns:
        raise ValueError("trades_df must have 'pnl' column")

    logger.info(f"Computing equity from {len(trades_df)} trades with "
                f"costs={costs_bps}bps, slippage={slippage_bps}bps")

    # Copy to avoid modifying original
    df = trades_df.copy()

    # Compute total cost rate in decimal
    total_cost_rate = (costs_bps + slippage_bps) / 10000.0

    # Estimate notional if not provided
    if 'notional' not in df.columns:
        # Rough estimate: notional = abs(pnl) / expected_return
        # Assuming ~1% expected return per trade
        df['notional'] = df['pnl'].abs() / 0.01
        logger.debug("Estimated notional from PnL (no notional column provided)")

    # Compute cost per trade
    df['cost'] = df['notional'].abs() * total_cost_rate

    # Compute net PnL (after costs)
    df['net_pnl'] = df['pnl'] - df['cost']

    # Cumulative sum to get equity
    df['equity'] = df['net_pnl'].cumsum()

    # Create result DataFrame indexed by date
    equity_df = df.set_index('date')[['equity']].copy()

    logger.info(f"Final equity: {equity_df['equity'].iloc[-1]:.2f}, "
                f"Total costs: {df['cost'].sum():.2f}")

    return equity_df


def summary_metrics(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame | None = None
) -> dict[str, Any]:
    """
    Compute summary performance metrics from equity curve.

    Args:
        equity_df: DataFrame with 'equity' column (indexed by date)
        trades_df: Optional DataFrame with 'pnl' column for hit_rate calculation

    Returns:
        Dictionary with metrics:
            - total_return: float, cumulative return
            - sharpe: float, daily Sharpe ratio (annualized by sqrt(252))
            - max_dd: float, maximum drawdown (negative value)
            - hit_rate: float, fraction of winning trades (if trades_df provided)

    Formulas:
        - total_return = (final_equity - initial_equity) / initial_equity
          (assumes initial_equity = 0, so total_return = final_equity)
        - sharpe = mean(daily_returns) / std(daily_returns) * sqrt(252)
        - max_dd = min((equity - running_max) / running_max)
        - hit_rate = (number of trades with pnl > 0) / (total trades)

    Notes:
        - If equity curve is empty, returns zeros
        - Sharpe is annualized (daily * sqrt(252))
        - max_dd is negative (e.g., -0.15 = 15% drawdown)
        - hit_rate only computed if trades_df provided

    Example:
        >>> equity_df = pd.DataFrame({
        ...     'equity': [100, 150, 120, 180]
        ... }, index=pd.date_range('2024-01-01', periods=4))
        >>> metrics = summary_metrics(equity_df)
        >>> print(metrics)
        {'total_return': 1.80, 'sharpe': 2.35, 'max_dd': -0.20, 'hit_rate': None}
    """
    if len(equity_df) == 0:
        logger.warning("Empty equity DataFrame, returning zero metrics")
        return {
            'total_return': 0.0,
            'sharpe': 0.0,
            'max_dd': 0.0,
            'hit_rate': None
        }

    if 'equity' not in equity_df.columns:
        raise ValueError("equity_df must have 'equity' column")

    logger.info(f"Computing summary metrics from {len(equity_df)} equity points")

    equity = equity_df['equity'].values

    # Total return (assuming starting from 0)
    # If equity starts at 0, total_return = final_equity
    # If equity starts at initial_capital, total_return = (final - initial) / initial
    # For simplicity, assume equity is cumulative PnL starting from 0
    if len(equity) > 0:
        total_return = equity[-1]
    else:
        total_return = 0.0

    # Daily returns (pct_change of equity)
    # Need to handle equity starting at 0
    # Use diff() instead for cumulative PnL
    if len(equity) > 1:
        daily_pnl = np.diff(equity)
        mean_daily_pnl = daily_pnl.mean()
        std_daily_pnl = daily_pnl.std()

        if std_daily_pnl > 0:
            # Daily Sharpe ratio, annualized
            sharpe = (mean_daily_pnl / std_daily_pnl) * np.sqrt(252)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Maximum drawdown
    # Drawdown = (equity - running_max) / running_max
    # For cumulative PnL starting at 0, need to add initial capital for percentage
    # Better: use running_max of equity directly
    running_max = np.maximum.accumulate(equity)

    # Avoid division by zero: if running_max is 0, drawdown is undefined
    # Use equity[0] + 1 as base if needed
    base = np.maximum(running_max, 1.0)  # Minimum base of 1
    drawdown = (equity - running_max) / base
    max_dd = float(drawdown.min())

    # Hit rate (from trades)
    hit_rate = None
    if trades_df is not None and 'pnl' in trades_df.columns:
        winning_trades = (trades_df['pnl'] > 0).sum()
        total_trades = len(trades_df)
        if total_trades > 0:
            hit_rate = winning_trades / total_trades
            logger.debug(f"Hit rate: {winning_trades}/{total_trades} = {hit_rate:.3f}")

    logger.info(f"Metrics: total_return={total_return:.2f}, sharpe={sharpe:.2f}, "
                f"max_dd={max_dd:.2%}, hit_rate={hit_rate}")

    return {
        'total_return': float(total_return),
        'sharpe': float(sharpe),
        'max_dd': float(max_dd),
        'hit_rate': float(hit_rate) if hit_rate is not None else None
    }
