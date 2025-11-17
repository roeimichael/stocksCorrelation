"""Minimal daily backtesting engine."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.logger import get_logger
from src.evals.metrics import equity_from_trades, summary_metrics
from src.modeling.similarity import rank_analogs
from src.modeling.vote import vote
from src.modeling.windows import normalize_window


logger = get_logger(__name__)


def generate_daily_signals(
    returns_df: pd.DataFrame,
    windows_bank: pd.DataFrame,
    cfg: dict[str, Any],
    date: pd.Timestamp
) -> pd.DataFrame:
    """
    Generate trading signals for all symbols on a given date.

    Args:
        returns_df: DataFrame with returns (index=dates, columns=symbols)
        windows_bank: DataFrame with all windows [symbol, start_date, end_date, features, label]
        cfg: Configuration dictionary
        date: Date for which to generate signals

    Returns:
        DataFrame with columns [symbol, p_up, signal, confidence]

    Process:
        1. For each symbol available on date-1:
           - Form target window: last X returns up to date-1
           - Normalize using cfg.windows.normalization
        2. Rank analogs from windows_bank with cutoff_date=date-1 (no leakage)
        3. Vote on analogs to generate signal
        4. Return signals DataFrame

    Notes:
        - cutoff_date=date-1 ensures no look-ahead bias
        - Only generates signals for symbols with sufficient history
        - ABSTAIN signals are included in output

    Example:
        >>> signals = generate_daily_signals(returns_df, windows_bank, cfg, pd.Timestamp('2024-01-15'))
        >>> print(signals)
           symbol   p_up signal  confidence
        0   AAPL   0.75     UP        0.25
        1   MSFT   0.45  ABSTAIN      0.05
    """
    window_length = cfg['windows']['length']
    normalization = cfg['windows']['normalization']
    similarity_metric = cfg['similarity']['metric']
    top_k = cfg['similarity']['top_k']
    min_sim = cfg['similarity'].get('min_sim', 0.0)
    vote_scheme = cfg['vote']['scheme']
    vote_threshold = cfg['vote']['threshold']
    abstain_if_below_k = cfg['vote']['abstain_if_below_k']

    # Cutoff date: use date-1 to prevent look-ahead bias
    cutoff_date = date - pd.Timedelta(days=1)

    logger.debug(f"Generating signals for {date.date()}, cutoff_date={cutoff_date.date()}")

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
            logger.warning(f"Failed to normalize target window for {symbol} on {date.date()}: {e}")
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

        # Add to signals
        signals_list.append({
            'symbol': symbol,
            'p_up': vote_result['p_up'],
            'signal': vote_result['signal'],
            'confidence': vote_result['confidence']
        })

    # Convert to DataFrame
    signals_df = pd.DataFrame(signals_list)

    if len(signals_df) > 0:
        # Log summary
        signal_counts = signals_df['signal'].value_counts().to_dict()
        logger.debug(f"Generated {len(signals_df)} signals: {signal_counts}")

    return signals_df


def run_backtest(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Run complete walk-forward backtest.

    Args:
        cfg: Configuration dictionary with:
            - data.start_date, data.end_date, data.test_start_date
            - windows.length, windows.normalization
            - similarity.metric, similarity.top_k, similarity.min_sim
            - vote.scheme, vote.threshold, vote.abstain_if_below_k
            - backtest.max_positions, backtest.costs_bps, backtest.slippage_bps

    Returns:
        Dictionary with summary metrics

    Process:
        1. Load returns and windows bank
        2. Walk from test_start_date to end_date:
           - Generate signals for each day
           - Pick up to max_positions by confidence
           - Enter at open, exit at close (same-day)
           - Apply costs and slippage
        3. Save results:
           - trades.csv: All trades
           - equity.csv: Equity curve
           - summary.json: Performance metrics
        4. Return summary

    Notes:
        - Uses data.test_start_date to split training/test
        - Only windows with end_date < signal_date are used (no leakage)
        - Creates timestamped output directory: results/backtests/<timestamp>/

    Example:
        >>> cfg = load_config('config.yaml')
        >>> summary = run_backtest(cfg)
        >>> print(f"Sharpe: {summary['sharpe']:.2f}")
        >>> print(f"Total Return: {summary['total_return']:.2f}")
    """
    logger.info("Starting backtest")

    # Load data
    returns_df = pd.read_parquet('data/processed/returns.parquet')
    windows_bank = pd.read_parquet('data/processed/windows.parquet')

    logger.info(f"Loaded {len(returns_df)} days of returns for {len(returns_df.columns)} symbols")
    logger.info(f"Loaded {len(windows_bank)} windows")

    # Apply light mode if enabled
    light_mode = cfg.get('light_mode', {})
    if light_mode.get('enabled', False):
        logger.info("=" * 60)
        logger.info("LIGHT MODE ENABLED - Using reduced dataset")
        logger.info("=" * 60)

        # Get top N stocks by market cap (assuming they appear first in columns)
        top_n = light_mode.get('top_n_stocks', 50)
        selected_symbols = returns_df.columns[:top_n].tolist()

        logger.info(f"Selected top {len(selected_symbols)} stocks: {selected_symbols[:10]}...")

        # Filter returns to selected symbols
        returns_df = returns_df[selected_symbols]

        # Filter windows to selected symbols
        windows_bank = windows_bank[windows_bank['symbol'].isin(selected_symbols)]

        logger.info(f"Filtered to {len(returns_df.columns)} symbols, {len(windows_bank)} windows")

        # Set test period to last N days
        test_days = light_mode.get('test_days', 10)
        end_date = pd.Timestamp(cfg['data']['end_date'])

        # Get last N business days
        all_dates = returns_df.index
        test_dates = all_dates[-test_days:]
        test_start_date = test_dates[0]

        logger.info(f"Light mode test period: {test_start_date.date()} to {end_date.date()} ({len(test_dates)} days)")
        logger.info("=" * 60)
    else:
        # Normal mode: use configured date range
        test_start_date = pd.Timestamp(cfg['data']['test_start_date'])
        end_date = pd.Timestamp(cfg['data']['end_date'])

        # Filter returns to test period
        test_dates = returns_df.loc[test_start_date:end_date].index

        logger.info(f"Test period: {test_start_date.date()} to {end_date.date()} ({len(test_dates)} days)")

    # Backtest parameters
    max_positions = cfg['backtest']['max_positions']
    costs_bps = cfg['backtest']['costs_bps']
    slippage_bps = cfg['backtest']['slippage_bps']

    # Walk through test dates
    all_trades = []

    for date_idx, date in enumerate(test_dates):
        # Generate signals for this date
        signals_df = generate_daily_signals(returns_df, windows_bank, cfg, date)

        if len(signals_df) == 0:
            continue

        # Filter out ABSTAIN signals
        active_signals = signals_df[signals_df['signal'] != 'ABSTAIN'].copy()

        if len(active_signals) == 0:
            continue

        # Sort by confidence descending and take top max_positions
        active_signals = active_signals.sort_values('confidence', ascending=False)
        selected_signals = active_signals.head(max_positions)

        # Execute trades
        for _, signal_row in selected_signals.iterrows():
            symbol = signal_row['symbol']
            signal_direction = signal_row['signal']

            # Get return for this day
            if symbol not in returns_df.columns or date not in returns_df.index:
                continue

            day_return = returns_df.loc[date, symbol]

            if pd.isna(day_return):
                continue

            # Compute PnL with configured notional per position
            notional = cfg['backtest'].get('notional_per_position', 10000.0)

            # Direction: UP signal = long (benefit from positive return)
            #            DOWN signal = short (benefit from negative return)
            if signal_direction == 'UP':
                pnl = notional * day_return
            elif signal_direction == 'DOWN':
                pnl = notional * (-day_return)  # Profit from decline
            else:
                continue  # Skip ABSTAIN (should already be filtered)

            # Record trade
            all_trades.append({
                'date': date,
                'symbol': symbol,
                'signal': signal_direction,
                'p_up': signal_row['p_up'],
                'confidence': signal_row['confidence'],
                'return': day_return,
                'pnl': pnl,
                'notional': notional
            })

        # Log progress
        if (date_idx + 1) % 50 == 0 or date_idx == len(test_dates) - 1:
            logger.info(f"Processed {date_idx + 1}/{len(test_dates)} test dates, {len(all_trades)} trades")

    # Convert trades to DataFrame
    if len(all_trades) == 0:
        logger.warning("No trades executed in backtest")
        return {
            'total_return': 0.0,
            'sharpe': 0.0,
            'max_dd': 0.0,
            'hit_rate': 0.0,
            'n_trades': 0
        }

    trades_df = pd.DataFrame(all_trades)

    logger.info(f"Backtest complete: {len(trades_df)} trades executed")

    # Compute equity curve
    equity_df = equity_from_trades(trades_df, costs_bps=costs_bps, slippage_bps=slippage_bps)

    # Compute summary metrics
    metrics = summary_metrics(equity_df, trades_df=trades_df)
    metrics['n_trades'] = len(trades_df)

    # Log summary
    logger.info(f"Backtest summary: total_return={metrics['total_return']:.2f}, "
                f"sharpe={metrics['sharpe']:.2f}, max_dd={metrics['max_dd']:.2%}, "
                f"hit_rate={metrics['hit_rate']:.2%}, n_trades={metrics['n_trades']}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'results/backtests/{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save trades
    trades_df.to_csv(output_dir / 'trades.csv', index=False)
    logger.info(f"Saved trades to {output_dir / 'trades.csv'}")

    # Save equity curve
    equity_df.to_csv(output_dir / 'equity.csv')
    logger.info(f"Saved equity curve to {output_dir / 'equity.csv'}")

    # Save summary
    with open(output_dir / 'summary.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        summary_serializable = {k: float(v) if v is not None else None for k, v in metrics.items()}
        json.dump(summary_serializable, f, indent=2)
    logger.info(f"Saved summary to {output_dir / 'summary.json'}")

    return metrics
