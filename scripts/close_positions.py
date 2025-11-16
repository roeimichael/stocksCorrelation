#!/usr/bin/env python
"""
End-of-day position closing logic.

Reads positions_state.json and signals/alerts, then:
1. Identifies positions that meet exit criteria (RED alert, reverse signal, etc.)
2. Closes positions by recording exit details
3. Updates portfolio_ledger.csv with closed trades
4. Updates positions_state.json

Usage:
    python scripts/close_positions.py [--config config.yaml] [--date YYYY-MM-DD]
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


def should_close_position(
    position: dict,
    signals_df: pd.DataFrame,
    cfg: dict
) -> tuple[bool, str]:
    """
    Determine if a position should be closed.

    Args:
        position: Position dict with symbol, side, alert_level, etc.
        signals_df: DataFrame with latest signals
        cfg: Configuration dictionary with close_policy section

    Returns:
        Tuple of (should_close, reason)

    Exit triggers:
        1. RED alert: Risk level too high
        2. Reverse signal: Signal opposite to position side
        3. ABSTAIN signal: No longer confident
        4. Manual stop (future enhancement)

    Policy (configurable):
        - close_on_red: Close on RED alert (default: True)
        - close_on_reverse: Close on reverse signal (default: True)
        - close_on_abstain: Close on ABSTAIN signal (default: False)
    """
    close_policy = cfg.get('close_policy', {})
    close_on_red = close_policy.get('close_on_red', True)
    close_on_reverse = close_policy.get('close_on_reverse', True)
    close_on_abstain = close_policy.get('close_on_abstain', False)

    symbol = position['symbol']
    side = position['side']

    # Check RED alert
    alert_level = position.get('alert_level', 'GREEN')
    if close_on_red and alert_level == 'RED':
        return True, 'RED_ALERT'

    # Check reverse signal
    if close_on_reverse and len(signals_df) > 0:
        # Find signal for this symbol
        symbol_signals = signals_df[signals_df['symbol'] == symbol]

        if len(symbol_signals) > 0:
            latest_signal = symbol_signals.iloc[0]['signal']

            # Check for reverse
            if side == 'UP' and latest_signal == 'DOWN' or side == 'DOWN' and latest_signal == 'UP':
                return True, 'REVERSE_SIGNAL'

            # Check for abstain
            if close_on_abstain and latest_signal == 'ABSTAIN':
                return True, 'ABSTAIN_SIGNAL'

    return False, ''


def get_prices(symbol: str, entry_date: str, exit_date: str, returns_df: pd.DataFrame) -> tuple[float, float]:
    """
    Get entry and exit prices for a position.

    Args:
        symbol: Symbol
        entry_date: Entry date (YYYY-MM-DD)
        exit_date: Exit date (YYYY-MM-DD)
        returns_df: DataFrame with returns

    Returns:
        Tuple of (entry_price, exit_price)

    Note:
        Since we don't have actual price data, we reconstruct from returns.
        For simplicity, we use cumulative returns to estimate price changes.
        In production, would use actual price data.
    """
    entry_ts = pd.Timestamp(entry_date)
    exit_ts = pd.Timestamp(exit_date)

    if symbol not in returns_df.columns:
        logger.warning(f"Symbol {symbol} not found in returns_df")
        return 100.0, 100.0

    try:
        # Get returns between entry and exit
        returns_between = returns_df.loc[entry_ts:exit_ts, symbol]

        # Compute cumulative return
        cum_return = (1 + returns_between).prod() - 1

        # Assume entry price of 100 for simplicity
        entry_price = 100.0
        exit_price = entry_price * (1 + cum_return)

        return entry_price, exit_price

    except Exception as e:
        logger.warning(f"Error computing prices for {symbol}: {e}")
        return 100.0, 100.0


def compute_pnl(
    position: dict,
    entry_price: float,
    exit_price: float,
    notional: float,
    cfg: dict
) -> tuple[float, float]:
    """
    Compute gross and net PnL for a position.

    Args:
        position: Position dict with side
        entry_price: Entry price
        exit_price: Exit price
        notional: Position notional ($)
        cfg: Configuration dictionary

    Returns:
        Tuple of (gross_pnl, net_pnl)

    Process:
        1. Compute gross PnL based on side (UP=long, DOWN=short)
        2. Apply transaction costs (entry + exit)
        3. Return gross and net
    """
    side = position['side']
    costs_bps = cfg.get('backtest', {}).get('costs_bps', 5.0)
    slippage_bps = cfg.get('backtest', {}).get('slippage_bps', 2.0)

    # Compute return
    if side == 'UP':
        # Long position
        position_return = (exit_price - entry_price) / entry_price
    else:
        # Short position (profit from decline)
        position_return = -(exit_price - entry_price) / entry_price

    # Gross PnL
    gross_pnl = notional * position_return

    # Transaction costs (entry + exit)
    total_costs_bps = (costs_bps + slippage_bps) * 2  # Both entry and exit
    transaction_costs = notional * (total_costs_bps / 10000.0)

    # Net PnL
    net_pnl = gross_pnl - transaction_costs

    return float(gross_pnl), float(net_pnl)


def append_to_ledger(trade: dict, ledger_file: Path):
    """
    Append trade to portfolio ledger CSV.

    Args:
        trade: Trade dict with all details
        ledger_file: Path to ledger CSV file
    """
    ledger_file.parent.mkdir(parents=True, exist_ok=True)

    # Create DataFrame from trade
    trade_df = pd.DataFrame([trade])

    # Append to CSV (create if doesn't exist)
    if ledger_file.exists():
        # Read existing ledger
        existing_ledger = pd.read_csv(ledger_file)
        # Append new trade
        updated_ledger = pd.concat([existing_ledger, trade_df], ignore_index=True)
        updated_ledger.to_csv(ledger_file, index=False)
    else:
        # Create new ledger
        trade_df.to_csv(ledger_file, index=False)

    logger.info(f"Appended trade to {ledger_file}")


def close_positions(cfg: dict, close_date: str = None):
    """
    Close positions that meet exit criteria.

    Args:
        cfg: Configuration dictionary
        close_date: Optional date string (YYYY-MM-DD) for closing.
                    If None, uses last available date in returns data.
    """
    logger.info("=" * 60)
    logger.info("POSITION CLOSING STARTING")
    logger.info("=" * 60)

    # Load positions state
    state_file = Path('results/live/positions_state.json')
    state = load_positions_state(state_file)

    open_positions = state.get('open_positions', [])

    if not open_positions:
        logger.info("No open positions to close")
        logger.info("=" * 60)
        return

    logger.info(f"Checking {len(open_positions)} open positions for closing")

    # Load returns data
    returns_df = pd.read_parquet('data/processed/returns.parquet')

    # Determine close date
    if close_date:
        today = pd.Timestamp(close_date)
        logger.info(f"Using specified date: {today.date()}")
    else:
        today = returns_df.index[-1]
        logger.info(f"Using last available date: {today.date()}")

    date_str = today.strftime('%Y-%m-%d')

    # Load signals for today
    signals_file = Path(f'results/live/signals_{date_str}.csv')
    if signals_file.exists():
        signals_df = pd.read_csv(signals_file)
        logger.info(f"Loaded signals from {signals_file}")
    else:
        logger.warning(f"No signals file found for {date_str}")
        signals_df = pd.DataFrame()

    # Check each position for closing
    positions_to_close = []
    positions_to_keep = []

    for position in open_positions:
        symbol = position['symbol']

        # Check if should close
        should_close, reason = should_close_position(position, signals_df, cfg)

        if should_close:
            logger.info(f"Closing {symbol}: {reason}")
            positions_to_close.append((position, reason))
        else:
            positions_to_keep.append(position)

    if not positions_to_close:
        logger.info("No positions to close")
        logger.info("=" * 60)
        return

    logger.info(f"Closing {len(positions_to_close)} positions")

    # Close positions and record in ledger
    ledger_file = Path('results/live/portfolio_ledger.csv')

    for position, reason in positions_to_close:
        symbol = position['symbol']
        entry_date = position['entry_date']
        side = position['side']
        notional = position.get('notional', 10000.0)  # Default notional

        # Get prices
        entry_price, exit_price = get_prices(symbol, entry_date, date_str, returns_df)

        # Compute PnL
        gross_pnl, net_pnl = compute_pnl(position, entry_price, exit_price, notional, cfg)

        # Create trade record
        trade = {
            'symbol': symbol,
            'entry_date': entry_date,
            'exit_date': date_str,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'notional': notional,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'reason': reason,
            'p_up': position.get('p_up', 0.5),
            'confidence': position.get('confidence', 0.0)
        }

        # Log details
        logger.info(f"  {symbol}:")
        logger.info(f"    Entry: {entry_date} @ ${entry_price:.2f}")
        logger.info(f"    Exit:  {date_str} @ ${exit_price:.2f}")
        logger.info(f"    Gross PnL: ${gross_pnl:,.2f}")
        logger.info(f"    Net PnL:   ${net_pnl:,.2f}")
        logger.info(f"    Reason: {reason}")

        # Append to ledger
        append_to_ledger(trade, ledger_file)

    # Update positions state
    state['open_positions'] = positions_to_keep
    save_positions_state(state, state_file)

    logger.info(f"\nClosed {len(positions_to_close)} positions")
    logger.info(f"Remaining open positions: {len(positions_to_keep)}")

    # Summary statistics
    total_gross_pnl = sum(pnl for _, pnl in
                          [(p, compute_pnl(p, *get_prices(p['symbol'], p['entry_date'], date_str, returns_df),
                                          p.get('notional', 10000.0), cfg)[0])
                           for p, _ in positions_to_close])
    total_net_pnl = sum(pnl for _, pnl in
                        [(p, compute_pnl(p, *get_prices(p['symbol'], p['entry_date'], date_str, returns_df),
                                        p.get('notional', 10000.0), cfg)[1])
                         for p, _ in positions_to_close])

    if len(positions_to_close) > 0:
        # Recompute for summary (we already computed above)
        logger.info("\nClosing Summary:")
        logger.info(f"  Total Gross PnL: ${total_gross_pnl:,.2f}")
        logger.info(f"  Total Net PnL:   ${total_net_pnl:,.2f}")

    logger.info("=" * 60)
    logger.info("POSITION CLOSING COMPLETE")
    logger.info("=" * 60)


def main():
    """Main entry point for close positions script."""
    parser = argparse.ArgumentParser(description='Close positions based on exit criteria')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Closing date (YYYY-MM-DD). If not provided, uses last available date.'
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {args.config}")

    try:
        close_positions(cfg, close_date=args.date)
    except Exception as e:
        logger.error(f"Position closing failed: {e}")
        raise


if __name__ == '__main__':
    main()
