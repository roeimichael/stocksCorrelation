#!/usr/bin/env python
"""
Daily Trading System Runner

Consolidates all daily operations into a single script:
1. Generate trading signals (live_daily)
2. Monitor open positions (monitor)
3. Close positions based on exit criteria (close_positions)

Usage:
    python daily_runner.py [--config config.yaml] [--date YYYY-MM-DD] [--skip-signals] [--skip-monitor] [--skip-close]

Examples:
    # Run full daily pipeline
    python daily_runner.py

    # Run with custom config and date
    python daily_runner.py --config my_config.yaml --date 2024-01-15

    # Run only signal generation
    python daily_runner.py --skip-monitor --skip-close
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.core.logger import get_logger
from src.dataio.live_append import append_new_bars
from src.dataio.prep import prepare_returns
from src.dataio.fetch import fetch_universe
from src.modeling.similarity import rank_analogs
from src.modeling.vote import vote
from src.modeling.windows import build_windows, normalize_window
from src.trading.monitor import (
    classify_alert,
    correlation_decay,
    directional_concordance,
    pattern_deviation_z,
    similarity_retention,
)


logger = get_logger(__name__)


# ==============================================================================
# State Management
# ==============================================================================

def load_positions_state(state_file: Path) -> dict:
    """Load positions state from JSON file."""
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {"open_positions": []}


def save_positions_state(state: dict, state_file: Path):
    """Save positions state to JSON file."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    logger.info(f"Saved positions state to {state_file}")


# ==============================================================================
# Step 1: Signal Generation
# ==============================================================================

def generate_live_signals(
    returns_df: pd.DataFrame,
    windows_bank: pd.DataFrame,
    cfg: dict,
    signal_date: pd.Timestamp
) -> pd.DataFrame:
    """Generate trading signals for all symbols on a given date."""
    window_length = cfg['windows']['length']
    normalization = cfg['windows']['normalization']
    similarity_metric = cfg['similarity']['metric']
    top_k = cfg['similarity']['top_k']
    min_sim = cfg['similarity'].get('min_sim', 0.0)
    vote_scheme = cfg['vote']['scheme']
    vote_threshold = cfg['vote']['threshold']
    abstain_if_below_k = cfg['vote']['abstain_if_below_k']

    cutoff_date = signal_date - pd.Timedelta(days=1)
    logger.info(f"Generating signals for {signal_date.date()}, cutoff_date={cutoff_date.date()}")

    signals_list = []

    for symbol in returns_df.columns:
        symbol_returns = returns_df.loc[:cutoff_date, symbol]

        if len(symbol_returns) < window_length:
            continue

        recent_returns = symbol_returns.iloc[-window_length:]
        if recent_returns.isna().any():
            continue

        target_window = recent_returns.values

        try:
            target_vec = normalize_window(target_window, method=normalization)
        except Exception as e:
            logger.warning(f"Failed to normalize target window for {symbol}: {e}")
            continue

        analogs_df = rank_analogs(
            target_vec=target_vec,
            bank_df=windows_bank,
            cutoff_date=cutoff_date,
            metric=similarity_metric,
            top_k=top_k,
            min_sim=min_sim,
            exclude_symbol=symbol
        )

        vote_result = vote(
            analogs_df=analogs_df,
            scheme=vote_scheme,
            threshold=vote_threshold,
            abstain_if_below_k=abstain_if_below_k
        )

        analogs_list = []
        for _, analog_row in analogs_df.iterrows():
            analogs_list.append({
                'symbol': analog_row['symbol'],
                'end_date': analog_row['end_date'].strftime('%Y-%m-%d'),
                'sim': float(analog_row['sim']),
                'label': int(analog_row['label'])
            })

        signals_list.append({
            'symbol': symbol,
            'p_up': vote_result['p_up'],
            'signal': vote_result['signal'],
            'confidence': vote_result['confidence'],
            'analogs': analogs_list
        })

    signals_df = pd.DataFrame(signals_list)

    if len(signals_df) > 0:
        signal_counts = signals_df['signal'].value_counts().to_dict()
        logger.info(f"Generated {len(signals_df)} signals: {signal_counts}")

    return signals_df


def run_signal_generation(cfg: dict, signal_date: str = None) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Run signal generation pipeline."""
    logger.info("=" * 80)
    logger.info("STEP 1: SIGNAL GENERATION")
    logger.info("=" * 80)

    # Append new bars
    logger.info("Appending new bars...")
    updated_symbols = append_new_bars(
        live_dir='data/live_ingest',
        raw_dir='data/raw'
    )
    if len(updated_symbols) > 0:
        logger.info(f"Updated {len(updated_symbols)} symbols: {updated_symbols}")
    else:
        logger.info("No new bars to ingest")

    # Prepare returns
    logger.info("Preparing returns...")
    all_symbols = fetch_universe(cfg)
    returns_df = prepare_returns(all_symbols, cfg)
    logger.info(f"Returns shape: {returns_df.shape}")

    # Build windows
    logger.info("Building windows...")
    windows_df = build_windows(returns_df, cfg)
    logger.info(f"Built {len(windows_df)} windows for {windows_df['symbol'].nunique()} symbols")

    # Determine signal date
    if signal_date:
        signal_ts = pd.Timestamp(signal_date)
        logger.info(f"Using specified signal date: {signal_ts.date()}")
    else:
        signal_ts = returns_df.index[-1]
        logger.info(f"Using last available date: {signal_ts.date()}")

    # Generate signals
    signals_df = generate_live_signals(returns_df, windows_df, cfg, signal_ts)

    # Save signals
    output_dir = Path('results/live')
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = signal_ts.strftime('%Y-%m-%d')
    signals_file = output_dir / f'signals_{date_str}.csv'

    signals_for_csv = signals_df.copy()
    if 'analogs' in signals_for_csv.columns:
        signals_for_csv['analogs'] = signals_for_csv['analogs'].apply(json.dumps)

    signals_for_csv.to_csv(signals_file, index=False)
    logger.info(f"Saved signals to {signals_file}")

    # Update positions state
    state_file = output_dir / 'positions_state.json'
    state = load_positions_state(state_file)

    active_signals = signals_df[signals_df['signal'] != 'ABSTAIN']

    new_positions = []
    for _, signal_row in active_signals.iterrows():
        position = {
            'symbol': signal_row['symbol'],
            'entry_date': date_str,
            'side': signal_row['signal'],
            'p_up': float(signal_row['p_up']),
            'confidence': float(signal_row['confidence']),
            'analogs': signal_row['analogs'],
            'notional': 10000.0  # Default position size
        }
        new_positions.append(position)

    state['open_positions'] = new_positions
    save_positions_state(state, state_file)

    logger.info(f"Updated positions state: {len(new_positions)} open positions")

    if len(active_signals) > 0:
        logger.info("\nSignal Summary:")
        logger.info(f"  UP signals: {len(active_signals[active_signals['signal'] == 'UP'])}")
        logger.info(f"  DOWN signals: {len(active_signals[active_signals['signal'] == 'DOWN'])}")
        logger.info(f"  Avg confidence: {active_signals['confidence'].mean():.2%}")

    return signal_ts, signals_df


# ==============================================================================
# Step 2: Position Monitoring
# ==============================================================================

def run_position_monitoring(cfg: dict, monitor_date: pd.Timestamp) -> pd.DataFrame:
    """Monitor all open positions and generate alerts."""
    logger.info("=" * 80)
    logger.info("STEP 2: POSITION MONITORING")
    logger.info("=" * 80)

    state_file = Path('results/live/positions_state.json')
    state = load_positions_state(state_file)

    open_positions = state.get('open_positions', [])

    if not open_positions:
        logger.info("No open positions to monitor")
        return pd.DataFrame()

    logger.info(f"Monitoring {len(open_positions)} open positions")

    # Load data
    returns_df = pd.read_parquet('data/processed/returns.parquet')
    windows_bank = pd.read_parquet('data/processed/windows.parquet')

    alerts = []

    for i, position in enumerate(open_positions):
        symbol = position['symbol']
        entry_date = position['entry_date']
        side = position['side']

        logger.info(f"[{i+1}/{len(open_positions)}] Monitoring {symbol} (entered {entry_date}, side={side})")

        try:
            sr = similarity_retention(position, monitor_date, returns_df, windows_bank, cfg)
            dc = directional_concordance(position, monitor_date, returns_df, cfg)
            corr_today, delta_corr = correlation_decay(position, monitor_date, returns_df, cfg)
            pdz = pattern_deviation_z(position, monitor_date, returns_df, cfg)

            alert_level = classify_alert(sr, dc, delta_corr, pdz, cfg)

            logger.info(f"  SR={sr:.3f}, DC={dc:.3f}, Corr={corr_today:.3f}, Δ={delta_corr:.3f}, PDZ={pdz:.3f} → {alert_level}")

            alert = {
                'date': monitor_date.strftime('%Y-%m-%d'),
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

            position['last_monitor_date'] = monitor_date.strftime('%Y-%m-%d')
            position['metrics'] = {
                'similarity_retention': sr,
                'directional_concordance': dc,
                'correlation_today': corr_today,
                'correlation_delta_3d': delta_corr,
                'pattern_deviation_z': pdz
            }
            position['alert_level'] = alert_level

        except Exception as e:
            logger.error(f"Failed to monitor {symbol}: {e}")
            alert = {
                'date': monitor_date.strftime('%Y-%m-%d'),
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

    # Save alerts
    date_str = monitor_date.strftime('%Y-%m-%d')
    alerts_file = Path(f'results/live/alerts_{date_str}.csv')
    alerts_file.parent.mkdir(parents=True, exist_ok=True)

    alerts_df = pd.DataFrame(alerts)
    alerts_df.to_csv(alerts_file, index=False)

    logger.info(f"Saved alerts to {alerts_file}")

    save_positions_state(state, state_file)

    if len(alerts_df) > 0:
        alert_counts = alerts_df['alert_level'].value_counts().to_dict()
        logger.info("\nAlert Summary:")
        for level in ['GREEN', 'YELLOW', 'RED', 'ERROR']:
            count = alert_counts.get(level, 0)
            if count > 0:
                logger.info(f"  {level}: {count}")

    return alerts_df


# ==============================================================================
# Step 3: Position Closing
# ==============================================================================

def should_close_position(position: dict, signals_df: pd.DataFrame, cfg: dict) -> tuple[bool, str]:
    """Determine if a position should be closed."""
    close_policy = cfg.get('close_policy', {})
    close_on_red = close_policy.get('close_on_red', True)
    close_on_reverse = close_policy.get('close_on_reverse', True)
    close_on_abstain = close_policy.get('close_on_abstain', False)

    symbol = position['symbol']
    side = position['side']
    alert_level = position.get('alert_level', 'GREEN')

    if close_on_red and alert_level == 'RED':
        return True, 'RED_ALERT'

    if close_on_reverse and len(signals_df) > 0:
        symbol_signals = signals_df[signals_df['symbol'] == symbol]
        if len(symbol_signals) > 0:
            latest_signal = symbol_signals.iloc[0]['signal']
            if (side == 'UP' and latest_signal == 'DOWN') or (side == 'DOWN' and latest_signal == 'UP'):
                return True, 'REVERSE_SIGNAL'
            if close_on_abstain and latest_signal == 'ABSTAIN':
                return True, 'ABSTAIN_SIGNAL'

    return False, ''


def get_prices(symbol: str, entry_date: str, exit_date: str, returns_df: pd.DataFrame) -> tuple[float, float]:
    """Get entry and exit prices for a position."""
    entry_ts = pd.Timestamp(entry_date)
    exit_ts = pd.Timestamp(exit_date)

    if symbol not in returns_df.columns:
        logger.warning(f"Symbol {symbol} not found in returns_df")
        return 100.0, 100.0

    try:
        returns_between = returns_df.loc[entry_ts:exit_ts, symbol]
        cum_return = (1 + returns_between).prod() - 1
        entry_price = 100.0
        exit_price = entry_price * (1 + cum_return)
        return entry_price, exit_price
    except Exception as e:
        logger.warning(f"Error computing prices for {symbol}: {e}")
        return 100.0, 100.0


def compute_pnl(position: dict, entry_price: float, exit_price: float, notional: float, cfg: dict) -> tuple[float, float]:
    """Compute gross and net PnL for a position."""
    side = position['side']
    costs_bps = cfg.get('backtest', {}).get('costs_bps', 5.0)
    slippage_bps = cfg.get('backtest', {}).get('slippage_bps', 2.0)

    if side == 'UP':
        position_return = (exit_price - entry_price) / entry_price
    else:
        position_return = -(exit_price - entry_price) / entry_price

    gross_pnl = notional * position_return
    total_costs_bps = (costs_bps + slippage_bps) * 2
    transaction_costs = notional * (total_costs_bps / 10000.0)
    net_pnl = gross_pnl - transaction_costs

    return float(gross_pnl), float(net_pnl)


def append_to_ledger(trade: dict, ledger_file: Path):
    """Append trade to portfolio ledger CSV."""
    ledger_file.parent.mkdir(parents=True, exist_ok=True)

    trade_df = pd.DataFrame([trade])

    if ledger_file.exists():
        existing_ledger = pd.read_csv(ledger_file)
        updated_ledger = pd.concat([existing_ledger, trade_df], ignore_index=True)
        updated_ledger.to_csv(ledger_file, index=False)
    else:
        trade_df.to_csv(ledger_file, index=False)


def run_position_closing(cfg: dict, close_date: pd.Timestamp, signals_df: pd.DataFrame):
    """Close positions that meet exit criteria."""
    logger.info("=" * 80)
    logger.info("STEP 3: POSITION CLOSING")
    logger.info("=" * 80)

    state_file = Path('results/live/positions_state.json')
    state = load_positions_state(state_file)

    open_positions = state.get('open_positions', [])

    if not open_positions:
        logger.info("No open positions to close")
        return

    logger.info(f"Checking {len(open_positions)} open positions for closing")

    returns_df = pd.read_parquet('data/processed/returns.parquet')

    positions_to_close = []
    positions_to_keep = []

    for position in open_positions:
        symbol = position['symbol']
        should_close, reason = should_close_position(position, signals_df, cfg)

        if should_close:
            logger.info(f"Closing {symbol}: {reason}")
            positions_to_close.append((position, reason))
        else:
            positions_to_keep.append(position)

    if not positions_to_close:
        logger.info("No positions to close")
        return

    logger.info(f"Closing {len(positions_to_close)} positions")

    ledger_file = Path('results/live/portfolio_ledger.csv')
    date_str = close_date.strftime('%Y-%m-%d')

    for position, reason in positions_to_close:
        symbol = position['symbol']
        entry_date = position['entry_date']
        side = position['side']
        notional = position.get('notional', 10000.0)

        entry_price, exit_price = get_prices(symbol, entry_date, date_str, returns_df)
        gross_pnl, net_pnl = compute_pnl(position, entry_price, exit_price, notional, cfg)

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

        logger.info(f"  {symbol}: Entry={entry_date}@${entry_price:.2f}, Exit={date_str}@${exit_price:.2f}, Net PnL=${net_pnl:,.2f}, Reason={reason}")

        append_to_ledger(trade, ledger_file)

    state['open_positions'] = positions_to_keep
    save_positions_state(state, state_file)

    logger.info(f"\nClosed {len(positions_to_close)} positions, {len(positions_to_keep)} remaining open")


# ==============================================================================
# Main Runner
# ==============================================================================

def main():
    """Main entry point for daily trading system."""
    parser = argparse.ArgumentParser(
        description='Daily Trading System - Consolidated Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python daily_runner.py                           # Run full daily pipeline
  python daily_runner.py --date 2024-01-15         # Run for specific date
  python daily_runner.py --skip-monitor            # Skip monitoring step
  python daily_runner.py --config my_config.yaml   # Use custom config
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Trading date (YYYY-MM-DD). If not provided, uses last available date.'
    )
    parser.add_argument(
        '--skip-signals',
        action='store_true',
        help='Skip signal generation step'
    )
    parser.add_argument(
        '--skip-monitor',
        action='store_true',
        help='Skip position monitoring step'
    )
    parser.add_argument(
        '--skip-close',
        action='store_true',
        help='Skip position closing step'
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {args.config}")

    logger.info("=" * 80)
    logger.info("DAILY TRADING SYSTEM - START")
    logger.info("=" * 80)

    try:
        # Step 1: Signal Generation
        if not args.skip_signals:
            signal_ts, signals_df = run_signal_generation(cfg, signal_date=args.date)
        else:
            logger.info("Skipping signal generation")
            if args.date:
                signal_ts = pd.Timestamp(args.date)
            else:
                returns_df = pd.read_parquet('data/processed/returns.parquet')
                signal_ts = returns_df.index[-1]

            # Load existing signals
            date_str = signal_ts.strftime('%Y-%m-%d')
            signals_file = Path(f'results/live/signals_{date_str}.csv')
            if signals_file.exists():
                signals_df = pd.read_csv(signals_file)
            else:
                signals_df = pd.DataFrame()

        # Step 2: Position Monitoring
        if not args.skip_monitor:
            run_position_monitoring(cfg, signal_ts)
        else:
            logger.info("\n" + "=" * 80)
            logger.info("Skipping position monitoring")

        # Step 3: Position Closing
        if not args.skip_close:
            run_position_closing(cfg, signal_ts, signals_df)
        else:
            logger.info("\n" + "=" * 80)
            logger.info("Skipping position closing")

        logger.info("\n" + "=" * 80)
        logger.info("DAILY TRADING SYSTEM - COMPLETE")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Daily trading system failed: {e}")
        raise


if __name__ == '__main__':
    main()
