"""Run paper trading with multiple parameter configurations simultaneously."""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import load_config
from src.core.constants import Paths, TradingConstants
from src.core.data_loader import load_returns
from src.core.logger import get_logger
from src.trading.paper_trading import PaperTradingPortfolio


logger = get_logger(__name__)


# Define multiple strategy configurations to test in parallel
STRATEGIES = {
    'conservative': {
        'windows': {'length': 10, 'normalization': 'zscore'},
        'similarity': {'metric': 'pearson', 'top_k': 30, 'min_sim': 0.30},
        'vote': {'threshold': 0.75, 'abstain_if_below_k': 15},
        'description': 'Conservative: High threshold, more analogs, stricter similarity'
    },
    'moderate': {
        'windows': {'length': 10, 'normalization': 'zscore'},
        'similarity': {'metric': 'pearson', 'top_k': 25, 'min_sim': 0.20},
        'vote': {'threshold': 0.70, 'abstain_if_below_k': 10},
        'description': 'Moderate: Default balanced parameters'
    },
    'aggressive': {
        'windows': {'length': 10, 'normalization': 'zscore'},
        'similarity': {'metric': 'pearson', 'top_k': 20, 'min_sim': 0.15},
        'vote': {'threshold': 0.65, 'abstain_if_below_k': 8},
        'description': 'Aggressive: Lower threshold, fewer analogs, more trades'
    },
    'spearman_rank': {
        'windows': {'length': 10, 'normalization': 'rank'},
        'similarity': {'metric': 'spearman', 'top_k': 25, 'min_sim': 0.20},
        'vote': {'threshold': 0.70, 'abstain_if_below_k': 10},
        'description': 'Rank-based: Spearman correlation with rank normalization'
    },
    'short_window': {
        'windows': {'length': 5, 'normalization': 'zscore'},
        'similarity': {'metric': 'pearson', 'top_k': 25, 'min_sim': 0.20},
        'vote': {'threshold': 0.70, 'abstain_if_below_k': 10},
        'description': 'Short window: 5-day patterns instead of 10'
    }
}


def run_strategy(strategy_name: str, strategy_config: dict, base_config: dict) -> dict:
    """Run paper trading for a single strategy configuration."""
    logger.info(f"\n{'='*60}")
    logger.info(f"STRATEGY: {strategy_name}")
    logger.info(f"Description: {strategy_config['description']}")
    logger.info(f"{'='*60}")

    # Merge strategy config with base config
    cfg = base_config.copy()
    cfg['windows'].update(strategy_config.get('windows', {}))
    cfg['similarity'].update(strategy_config.get('similarity', {}))
    cfg['vote'].update(strategy_config.get('vote', {}))

    # Use separate state file for each strategy
    state_file = f"{strategy_name}_portfolio_state.json"
    initial_capital = cfg.get('paper_trading', {}).get('initial_capital', TradingConstants.DEFAULT_INITIAL_CAPITAL)

    # Initialize portfolio for this strategy
    portfolio = PaperTradingPortfolio(
        initial_capital=initial_capital,
        state_file=state_file
    )
    portfolio.load_state()

    # Import here to avoid circular imports
    from scripts.paper_trade_daily import (
        get_current_prices,
        generate_signals,
        save_trades_log
    )

    today = datetime.now().strftime('%Y-%m-%d')

    # Get current prices
    returns_df = load_returns()
    all_symbols = returns_df.columns.tolist()
    for symbol in portfolio.positions.keys():
        if symbol not in all_symbols:
            all_symbols.append(symbol)

    current_prices = get_current_prices(all_symbols)

    # Update positions
    portfolio.update_positions(current_prices)

    # Close old positions
    max_hold_days = cfg.get('paper_trading', {}).get('max_hold_days', 5)
    positions_to_close = []

    for symbol, position in portfolio.positions.items():
        entry_date = pd.Timestamp(position.entry_date)
        days_held = (pd.Timestamp.now() - entry_date).days
        if days_held >= max_hold_days:
            positions_to_close.append(symbol)

    for symbol in positions_to_close:
        if symbol in current_prices:
            portfolio.close_position(symbol, today, current_prices[symbol])

    # Generate signals with this strategy's config
    signals_df = generate_signals(cfg)

    # Open new positions
    max_positions = cfg['backtest']['max_positions']
    position_size = cfg.get('paper_trading', {}).get('position_size', TradingConstants.DEFAULT_POSITION_SIZE)
    open_slots = max_positions - len(portfolio.positions)

    if open_slots > 0 and len(signals_df) > 0:
        signals_df = signals_df.sort_values('confidence', ascending=False)
        new_signals = signals_df.head(open_slots)

        for _, signal in new_signals.iterrows():
            symbol = signal['symbol']
            if symbol in current_prices:
                portfolio.open_position(
                    symbol=symbol,
                    signal=signal['signal'],
                    entry_date=today,
                    entry_price=current_prices[symbol],
                    position_size=position_size
                )

    # Record daily P&L
    portfolio.record_daily_pnl(today)

    # Save state
    portfolio.save_state()

    # Save logs to strategy-specific directory
    output_dir = Paths.RESULTS_PAPER_TRADING / strategy_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if portfolio.closed_trades:
        trades_df = pd.DataFrame(portfolio.closed_trades)
        trades_df.to_csv(output_dir / 'closed_trades.csv', index=False)

    if portfolio.daily_pnl:
        daily_df = pd.DataFrame(portfolio.daily_pnl)
        daily_df.to_csv(output_dir / 'daily_pnl.csv', index=False)

    if portfolio.positions:
        positions_data = [pos.to_dict() for pos in portfolio.positions.values()]
        positions_df = pd.DataFrame(positions_data)
        positions_df.to_csv(output_dir / f'open_positions_{today}.csv', index=False)

    # Get summary
    summary = portfolio.get_summary()
    summary['strategy'] = strategy_name

    logger.info(f"\nStrategy Summary:")
    logger.info(f"  Portfolio Value: ${summary['current_value']:,.2f}")
    logger.info(f"  Total P&L: ${summary['total_pnl']:,.2f}")
    logger.info(f"  Total Return: {summary['total_return_pct']:.2f}%")
    logger.info(f"  Open Positions: {summary['num_open_positions']}")
    logger.info(f"  Closed Trades: {summary['num_closed_trades']}")
    if 'win_rate' in summary:
        logger.info(f"  Win Rate: {summary['win_rate']*100:.1f}%")

    return summary


def generate_comparison_report(summaries: list[dict]) -> None:
    """Generate comparison report across all strategies."""
    logger.info(f"\n{'='*80}")
    logger.info("MULTI-STRATEGY COMPARISON")
    logger.info(f"{'='*80}\n")

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(summaries)

    # Sort by total return
    comparison_df = comparison_df.sort_values('total_return_pct', ascending=False)

    # Display formatted table
    logger.info(f"\n{'Strategy':<20} {'Return %':<12} {'P&L':<15} {'Positions':<12} {'Trades':<10} {'Win Rate':<10}")
    logger.info("-" * 80)

    for _, row in comparison_df.iterrows():
        win_rate = f"{row.get('win_rate', 0)*100:.1f}%" if 'win_rate' in row and row.get('num_closed_trades', 0) > 0 else "N/A"
        logger.info(f"{row['strategy']:<20} {row['total_return_pct']:>10.2f}% ${row['total_pnl']:>13,.2f} "
              f"{row['num_open_positions']:>11} {row['num_closed_trades']:>9} {win_rate:>9}")

    # Save comparison to CSV
    output_file = Paths.RESULTS_PAPER_TRADING / 'strategy_comparison.csv'
    comparison_df.to_csv(output_file, index=False)
    logger.info(f"\nComparison saved to: {output_file}")

    # Identify best strategy
    best_strategy = comparison_df.iloc[0]
    logger.info(f"\nBEST PERFORMING STRATEGY: {best_strategy['strategy']}")
    logger.info(f"   Return: {best_strategy['total_return_pct']:.2f}%")
    logger.info(f"   P&L: ${best_strategy['total_pnl']:,.2f}")


def main():
    """Main entry point for multi-strategy paper trading."""
    try:
        logger.info("Starting multi-strategy paper trading...")

        # Load base configuration
        base_config = load_config()

        # Run all strategies
        summaries = []
        for strategy_name, strategy_config in STRATEGIES.items():
            summary = run_strategy(strategy_name, strategy_config, base_config)
            summaries.append(summary)

        # Generate comparison report
        generate_comparison_report(summaries)

        logger.info("\nMulti-strategy paper trading completed successfully!")

    except Exception as e:
        logger.error(f"Multi-strategy paper trading failed: {e}")
        raise


if __name__ == '__main__':
    main()
