"""Daily paper trading workflow - run this once per day after market close."""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import load_config
from src.core.constants import Paths, TradingConstants
from src.core.data_loader import load_returns, load_windows
from src.core.logger import get_logger
from src.modeling.similarity import rank_analogs
from src.modeling.vote import vote
from src.modeling.windows import build_windows, normalize_window
from src.trading.paper_trading import PaperTradingPortfolio


logger = get_logger(__name__)


def get_current_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch current prices for symbols."""
    prices = {}

    logger.info(f"Fetching current prices for {len(symbols)} symbols...")

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d')

            if not hist.empty:
                prices[symbol] = float(hist['Close'].iloc[-1])
                logger.debug(f"{symbol}: ${prices[symbol]:.2f}")
            else:
                logger.warning(f"No price data for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")

    logger.info(f"Fetched prices for {len(prices)}/{len(symbols)} symbols")
    return prices


def generate_signals(cfg: dict) -> pd.DataFrame:
    """Generate trading signals for today."""
    logger.info("Generating trading signals...")

    # Load data
    returns_df = load_returns()
    windows_bank = load_windows()

    window_length = cfg['windows']['length']
    normalization = cfg['windows']['normalization']
    epsilon = cfg['windows'].get('epsilon', TradingConstants.EPSILON)
    similarity_metric = cfg['similarity']['metric']
    top_k = cfg['similarity']['top_k']
    min_sim = cfg['similarity'].get('min_sim', 0.0)
    vote_scheme = cfg['vote']['scheme']
    vote_threshold = cfg['vote']['threshold']
    abstain_if_below_k = cfg['vote']['abstain_if_below_k']

    # Use today as cutoff date
    cutoff_date = returns_df.index[-1]
    logger.info(f"Generating signals with data up to {cutoff_date.date()}")

    signals_list = []

    for symbol in returns_df.columns:
        symbol_returns = returns_df.loc[:cutoff_date, symbol]

        if len(symbol_returns) < window_length:
            continue

        recent_returns = symbol_returns.iloc[-window_length:]
        if recent_returns.isna().any():
            continue

        # Form and normalize target window
        try:
            target_vec = normalize_window(recent_returns.values, method=normalization, epsilon=epsilon)
        except (ValueError, ZeroDivisionError) as e:
            logger.debug(f"Failed to normalize window for {symbol}: {e}")
            continue

        # Rank analogs
        analogs_df = rank_analogs(
            target_vec=target_vec,
            bank_df=windows_bank,
            cutoff_date=cutoff_date,
            metric=similarity_metric,
            top_k=top_k,
            min_sim=min_sim,
            exclude_symbol=symbol
        )

        # Vote
        vote_result = vote(
            analogs_df=analogs_df,
            scheme=vote_scheme,
            threshold=vote_threshold,
            abstain_if_below_k=abstain_if_below_k
        )

        if vote_result['signal'] != 'ABSTAIN':
            signals_list.append({
                'symbol': symbol,
                'signal': vote_result['signal'],
                'p_up': vote_result['p_up'],
                'confidence': vote_result['confidence'],
                'n_analogs': vote_result['n_analogs']
            })

    signals_df = pd.DataFrame(signals_list)

    if len(signals_df) > 0:
        logger.info(f"Generated {len(signals_df)} signals: {signals_df['signal'].value_counts().to_dict()}")
    else:
        logger.warning("No signals generated")

    return signals_df


def run_daily_paper_trading(cfg: dict) -> None:
    """Execute daily paper trading workflow."""
    today = datetime.now().strftime('%Y-%m-%d')
    logger.info("=" * 60)
    logger.info(f"PAPER TRADING - {today}")
    logger.info("=" * 60)

    # Initialize portfolio
    initial_capital = cfg.get('paper_trading', {}).get('initial_capital', TradingConstants.DEFAULT_INITIAL_CAPITAL)
    portfolio = PaperTradingPortfolio(initial_capital=initial_capital)
    portfolio.load_state()

    # Get current prices for all symbols
    returns_df = load_returns()
    all_symbols = returns_df.columns.tolist()

    # Add symbols from open positions
    for symbol in portfolio.positions.keys():
        if symbol not in all_symbols:
            all_symbols.append(symbol)

    current_prices = get_current_prices(all_symbols)

    # Update open positions with current prices
    logger.info("Updating open positions...")
    portfolio.update_positions(current_prices)

    # Check exit conditions (simple: close after max_hold_days)
    max_hold_days = cfg.get('paper_trading', {}).get('max_hold_days', 5)
    positions_to_close = []

    for symbol, position in portfolio.positions.items():
        entry_date = pd.Timestamp(position.entry_date)
        days_held = (pd.Timestamp.now() - entry_date).days

        if days_held >= max_hold_days:
            positions_to_close.append(symbol)
            logger.info(f"Closing {symbol}: held for {days_held} days")

    # Close positions
    for symbol in positions_to_close:
        if symbol in current_prices:
            portfolio.close_position(symbol, today, current_prices[symbol])
        else:
            logger.warning(f"Cannot close {symbol}: no current price available")

    # Generate new signals
    signals_df = generate_signals(cfg)

    # Execute new trades (respect max_positions limit)
    max_positions = cfg['backtest']['max_positions']
    position_size = cfg.get('paper_trading', {}).get('position_size', TradingConstants.DEFAULT_POSITION_SIZE)

    open_slots = max_positions - len(portfolio.positions)

    if open_slots > 0 and len(signals_df) > 0:
        logger.info(f"Opening new positions: {open_slots} slots available")

        # Sort by confidence and take top signals
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
            else:
                logger.warning(f"Cannot open {symbol}: no current price available")
    else:
        logger.info(f"No new positions: {len(portfolio.positions)}/{max_positions} slots filled")

    # Record daily P&L
    portfolio.record_daily_pnl(today)

    # Save state
    portfolio.save_state()

    # Print summary
    logger.info("=" * 60)
    logger.info("PORTFOLIO SUMMARY")
    logger.info("=" * 60)
    summary = portfolio.get_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            logger.info(f"{key}: ${value:,.2f}" if 'pct' not in key else f"{key}: {value:.2f}%")
        else:
            logger.info(f"{key}: {value}")

    # Save trades log
    save_trades_log(portfolio, today)


def save_trades_log(portfolio: PaperTradingPortfolio, today: str) -> None:
    """Save trades and daily P&L to CSV files."""
    output_dir = Paths.RESULTS_PAPER_TRADING
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save closed trades
    if portfolio.closed_trades:
        trades_df = pd.DataFrame(portfolio.closed_trades)
        trades_file = output_dir / 'closed_trades.csv'
        trades_df.to_csv(trades_file, index=False)
        logger.info(f"Saved {len(trades_df)} closed trades to {trades_file}")

    # Save daily P&L
    if portfolio.daily_pnl:
        daily_df = pd.DataFrame(portfolio.daily_pnl)
        daily_file = output_dir / 'daily_pnl.csv'
        daily_df.to_csv(daily_file, index=False)
        logger.info(f"Saved daily P&L to {daily_file}")

    # Save open positions
    if portfolio.positions:
        positions_data = [pos.to_dict() for pos in portfolio.positions.values()]
        positions_df = pd.DataFrame(positions_data)
        positions_file = output_dir / f'open_positions_{today}.csv'
        positions_df.to_csv(positions_file, index=False)
        logger.info(f"Saved {len(positions_df)} open positions to {positions_file}")


def main():
    """Main entry point for paper trading."""
    try:
        cfg = load_config()
        run_daily_paper_trading(cfg)
        logger.info("Paper trading completed successfully!")
    except Exception as e:
        logger.error(f"Paper trading failed: {e}")
        raise


if __name__ == '__main__':
    main()
