"""Minimal backtesting engine with costs and slippage."""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from src.modeling.windows import Window
from src.modeling.vote import Signal, signal_to_position, filter_top_signals
from src.core.logger import get_logger

logger = get_logger()


@dataclass
class Trade:
    """Record of a single trade."""
    date: pd.Timestamp
    symbol: str
    direction: int  # 1 = long, -1 = short
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    return_pct: float
    costs: float


class BacktestEngine:
    """
    Simple walk-forward backtester for daily signals.
    Entry at open, exit at close on the same day.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        config: dict,
        initial_capital: float = 100000.0
    ):
        """
        Initialize backtest engine.

        Args:
            prices: DataFrame with prices (dates x tickers)
            returns: DataFrame with returns (dates x tickers)
            config: Configuration dictionary
            initial_capital: Starting capital
        """
        self.prices = prices
        self.returns = returns
        self.config = config
        self.initial_capital = initial_capital

        # Backtest config
        self.entry = config['backtest']['entry']
        self.exit = config['backtest']['exit']
        self.costs_bps = config['backtest']['costs_bps']
        self.slippage_bps = config['backtest']['slippage_bps']
        self.max_positions = config['backtest']['max_positions']
        self.position_pct = config['backtest']['position_pct']
        self.side = config['backtest']['side']

        # Results
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.dates: List[pd.Timestamp] = []

    def get_price(self, date: pd.Timestamp, symbol: str, price_type: str) -> float:
        """
        Get price for a symbol on a date.

        Args:
            date: Trade date
            symbol: Ticker symbol
            price_type: 'open' or 'close'

        Returns:
            Price (uses close as proxy if open not available)
        """
        if date not in self.prices.index or symbol not in self.prices.columns:
            return np.nan

        # In this simple version, we only have adjusted close
        # In production, you'd have OHLC data
        return self.prices.loc[date, symbol]

    def compute_costs(self, entry_price: float, shares: float) -> float:
        """
        Compute transaction costs.

        Args:
            entry_price: Entry price
            shares: Number of shares

        Returns:
            Total costs
        """
        position_value = entry_price * abs(shares)
        costs = position_value * (self.costs_bps + self.slippage_bps) / 10000
        return costs

    def execute_trade(
        self,
        date: pd.Timestamp,
        symbol: str,
        direction: int
    ) -> Trade:
        """
        Execute a single trade.

        Args:
            date: Trade date
            symbol: Ticker symbol
            direction: 1 for long, -1 for short

        Returns:
            Trade record
        """
        # Get prices (using close as proxy for open)
        entry_price = self.get_price(date, symbol, self.entry)
        exit_price = self.get_price(date, symbol, self.exit)

        if np.isnan(entry_price) or np.isnan(exit_price):
            # Can't execute trade
            return None

        # Calculate position size
        current_equity = self.equity_curve[-1]
        position_value = current_equity * self.position_pct
        shares = position_value / entry_price

        # Calculate costs
        costs = self.compute_costs(entry_price, shares)

        # Calculate PnL
        if direction == 1:  # Long
            price_return = (exit_price - entry_price) / entry_price
        else:  # Short
            price_return = (entry_price - exit_price) / entry_price

        gross_pnl = position_value * price_return
        net_pnl = gross_pnl - costs

        trade = Trade(
            date=date,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            shares=shares,
            pnl=net_pnl,
            return_pct=net_pnl / current_equity,
            costs=costs
        )

        return trade

    def run_backtest(
        self,
        signals_by_date: Dict[pd.Timestamp, List[Tuple[Window, Signal]]]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Run backtest for all signals.

        Args:
            signals_by_date: Dictionary mapping dates to (window, signal) lists

        Returns:
            Tuple of (trades DataFrame, equity curve Series)
        """
        logger.info("Running backtest")

        dates = sorted(signals_by_date.keys())

        for date in dates:
            daily_signals = signals_by_date[date]

            # Filter to top signals
            if len(daily_signals) > self.max_positions:
                daily_signals = filter_top_signals(
                    [s for _, s in daily_signals],
                    [w for w, _ in daily_signals],
                    self.max_positions
                )

            # Execute trades
            daily_pnl = 0.0

            for window, signal in daily_signals:
                position = signal_to_position(signal, side=self.side)

                if position == 0:
                    continue

                trade = self.execute_trade(date, window.symbol, position)

                if trade:
                    self.trades.append(trade)
                    daily_pnl += trade.pnl

            # Update equity
            new_equity = self.equity_curve[-1] + daily_pnl
            self.equity_curve.append(new_equity)
            self.dates.append(date)

        logger.info(f"Backtest complete: {len(self.trades)} trades executed")

        # Convert to DataFrames
        if self.trades:
            trades_df = pd.DataFrame([
                {
                    'date': t.date,
                    'symbol': t.symbol,
                    'direction': 'LONG' if t.direction == 1 else 'SHORT',
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'shares': t.shares,
                    'pnl': t.pnl,
                    'return_pct': t.return_pct,
                    'costs': t.costs
                }
                for t in self.trades
            ])
        else:
            trades_df = pd.DataFrame()

        equity_series = pd.Series(
            self.equity_curve[1:],  # Skip initial equity
            index=self.dates
        )

        return trades_df, equity_series

    def save_results(
        self,
        trades_df: pd.DataFrame,
        equity_series: pd.Series,
        output_dir: str = 'results/backtests'
    ):
        """
        Save backtest results.

        Args:
            trades_df: Trades DataFrame
            equity_series: Equity curve Series
            output_dir: Output directory
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save trades
        if len(trades_df) > 0:
            trades_df.to_csv(output_path / 'trades.csv', index=False)
            logger.info(f"Saved trades to {output_path / 'trades.csv'}")

        # Save equity curve
        if len(equity_series) > 0:
            equity_series.to_csv(output_path / 'equity_curve.csv', header=['equity'])
            logger.info(f"Saved equity curve to {output_path / 'equity_curve.csv'}")


def walk_forward_backtest(
    windows: List[Window],
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    config: dict,
    test_start_date: pd.Timestamp,
    initial_capital: float = 100000.0
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Perform walk-forward backtest.

    Args:
        windows: All available windows
        returns: Returns DataFrame
        prices: Prices DataFrame
        config: Configuration dictionary
        test_start_date: Start date for testing
        initial_capital: Starting capital

    Returns:
        Tuple of (trades DataFrame, equity Series, metrics dict)
    """
    from src.modeling.similarity import find_top_analogs
    from src.modeling.vote import generate_signal, signal_to_position
    from src.evals.metrics import compute_returns_metrics, compute_trade_statistics

    logger.info(f"Walk-forward backtest starting from {test_start_date.date()}")

    # Initialize engine
    engine = BacktestEngine(prices, returns, config, initial_capital)

    # Group windows by symbol and end date
    windows_by_date_symbol = {}
    for w in windows:
        key = (w.end_date, w.symbol)
        windows_by_date_symbol[key] = w

    # Get test dates
    test_dates = returns.index[returns.index >= test_start_date]

    signals_by_date = {}

    for date_idx, date in enumerate(test_dates):
        # Get candidate pool (strictly before this date)
        candidate_pool = [w for w in windows if w.end_date < date and w.label != -1]

        if len(candidate_pool) < config['similarity']['top_k']:
            continue

        # Get target windows for this date (as of previous day)
        target_windows = []
        for symbol in returns.columns:
            key = (date - pd.Timedelta(days=1), symbol)
            if key in windows_by_date_symbol:
                target_windows.append(windows_by_date_symbol[key])

        if not target_windows:
            continue

        # Generate signals
        daily_signals = []

        for target_window in target_windows:
            analogs = find_top_analogs(
                target_window,
                candidate_pool,
                top_k=config['similarity']['top_k'],
                metric=config['similarity']['metric'],
                min_similarity=config['similarity']['min_sim']
            )

            signal = generate_signal(analogs, config)

            if signal.direction != 'ABSTAIN':
                daily_signals.append((target_window, signal))

        if daily_signals:
            signals_by_date[date] = daily_signals

        if (date_idx + 1) % 50 == 0:
            logger.info(f"Processed {date_idx + 1}/{len(test_dates)} test dates")

    # Run backtest
    trades_df, equity_series = engine.run_backtest(signals_by_date)

    # Compute metrics
    metrics = {}

    if len(trades_df) > 0:
        # Returns metrics
        returns_metrics = compute_returns_metrics(
            pd.Series(trades_df['return_pct'].values)
        )
        metrics.update(returns_metrics)

        # Trade statistics
        trade_stats = compute_trade_statistics(trades_df)
        metrics.update(trade_stats)

    # Save results
    engine.save_results(trades_df, equity_series)

    return trades_df, equity_series, metrics
