"""Paper trading engine for simulating live trades without real money."""
import json
from datetime import datetime
from typing import Any

import pandas as pd

from src.core.constants import Paths, TradingConstants
from src.core.logger import get_logger


logger = get_logger(__name__)


class Position:
    """Represents a single trading position."""

    def __init__(
        self,
        symbol: str,
        signal: str,
        entry_date: str,
        entry_price: float,
        shares: int,
        notional: float
    ):
        self.symbol = symbol
        self.signal = signal  # UP or DOWN
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.notional = notional
        self.current_price = entry_price
        self.unrealized_pnl = 0.0

    def update_price(self, current_price: float) -> None:
        """Update current price and unrealized P&L."""
        self.current_price = current_price

        if self.signal == 'UP':
            # Long position: profit when price goes up
            self.unrealized_pnl = self.shares * (current_price - self.entry_price)
        else:
            # Short position: profit when price goes down
            self.unrealized_pnl = self.shares * (self.entry_price - current_price)

    def to_dict(self) -> dict:
        """Convert position to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'signal': self.signal,
            'entry_date': self.entry_date,
            'entry_price': self.entry_price,
            'shares': self.shares,
            'notional': self.notional,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Position':
        """Create position from dictionary."""
        pos = cls(
            symbol=data['symbol'],
            signal=data['signal'],
            entry_date=data['entry_date'],
            entry_price=data['entry_price'],
            shares=data['shares'],
            notional=data['notional']
        )
        pos.current_price = data.get('current_price', data['entry_price'])
        pos.unrealized_pnl = data.get('unrealized_pnl', 0.0)
        return pos


class PaperTradingPortfolio:
    """Manages portfolio state for paper trading."""

    def __init__(
        self,
        initial_capital: float = TradingConstants.DEFAULT_INITIAL_CAPITAL,
        state_file: str = None
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.closed_trades: list[dict] = []
        self.daily_pnl: list[dict] = []
        self.state_file = Paths.PORTFOLIO_STATE if state_file is None else Paths.DATA_PAPER_TRADING / state_file
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def save_state(self) -> None:
        """Save portfolio state to disk."""
        state = {
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
            'closed_trades': self.closed_trades,
            'daily_pnl': self.daily_pnl
        }

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved portfolio state to {self.state_file}")

    def load_state(self) -> None:
        """Load portfolio state from disk."""
        if not self.state_file.exists():
            logger.info("No existing portfolio state found, starting fresh")
            return

        try:
            with open(self.state_file) as f:
                state = json.load(f)

            self.initial_capital = state.get('initial_capital', self.initial_capital)
            self.cash = state['cash']
            self.positions = {
                symbol: Position.from_dict(pos_data)
                for symbol, pos_data in state.get('positions', {}).items()
            }
            self.closed_trades = state.get('closed_trades', [])
            self.daily_pnl = state.get('daily_pnl', [])

            logger.info(f"Loaded portfolio state: {len(self.positions)} open positions, ${self.cash:,.2f} cash")
        except Exception as e:
            logger.error(f"Error loading portfolio state: {e}")
            logger.warning("Starting with fresh portfolio")

    def open_position(
        self,
        symbol: str,
        signal: str,
        entry_date: str,
        entry_price: float,
        position_size: float
    ) -> bool:
        """Open a new position."""
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}, skipping")
            return False

        # Calculate shares based on position size
        notional = position_size
        shares = int(notional / entry_price)

        if shares == 0:
            logger.warning(f"Position size too small for {symbol}, skipping")
            return False

        # Check if we have enough cash
        required_cash = shares * entry_price
        if required_cash > self.cash:
            logger.warning(f"Insufficient cash for {symbol}: need ${required_cash:,.2f}, have ${self.cash:,.2f}")
            return False

        # Create position
        position = Position(
            symbol=symbol,
            signal=signal,
            entry_date=entry_date,
            entry_price=entry_price,
            shares=shares,
            notional=notional
        )

        self.positions[symbol] = position
        self.cash -= required_cash

        logger.info(f"Opened {signal} position: {symbol} @ ${entry_price:.2f} x {shares} shares = ${required_cash:,.2f}")

        return True

    def close_position(self, symbol: str, exit_date: str, exit_price: float) -> dict | None:
        """Close an existing position."""
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return None

        position = self.positions[symbol]

        # Calculate realized P&L
        if position.signal == 'UP':
            realized_pnl = position.shares * (exit_price - position.entry_price)
        else:
            realized_pnl = position.shares * (position.entry_price - exit_price)

        # Return cash
        cash_returned = position.shares * exit_price
        self.cash += cash_returned

        # Record trade
        trade = {
            'symbol': symbol,
            'signal': position.signal,
            'entry_date': position.entry_date,
            'entry_price': position.entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'shares': position.shares,
            'notional': position.notional,
            'realized_pnl': realized_pnl,
            'return_pct': (realized_pnl / position.notional) * 100
        }

        self.closed_trades.append(trade)
        del self.positions[symbol]

        logger.info(f"Closed {position.signal} position: {symbol} @ ${exit_price:.2f}, P&L: ${realized_pnl:,.2f}")

        return trade

    def update_positions(self, prices: dict[str, float]) -> None:
        """Update all positions with current prices."""
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])

    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value (cash + positions)."""
        positions_value = sum(
            pos.shares * pos.current_price
            for pos in self.positions.values()
        )
        return self.cash + positions_value

    def get_total_pnl(self) -> float:
        """Calculate total P&L (realized + unrealized)."""
        realized_pnl = sum(trade['realized_pnl'] for trade in self.closed_trades)
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        return realized_pnl + unrealized_pnl

    def record_daily_pnl(self, date: str) -> None:
        """Record daily P&L snapshot."""
        portfolio_value = self.get_portfolio_value()
        total_pnl = self.get_total_pnl()

        self.daily_pnl.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions_value': portfolio_value - self.cash,
            'total_pnl': total_pnl,
            'total_return_pct': ((portfolio_value - self.initial_capital) / self.initial_capital) * 100,
            'num_positions': len(self.positions)
        })

        logger.info(f"Daily snapshot: Portfolio=${portfolio_value:,.2f}, P&L=${total_pnl:,.2f}, Positions={len(self.positions)}")

    def get_summary(self) -> dict[str, Any]:
        """Get portfolio summary statistics."""
        portfolio_value = self.get_portfolio_value()
        total_pnl = self.get_total_pnl()

        summary = {
            'initial_capital': self.initial_capital,
            'current_value': portfolio_value,
            'cash': self.cash,
            'total_pnl': total_pnl,
            'total_return_pct': ((portfolio_value - self.initial_capital) / self.initial_capital) * 100,
            'num_open_positions': len(self.positions),
            'num_closed_trades': len(self.closed_trades)
        }

        if self.closed_trades:
            wins = sum(1 for t in self.closed_trades if t['realized_pnl'] > 0)
            summary['win_rate'] = wins / len(self.closed_trades)
            summary['avg_pnl_per_trade'] = sum(t['realized_pnl'] for t in self.closed_trades) / len(self.closed_trades)

        return summary
