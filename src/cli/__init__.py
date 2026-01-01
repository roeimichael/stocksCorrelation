"""CLI commands for the stock correlation trading system."""

from .preprocess import main as preprocess
from .backtest import main as backtest
from .gridsearch import main as gridsearch
from .live_daily import main as live_daily
from .paper_trade_daily import main as paper_trade
from .run_multi_strategy_paper_trading import main as multi_strategy
from .daily_runner import main as daily_runner

__all__ = [
    "preprocess",
    "backtest",
    "gridsearch",
    "live_daily",
    "paper_trade",
    "multi_strategy",
    "daily_runner",
]
