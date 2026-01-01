#!/usr/bin/env python3
"""
Unified CLI Runner for Stock Correlation Trading System
Main entry point for all operations.

Usage:
    python -m src.run preprocess [--config CONFIG] [--force-full]
    python -m src.run backtest [--config CONFIG]
    python -m src.run gridsearch [--config CONFIG]
    python -m src.run live-signals [--config CONFIG] [--date DATE]
    python -m src.run paper-trade [--config CONFIG]
    python -m src.run multi-strategy [--config CONFIG]
    python -m src.run daily [--config CONFIG] [--date DATE] [--skip-signals] [--skip-monitor] [--skip-close]
    python -m src.run api [--host HOST] [--port PORT] [--reload]
"""

import argparse
import sys
from pathlib import Path

from src.core.logger import get_logger

logger = get_logger(__name__)


def run_preprocess(args):
    """Run data preprocessing pipeline."""
    from src.cli.preprocess import main as preprocess_main

    sys.argv = ["preprocess"]
    if args.config:
        sys.argv.extend(["--config", args.config])
    if args.force_full:
        sys.argv.append("--force-full")

    preprocess_main()


def run_backtest(args):
    """Run historical backtest."""
    from src.cli.backtest import main as backtest_main

    sys.argv = ["backtest"]
    if args.config:
        sys.argv.extend(["--config", args.config])

    backtest_main()


def run_gridsearch(args):
    """Run parameter grid search."""
    from src.cli.gridsearch import main as gridsearch_main

    sys.argv = ["gridsearch"]
    if args.config:
        sys.argv.extend(["--config", args.config])

    gridsearch_main()


def run_live_signals(args):
    """Generate live trading signals."""
    from src.cli.live_daily import main as live_daily_main

    sys.argv = ["live_daily"]
    if args.config:
        sys.argv.extend(["--config", args.config])
    if args.date:
        sys.argv.extend(["--date", args.date])

    live_daily_main()


def run_paper_trade(args):
    """Run paper trading simulation."""
    from src.cli.paper_trade_daily import main as paper_trade_main

    sys.argv = ["paper_trade_daily"]
    if args.config:
        sys.argv.extend(["--config", args.config])

    paper_trade_main()


def run_multi_strategy(args):
    """Run multi-strategy paper trading."""
    from src.cli.run_multi_strategy_paper_trading import main as multi_strategy_main

    sys.argv = ["run_multi_strategy_paper_trading"]
    if args.config:
        sys.argv.extend(["--config", args.config])

    multi_strategy_main()


def run_daily(args):
    """Run daily trading workflow (signals + monitoring + closing)."""
    from src.cli.daily_runner import main as daily_runner_main

    sys.argv = ["daily_runner"]
    if args.config:
        sys.argv.extend(["--config", args.config])
    if args.date:
        sys.argv.extend(["--date", args.date])
    if args.skip_signals:
        sys.argv.append("--skip-signals")
    if args.skip_monitor:
        sys.argv.append("--skip-monitor")
    if args.skip_close:
        sys.argv.append("--skip-close")

    daily_runner_main()


def run_api(args):
    """Start FastAPI server."""
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Unified CLI Runner for Stock Correlation Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Run data preprocessing")
    preprocess_parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    preprocess_parser.add_argument("--force-full", action="store_true", help="Force full data refresh")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run historical backtest")
    backtest_parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")

    # Grid search command
    gridsearch_parser = subparsers.add_parser("gridsearch", help="Run parameter grid search")
    gridsearch_parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")

    # Live signals command
    live_parser = subparsers.add_parser("live-signals", help="Generate live trading signals")
    live_parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    live_parser.add_argument("--date", type=str, help="Date for signal generation (YYYY-MM-DD)")

    # Paper trade command
    paper_parser = subparsers.add_parser("paper-trade", help="Run paper trading")
    paper_parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")

    # Multi-strategy command
    multi_parser = subparsers.add_parser("multi-strategy", help="Run multi-strategy paper trading")
    multi_parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")

    # Daily workflow command
    daily_parser = subparsers.add_parser("daily", help="Run daily trading workflow")
    daily_parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    daily_parser.add_argument("--date", type=str, help="Date for workflow (YYYY-MM-DD)")
    daily_parser.add_argument("--skip-signals", action="store_true", help="Skip signal generation")
    daily_parser.add_argument("--skip-monitor", action="store_true", help="Skip position monitoring")
    daily_parser.add_argument("--skip-close", action="store_true", help="Skip position closing")

    # API server command
    api_parser = subparsers.add_parser("api", help="Start FastAPI server")
    api_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate function
    commands = {
        "preprocess": run_preprocess,
        "backtest": run_backtest,
        "gridsearch": run_gridsearch,
        "live-signals": run_live_signals,
        "paper-trade": run_paper_trade,
        "multi-strategy": run_multi_strategy,
        "daily": run_daily,
        "api": run_api,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error running {args.command}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
