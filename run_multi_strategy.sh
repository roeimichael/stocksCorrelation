#!/bin/bash
# Multi-Strategy Paper Trading Runner
# Run this to test 5 different parameter configurations simultaneously

set -e  # Exit on error

echo "=============================================="
echo "SP500-ANALOGS MULTI-STRATEGY PAPER TRADING"
echo "$(date)"
echo "=============================================="
echo ""
echo "Testing 5 different strategies in parallel:"
echo "  1. Conservative (high threshold)"
echo "  2. Moderate (balanced default)"
echo "  3. Aggressive (lower threshold)"
echo "  4. Spearman Rank (rank-based)"
echo "  5. Short Window (5-day patterns)"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run: python -m venv .venv"
    exit 1
fi

# Activate virtual environment
echo "[1/2] Activating virtual environment..."
source .venv/bin/activate

# Run multi-strategy paper trading
echo ""
echo "[2/2] Running multi-strategy paper trading..."
python scripts/run_multi_strategy_paper_trading.py

echo ""
echo "=============================================="
echo "MULTI-STRATEGY PAPER TRADING COMPLETE!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - results/paper_trading/conservative/"
echo "  - results/paper_trading/moderate/"
echo "  - results/paper_trading/aggressive/"
echo "  - results/paper_trading/spearman_rank/"
echo "  - results/paper_trading/short_window/"
echo "  - results/paper_trading/strategy_comparison.csv"
echo ""
echo "Next steps:"
echo "  - Review strategy_comparison.csv to see which performed best"
echo "  - After 1 month, compare total returns and win rates"
echo "  - Continue with top performing strategies"
echo ""
