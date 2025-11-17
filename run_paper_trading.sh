#!/bin/bash
# Daily Paper Trading Runner
# Run this script once per day after market close

set -e  # Exit on error

echo "=============================================="
echo "SP500-ANALOGS PAPER TRADING"
echo "$(date)"
echo "=============================================="
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

# Run paper trading
echo ""
echo "[2/2] Running paper trading..."
python scripts/paper_trade_daily.py

echo ""
echo "=============================================="
echo "PAPER TRADING COMPLETE!"
echo "=============================================="
echo ""
echo "To view results:"
echo "  - Check console output above"
echo "  - View CSV files in results/paper_trading/"
echo "  - Generate Excel report: python scripts/export_to_excel.py"
echo ""
