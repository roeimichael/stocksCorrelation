# Paper Trading Guide

## 🎯 Overview

This guide will help you set up and run paper trading (simulated live trading) to test the sp500-analogs strategy with real market data but without risking real money.

---

## 📋 What You Get

- **Automated daily workflow** - one command to run each day
- **Position tracking** - tracks all open and closed positions
- **Performance logging** - daily P&L snapshots
- **Excel exports** - monthly reports for review
- **Portfolio persistence** - state saved between runs
- **Real market prices** - fetches current prices via yfinance

---

## 🚀 Quick Start

### 1. Initial Setup (One Time)

```bash
# Make sure you're in the project directory
cd /path/to/stocksCorrelation

# Activate virtual environment
source .venv/bin/activate  # Unix/Mac
# or
.venv\Scripts\activate  # Windows

# Configure paper trading in configs/default.yaml
# (Already configured with defaults - see Configuration section below)
```

### 2. Daily Workflow (Run Every Day After Market Close)

```bash
# Step 1: Update data with latest prices (if needed)
python scripts/preprocess.py

# Step 2: Run paper trading
python scripts/paper_trade_daily.py
```

That's it! The script will:
- ✅ Load your portfolio state
- ✅ Fetch current prices
- ✅ Update open positions
- ✅ Close positions that hit exit criteria
- ✅ Generate new signals
- ✅ Open new positions (if slots available)
- ✅ Record daily P&L
- ✅ Save everything

### 3. Generate Monthly Report

```bash
# Export to Excel for review
python scripts/export_to_excel.py
```

This creates `results/paper_trading/paper_trading_report_YYYYMMDD_HHMMSS.xlsx` with:
- Daily P&L
- All closed trades
- Monthly summary
- Trade statistics
- Performance by signal type
- Current open positions

---

## ⚙️ Configuration

Edit `configs/default.yaml`:

```yaml
paper_trading:
  enabled: true               # Enable/disable paper trading
  initial_capital: 100000.0   # Starting capital ($100,000)
  position_size: 10000.0      # $ per position ($10,000)
  max_hold_days: 5            # Close positions after 5 days
  state_file: "data/paper_trading/portfolio_state.json"
```

### Key Parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `initial_capital` | Starting capital in USD | $100,000 |
| `position_size` | Notional value per position | $10,000 |
| `max_hold_days` | Auto-close positions after N days | 5 days |
| `max_positions` | Max concurrent positions (from backtest config) | 10 |

---

## 📊 Understanding the Output

### Daily Console Output

```
========================================================
PAPER TRADING - 2025-01-15
========================================================
Updating open positions...
Closing AAPL: held for 5 days
Closed UP position: AAPL @ $185.50, P&L: $375.00

Opening new positions: 2 slots available
Opened UP position: MSFT @ $420.30 x 23 shares = $9,666.90
Opened DOWN position: TSLA @ $245.60 x 40 shares = $9,824.00

Daily snapshot: Portfolio=$102,345.00, P&L=$2,345.00, Positions=8

========================================================
PORTFOLIO SUMMARY
========================================================
initial_capital: $100,000.00
current_value: $102,345.00
cash: $42,123.45
total_pnl: $2,345.00
total_return_pct: 2.35%
num_open_positions: 8
num_closed_trades: 15
win_rate: 0.60
avg_pnl_per_trade: $156.33
```

### Files Created

```
results/paper_trading/
├── closed_trades.csv          # All completed trades
├── daily_pnl.csv              # Daily portfolio snapshots
├── open_positions_2025-01-15.csv  # Current positions snapshot
└── paper_trading_report_20250115_180000.xlsx  # Excel report

data/paper_trading/
└── portfolio_state.json       # Portfolio state (positions, cash, history)
```

---

## 📈 Excel Report Sheets

### 1. Daily P&L
- Date
- Portfolio value
- Cash
- Positions value
- Total P&L
- Total return %
- Number of positions
- Daily P&L change

### 2. Closed Trades
- Symbol
- Signal (UP/DOWN)
- Entry/Exit dates and prices
- Shares
- Realized P&L
- Return %
- Hold days

### 3. Monthly Summary
- Month
- Final portfolio value
- Final total P&L
- Final return %
- Average positions
- Monthly P&L change

### 4. Trade Statistics
- Total trades
- Win/loss counts
- Win rate
- Average P&L per trade
- Average win/loss
- Best/worst trades
- Profit factor
- Average hold days

### 5. Performance by Signal
- Performance breakdown by UP vs DOWN signals

### 6. Current Positions
- All open positions with unrealized P&L

---

## 🔄 Daily Routine

### Best Practice Schedule

**Every weekday after market close (4:00 PM ET):**

1. **Update data** (5-10 minutes)
   ```bash
   python scripts/preprocess.py
   ```

2. **Run paper trading** (2-3 minutes)
   ```bash
   python scripts/paper_trade_daily.py
   ```

3. **Review output** (2 minutes)
   - Check console output for trades
   - Review P&L
   - Note any warnings

**Monthly (1st of each month):**

1. **Generate Excel report** (1 minute)
   ```bash
   python scripts/export_to_excel.py
   ```

2. **Review performance** (10-15 minutes)
   - Open the Excel file
   - Review monthly summary
   - Check win rate and average P&L
   - Analyze which signals perform better
   - Look for patterns in losses

3. **Make decisions**
   - Continue if strategy is profitable
   - Adjust parameters if needed
   - Stop if consistent losses

---

## 🎛️ Advanced Features

### Modifying Position Sizing

To adjust how much you trade per position:

```yaml
paper_trading:
  position_size: 5000.0  # Trade $5,000 per position instead of $10,000
```

### Changing Hold Period

To hold positions longer or shorter:

```yaml
paper_trading:
  max_hold_days: 3  # Close after 3 days instead of 5
```

### Adjusting Number of Positions

To trade more or fewer concurrent positions:

```yaml
backtest:
  max_positions: 5  # Hold max 5 positions instead of 10
```

---

## 🔍 Monitoring Performance

### Key Metrics to Track

**Weekly:**
- Total return %
- Win rate
- Average P&L per trade

**Monthly:**
- Cumulative P&L trend
- Profit factor (> 1.0 is good)
- Max drawdown
- Sharpe ratio (if > 0.5, strategy is working)

### Red Flags to Watch For

⚠️ **Stop trading if:**
- Win rate < 40% for 2+ months
- Profit factor < 0.8 for 2+ months
- Total return negative for 2+ months
- Large consecutive losses (> 10 in a row)

✅ **Good signs:**
- Win rate > 55%
- Profit factor > 1.2
- Consistent positive monthly returns
- Average win > 1.5x average loss

---

## 🛠️ Troubleshooting

### "No price data for XYZ"
**Cause:** yfinance couldn't fetch the stock price

**Solution:** Stock may be delisted or have trading issues. The script will skip it automatically.

### "Insufficient cash for XYZ"
**Cause:** Not enough cash to open a new position

**Solution:** This is normal when all capital is deployed. Position will be skipped.

### "No signals generated"
**Cause:** Market conditions don't match historical patterns

**Solution:** This is normal. Not every day will have tradeable signals.

### Portfolio state seems wrong
**Solution:** Check `data/paper_trading/portfolio_state.json`. You can delete it to start fresh.

---

## 🎓 Tips for Success

1. **Be consistent** - Run every trading day without skipping
2. **Keep records** - Save Excel reports each month
3. **Give it time** - Need at least 1 month to evaluate, preferably 3
4. **Don't overtrade** - Respect position limits
5. **Review regularly** - Monthly Excel reviews are crucial
6. **Trust the process** - Don't manually override signals
7. **Track patterns** - Note market conditions when strategy works best

---

## 📞 Support

If you encounter issues:

1. Check the logs in console output
2. Review `results/paper_trading/` files
3. Verify data files exist in `data/processed/`
4. Ensure config is correct in `configs/default.yaml`

---

## 🎯 Next Steps

After 1 month of paper trading:

1. **If profitable (> 5% return, > 55% win rate):**
   - Continue for 2 more months
   - Consider increasing position size
   - Start planning for live trading

2. **If breakeven (0-5% return, 45-55% win rate):**
   - Continue for 2 more months
   - Review and optimize parameters
   - Analyze losing trades for patterns

3. **If unprofitable (< 0% return, < 45% win rate):**
   - Review strategy assumptions
   - Try different parameter combinations
   - Consider market conditions

**Remember:** Paper trading is risk-free learning. Use this time to build confidence and understand the strategy before considering real money.

Good luck! 🚀
