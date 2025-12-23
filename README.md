# S&P 500 Analog Pattern Matching System

**A quantitative trading system that predicts next-day stock movements using analog pattern matching across the S&P 500 universe.**

---

## 📊 What Is This?

This is a **pattern-based trading system** that finds historical "analogs" (similar price patterns) to predict future stock movements.

### The Core Idea

1. **Look at recent price action** - Take the last X trading days of a stock (e.g., 10 days)
2. **Find similar historical patterns** - Search the entire S&P 500 history for the Y most similar windows (e.g., top 25)
3. **Vote on what happened next** - If at least Z% of those analogs moved UP next day, emit an UP signal
4. **Trade the signal** - Enter positions based on high-confidence predictions

### Example

*"AAPL's last 10 days look similar to 25 historical patterns. 18 of those went UP next day (72% vote). Since 72% > 70% threshold → BUY AAPL."*

---

## 🎯 Core Capabilities

### 1. **Historical Backtesting**
Test the strategy on historical data with strict no-look-ahead bias, transaction costs, and slippage modeling.

### 2. **Parameter Optimization**
Grid search to find optimal values for:
- **X** (Window length): How many days of history to match
- **Y** (Top K analogs): How many similar patterns to consider
- **Z** (Vote threshold): What percentage agreement is needed to trade

### 3. **Live Signal Generation**
Daily production pipeline that:
- Ingests new market data
- Generates trading signals
- Monitors open positions for drift/degradation
- Closes positions based on exit criteria

### 4. **Position Monitoring**
Tracks pattern "drift" in open positions using:
- **Similarity Retention**: Are the analogs still similar?
- **Directional Concordance**: Are the analogs still going the same direction?
- **Correlation Decay**: Is the pattern breaking down?
- **Pattern Deviation**: Is P&L deviating from expected?

### 5. **Risk Management**
- Transaction cost modeling (slippage + commissions)
- Maximum position limits
- Alert system (GREEN/YELLOW/RED) for position health
- Automatic position closing on adverse signals

---

## 🏗️ Project Structure

```
stocksCorrelation/
│
├── daily_runner.py          # Main daily trading system (signals → monitor → close)
├── config.yaml              # All system parameters (X/Y/Z, costs, alerts)
│
├── scripts/                 # One-time/occasional scripts
│   ├── preprocess.py        # Initial data setup (fetch prices, build windows)
│   ├── backtest.py          # Historical strategy testing
│   └── gridsearch.py        # Parameter optimization
│
├── src/                     # Core library modules
│   ├── core/                # Config loader, logger
│   ├── dataio/              # Data fetching and preparation
│   ├── modeling/            # Windows, similarity, voting logic
│   ├── evals/               # Performance metrics
│   └── trading/             # Backtest engine, monitoring, hedging
│
├── configs/
│   └── default.yaml         # Default configuration template
│
├── data/
│   ├── raw/                 # Downloaded price data (parquet)
│   ├── processed/           # Returns, windows bank (parquet)
│   └── live_ingest/         # New daily data drops here
│
└── results/
    ├── backtests/           # Historical test results
    ├── experiments/         # Grid search results
    └── live/                # Daily signals, alerts, ledger
        ├── signals_YYYY-MM-DD.csv
        ├── alerts_YYYY-MM-DD.csv
        ├── positions_state.json
        └── portfolio_ledger.csv
```

---

## ⚙️ Configuration

All system behavior is controlled by `config.yaml`:

### Key Parameters

```yaml
# Pattern Matching (X/Y/Z)
windows:
  length: 10                    # X: Pattern window length (days)
  normalization: "zscore"       # How to normalize patterns

similarity:
  metric: "pearson"             # pearson, spearman, cosine, euclidean
  top_k: 25                     # Y: Number of analogs to use
  min_sim: 0.20                 # Minimum similarity threshold

vote:
  scheme: "similarity_weighted" # majority or similarity_weighted
  threshold: 0.70               # Z: Vote threshold (70% = 0.70)
  abstain_if_below_k: 10        # Don't trade if <10 analogs found

# Trading
backtest:
  costs_bps: 5                  # Transaction costs (0.05%)
  slippage_bps: 3               # Slippage (0.03%)
  max_positions: 10             # Maximum concurrent positions
  position_pct: 0.10            # 10% of equity per position

# Position Monitoring
monitor:
  sr_floor: 0.20                # Similarity retention floor
  dc_floor: 0.55                # Directional concordance floor
  cd_drop_alert: -0.30          # Alert if correlation drops 30%
  pds_z_alert: 1.5              # Pattern deviation threshold

# Exit Policy
close_policy:
  close_on_red: true            # Close on RED alert
  close_on_reverse: true        # Close on reverse signal
  close_on_abstain: false       # Close on ABSTAIN signal
```

---

## 🚀 Getting Started

### 1. **Setup**

```bash
# Activate virtual environment
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. **Initial Data Download** (One-time)

```bash
python scripts/preprocess.py --config config.yaml
```

This will:
- Download S&P 500 ticker list
- Fetch historical price data from Yahoo Finance
- Compute returns
- Build the windows bank (database of all historical patterns)

**Time**: ~10-20 minutes depending on network and number of stocks

### 3. **Parameter Optimization** (Optional)

```bash
python scripts/gridsearch.py --config config.yaml
```

Tests all combinations of X/Y/Z to find the best parameters. Results saved to `results/experiments/gridsearch_TIMESTAMP.csv`.

**Time**: ~30-60 minutes (depends on parameter grid size)

### 4. **Historical Backtest**

```bash
python scripts/backtest.py --config config.yaml
```

Runs walk-forward backtest with your current parameters. Outputs:
- `results/backtests/trades.csv` - All trades
- `results/backtests/equity_curve.csv` - Equity over time
- Summary metrics (Sharpe, max drawdown, hit rate, etc.)

**Time**: ~5-10 minutes

### 5. **Daily Production Run**

```bash
python daily_runner.py --config config.yaml
```

This is your **main daily workflow**. It runs three steps:
1. **Signal Generation** - Find trading opportunities
2. **Position Monitoring** - Check health of open positions
3. **Position Closing** - Exit positions that meet criteria

**Time**: ~2-5 minutes

---

## 📅 Daily Workflow - Full Walkthrough

### Morning: Before Market Open (9:00 AM)

**1. Place new data in live_ingest/**
```bash
# Manually download or automated script drops:
# data/live_ingest/AAPL_2024-01-15.csv
# data/live_ingest/MSFT_2024-01-15.csv
```

**2. Run daily system**
```bash
python daily_runner.py
```

**What happens:**

#### Step 1: Signal Generation
- Reads new data from `data/live_ingest/`
- Appends to `data/raw/` price files
- Rebuilds returns and windows for updated symbols
- For each stock, finds top 25 most similar historical patterns
- Votes on whether those patterns went UP or DOWN next day
- Emits UP/DOWN/ABSTAIN signal based on vote threshold
- Saves signals to `results/live/signals_2024-01-15.csv`

**Output Example:**
```
Generated 47 signals:
  UP signals: 12
  DOWN signals: 8
  ABSTAIN: 27
  Avg confidence: 23.5%
```

#### Step 2: Position Monitoring
- Loads open positions from `results/live/positions_state.json`
- For each open position, computes:
  - **Similarity Retention** (SR): Are the analogs still similar today?
  - **Directional Concordance** (DC): Are they still going the same way?
  - **Correlation Decay** (ΔCorr): Has the pattern correlation dropped?
  - **Pattern Deviation Z** (PDZ): Is P&L behaving abnormally?
- Classifies alert level: GREEN (healthy) / YELLOW (caution) / RED (danger)
- Saves alerts to `results/live/alerts_2024-01-15.csv`

**Output Example:**
```
Monitoring 12 open positions
[1/12] AAPL (entered 2024-01-10, side=UP)
  SR=0.78, DC=0.85, Corr=0.72, Δ=-0.05, PDZ=0.3 → GREEN

[2/12] TSLA (entered 2024-01-12, side=UP)
  SR=0.42, DC=0.48, Corr=0.35, Δ=-0.45, PDZ=2.1 → RED

Alert Summary:
  GREEN: 9
  YELLOW: 2
  RED: 1
```

#### Step 3: Position Closing
- Loads latest signals and alert levels
- For each open position, checks exit criteria:
  - **RED alert** → Close position
  - **Reverse signal** (was UP, now DOWN) → Close position
  - **ABSTAIN signal** (optional, configurable)
- Computes P&L for closed positions
- Appends trades to `results/live/portfolio_ledger.csv`
- Updates `positions_state.json` with remaining positions

**Output Example:**
```
Checking 12 open positions for closing

Closing TSLA: RED_ALERT
  Entry: 2024-01-12 @ $100.00
  Exit:  2024-01-15 @ $97.50
  Gross PnL: -$250.00
  Net PnL:   -$264.00

Closing GE: REVERSE_SIGNAL
  Entry: 2024-01-13 @ $100.00
  Exit:  2024-01-15 @ $101.20
  Gross PnL: $120.00
  Net PnL:   $106.00

Closed 2 positions, 10 remaining open
```

---

### During Market Hours: Review & Execute

**Review generated signals:**
```bash
cat results/live/signals_2024-01-15.csv
```

**Review alerts:**
```bash
cat results/live/alerts_2024-01-15.csv
```

**Review closed trades:**
```bash
tail -20 results/live/portfolio_ledger.csv
```

**Execute trades** (manually or via broker API):
- Enter new positions from high-confidence signals
- Close positions flagged for exit
- Adjust positions based on alert levels

---

### Evening: Post-Market Review

**Check portfolio state:**
```bash
cat results/live/positions_state.json
```

**Analyze ledger performance:**
```python
import pandas as pd
ledger = pd.read_parquet('results/live/portfolio_ledger.csv')
print(f"Total trades: {len(ledger)}")
print(f"Win rate: {(ledger['net_pnl'] > 0).mean():.2%}")
print(f"Total P&L: ${ledger['net_pnl'].sum():,.2f}")
```

---

## 📊 Understanding the Output Files

### `signals_YYYY-MM-DD.csv`
Daily trading signals with columns:
- `symbol`: Stock ticker
- `signal`: UP, DOWN, or ABSTAIN
- `p_up`: Probability of UP move (vote fraction)
- `confidence`: Confidence score (distance from 0.5)
- `analogs`: JSON list of similar historical patterns

### `alerts_YYYY-MM-DD.csv`
Position health alerts with columns:
- `symbol`, `entry_date`, `side`
- `similarity_retention`, `directional_concordance`, `correlation_today`, `correlation_delta_3d`, `pattern_deviation_z`
- `alert_level`: GREEN, YELLOW, RED, or ERROR

### `positions_state.json`
Current open positions with full context:
```json
{
  "open_positions": [
    {
      "symbol": "AAPL",
      "entry_date": "2024-01-10",
      "side": "UP",
      "p_up": 0.75,
      "confidence": 0.25,
      "notional": 10000.0,
      "analogs": [...],
      "last_monitor_date": "2024-01-15",
      "metrics": {...},
      "alert_level": "GREEN"
    }
  ]
}
```

### `portfolio_ledger.csv`
Full trade history with columns:
- `symbol`, `entry_date`, `exit_date`, `side`
- `entry_price`, `exit_price`, `notional`
- `gross_pnl`, `net_pnl`, `reason`
- `p_up`, `confidence`

---

## 🔧 Command Reference

### Daily Production
```bash
# Run full daily pipeline
python daily_runner.py

# Run with custom config
python daily_runner.py --config my_config.yaml

# Run for specific date
python daily_runner.py --date 2024-01-15

# Skip monitoring step
python daily_runner.py --skip-monitor

# Skip closing step
python daily_runner.py --skip-close

# Only generate signals
python daily_runner.py --skip-monitor --skip-close
```

### One-Time Setup
```bash
# Initial data download
python scripts/preprocess.py

# Force full refresh (ignore existing data)
python scripts/preprocess.py --force-full
```

### Research & Optimization
```bash
# Run backtest
python scripts/backtest.py

# Parameter grid search
python scripts/gridsearch.py
```

---

## 🧠 How It Works - Technical Details

### Pattern Normalization

Before comparing patterns, each window is normalized to focus on **shape** rather than scale:

- **Z-score**: `(x - mean) / std` - Standard normalization
- **Rank**: Convert to percentile ranks (0-1)
- **Vol**: Divide by volatility
- **None**: Raw returns

**Why?** A 5% daily move in TSLA has different meaning than in JNJ. Normalization makes patterns comparable across stocks and time periods.

### Similarity Metrics

How we measure "similarity" between two patterns:

- **Pearson**: Linear correlation (-1 to +1)
- **Spearman**: Rank correlation (handles non-linear)
- **Cosine**: Angle between vectors (0 to 1)
- **Euclidean**: Distance in N-dimensional space (lower = more similar)

**Default**: Pearson correlation works well for price patterns.

### Voting Schemes

How we aggregate analog outcomes:

1. **Majority Vote**: Simple count of UP vs DOWN
   - If 18/25 analogs went UP → p_up = 0.72

2. **Similarity-Weighted Vote**: Weight by similarity score
   - More similar analogs get more weight
   - Formula: `p_up = Σ(sim_i * label_i) / Σ(sim_i)`

**Default**: Similarity-weighted tends to perform better.

### Walk-Forward Backtest

Strict no-look-ahead testing:
1. For each day in test period:
   - Only use data available up to that day
   - Find analogs from historical data (cutoff_date < signal_date)
   - Generate signal
   - Track position P&L
2. Roll forward day-by-day
3. Account for costs, slippage, position limits

### Position Monitoring Metrics

**Similarity Retention (SR)**
- Compare today's pattern to entry pattern
- Measure how similar the analogs still are
- Low SR → Pattern has changed significantly

**Directional Concordance (DC)**
- Are the analogs still moving in the predicted direction?
- DC = fraction of analogs that went the right way so far

**Correlation Decay (ΔCorr)**
- Track correlation between target stock and analog basket over time
- Large drop → Pattern breaking down

**Pattern Deviation Z-score (PDZ)**
- Compare actual P&L to expected P&L from analogs
- High PDZ → Behaving abnormally (good or bad)

---

## 📈 Performance Metrics

The system reports standard quant metrics:

- **Total Return**: Portfolio $ gain/loss
- **Sharpe Ratio**: Risk-adjusted return (higher is better, >1.0 is good)
- **Max Drawdown**: Worst peak-to-trough decline
- **Hit Rate**: % of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Win/Loss Ratio**: Average win size / Average loss size

---

## 🛡️ Risk Considerations

### What This System Does Well
✅ Finds repeating price patterns
✅ Aggregates evidence from many historical examples
✅ Monitors pattern degradation in real-time
✅ Handles transaction costs realistically

### What This System Doesn't Do
❌ Predict black swan events
❌ Account for fundamental changes (earnings, news)
❌ Guarantee profitability (patterns can break down)
❌ Replace human judgment (use as decision support)

### Best Practices
1. **Start small** - Test with paper trading or small capital
2. **Monitor alerts** - Don't ignore YELLOW/RED warnings
3. **Diversify** - Don't put all capital in one signal
4. **Reoptimize** - Run grid search quarterly as markets evolve
5. **Know when to stop** - If metrics degrade persistently, halt trading

---

## 🔬 Research & Development

### Extending the System

**Add new similarity metrics** (src/modeling/similarity.py):
```python
def compute_similarity(vec1, vec2, metric='pearson'):
    if metric == 'my_custom_metric':
        return my_custom_similarity(vec1, vec2)
```

**Customize exit logic** (daily_runner.py):
```python
def should_close_position(position, signals_df, cfg):
    # Add custom exit rules
    if my_custom_condition(position):
        return True, 'CUSTOM_EXIT'
```

**Modify voting** (src/modeling/vote.py):
```python
def vote(analogs_df, scheme='majority', ...):
    if scheme == 'my_custom_vote':
        return custom_voting_logic(analogs_df)
```

### Future Enhancements

- Sector-neutral hedging
- Options strategies based on volatility analogs
- Multi-timeframe patterns (daily + weekly)
- Machine learning meta-model on top of analog signals
- Real-time broker integration

---

## 📚 Additional Resources

**Configuration**: See `configs/default.yaml` for all tunable parameters
**Code Documentation**: Each module has detailed docstrings
**Logs**: Check console output for detailed execution logs

---

## ⚠️ Disclaimer

This is a research/educational project. Past performance does not guarantee future results. Use at your own risk. Always validate with paper trading before deploying real capital.

---

**Built with**: Python 3.12+ | pandas | numpy | scipy | yfinance | PyYAML

**License**: MIT (or your chosen license)

---

*For questions, issues, or contributions, please open a GitHub issue.*
