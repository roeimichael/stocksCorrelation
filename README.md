# sp500-analogs: Analog Pattern Matching for Next-Day Direction

Predict next-day stock movements by finding similar historical price patterns and voting on their outcomes.

## Overview

Given the last **X** trading days of a target stock, this system:
1. Finds the **Y** most similar historical windows across the stock universe
2. "Votes" on what happened the next day in those analogs
3. Emits a next-day up/down signal when at least **Z** of the analogs agree

## Quick Start

### 1. Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Unix/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Preprocess Data

Download stock data, compute returns, and build windows:

```bash
python scripts/preprocess.py
```

This will:
- Fetch S&P 500 constituents (configurable in `configs/default.yaml`)
- Download adjusted OHLCV data
- Clean and align data
- Compute returns
- Build rolling windows
- Save processed data to `data/processed/`

### 3. Run Backtest

Test the strategy with walk-forward validation:

```bash
python scripts/backtest.py
```

Results saved to `results/backtests/`:
- `trades.csv` - All executed trades
- `equity_curve.csv` - Equity over time
- `metrics.csv` - Performance metrics
- Plots: equity curve and drawdown

### 4. Grid Search (Optional)

Find optimal X, Y, Z parameters:

```bash
python scripts/gridsearch.py
```

Results saved to `results/experiments/gridsearch_results.csv`

### 5. Generate Live Signals

Generate signals for today's trading:

```bash
python scripts/live.py
```

Signals saved to `results/live/signals_YYYYMMDD.csv`

## Configuration

Edit `configs/default.yaml` to customize:

- **Data**: Date range, universe (S&P 500 top N), data source
- **Windows**: Length (X), normalization method
- **Similarity**: Metric (Pearson/Spearman/cosine), top K (Y)
- **Vote**: Threshold (Z), vote scheme (majority/similarity-weighted)
- **Backtest**: Entry/exit, costs, slippage, position sizing

### Key Parameters

- **X (window_length)**: Number of consecutive trading days in each pattern (default: 10)
- **Y (top_k)**: Number of most similar historical windows to use for voting (default: 25)
- **Z (threshold)**: Minimum fraction of analogs that must agree (default: 0.70 = 70%)

## Project Structure

```
sp500-analogs/
├── configs/
│   └── default.yaml          # Configuration parameters
├── data/
│   ├── raw/                  # Downloaded OHLCV data
│   └── processed/            # Cleaned returns and windows
├── results/
│   ├── experiments/          # Grid search results and EDA
│   ├── backtests/            # Backtest results and plots
│   └── live/                 # Daily signals
├── scripts/
│   ├── preprocess.py         # Data pipeline
│   ├── gridsearch.py         # Parameter optimization
│   ├── backtest.py           # Walk-forward backtest
│   └── live.py               # Signal generation
└── src/
    ├── core/                 # Config and logging
    ├── dataio/               # Data fetching and preprocessing
    ├── modeling/             # Windows, similarity, voting
    ├── evals/                # Metrics and analysis
    └── trading/              # Backtest engine
```

## Algorithm

### 1. Data Preprocessing

- Fetch adjusted daily prices for S&P 500 universe
- Compute daily returns: r_t = (P_t / P_{t-1}) - 1
- Align trading calendars across all symbols

### 2. Window Construction

- For each symbol, create rolling windows of length X
- Normalize each window (z-score, rank, or volatility scaling)
- Label each window with next-day direction (up/down)

### 3. Analog Search

For a target window:
- Compute similarity to all historical windows (dated ≤ target end date)
- Select top Y windows by similarity
- Filter by minimum similarity threshold (optional)

### 4. Voting

- **Majority vote**: Simple average of analog next-day directions
- **Similarity-weighted vote**: Weight each analog by its similarity score

If vote share ≥ Z → emit signal (UP or DOWN)
Otherwise → ABSTAIN

### 5. Backtesting

- Walk-forward: at each day t, use only data ≤ t
- Enter at open, exit at close (same day)
- Apply transaction costs and slippage
- Position sizing: fixed percentage of equity per trade

## Metrics

The system tracks:
- **Hit rate**: Accuracy of directional predictions
- **Precision/Recall**: For both up and down predictions
- **Sharpe ratio**: Risk-adjusted returns
- **Max drawdown**: Largest peak-to-trough decline
- **Profit factor**: Ratio of gross profits to gross losses
- **Win rate**: Percentage of profitable trades

## Data Sources

- **S&P 500 constituents**: Wikipedia (top N by market cap)
- **Price data**: Yahoo Finance via yfinance
- **Frequency**: Daily adjusted close prices

## Limitations

- Uses close prices as proxy for open (in this simple version)
- Does not account for corporate actions beyond adjustments
- Transaction costs are estimated (not real broker fees)
- Look-ahead bias prevented through strict date filtering
- Market impact and liquidity constraints not modeled

## Next Steps

1. **Optimize parameters**: Run grid search to find best X/Y/Z for your data
2. **Analyze results**: Review correlation matrices and backtest metrics
3. **Refine strategy**: Experiment with different similarity metrics and normalization
4. **Live testing**: Generate signals and paper trade before real deployment

## References

This implements analog pattern matching for financial time series, inspired by:
- K-Nearest Neighbors for time series classification
- Technical analysis pattern recognition
- Ensemble voting for prediction aggregation

## License

This is a research and educational project. Use at your own risk.
