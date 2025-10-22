# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**sp500-analogs** is an analog pattern matching system for predicting next-day stock movements. Given the last X trading days of a target stock, it finds the Y most similar historical windows across the S&P 500 universe, votes on what happened next in those analogs, and emits a signal when at least Z of the analogs agree.

Key concepts:
- **X**: Window length (days) - number of consecutive trading days in each pattern
- **Y**: Top K analogs - number of most similar historical windows to use for voting
- **Z**: Vote threshold - minimum fraction of analogs that must agree to emit a signal

## Development Environment

- **Python Version**: 3.12.6
- **Virtual Environment**: `.venv` directory (activate with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Unix)
- **IDE**: PyCharm (configuration in `.idea/`)
- **Dependencies**: yfinance, pandas, numpy, scipy, pyyaml, matplotlib, seaborn, tqdm

## Setup Commands

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Unix/Mac

# Install dependencies
pip install -r requirements.txt

# Run complete workflow
python scripts/preprocess.py      # Download data and build windows
python scripts/gridsearch.py      # Optional: optimize X/Y/Z parameters
python scripts/backtest.py        # Run walk-forward backtest
python scripts/live.py            # Generate today's signals
```

## Project Structure

```
sp500-analogs/
├── configs/default.yaml          # All configuration parameters (X/Y/Z, dates, costs)
├── data/
│   ├── raw/                      # Downloaded OHLCV (parquet)
│   └── processed/                # Cleaned returns, windows.pkl
├── results/
│   ├── experiments/              # Grid search, correlation analysis
│   ├── backtests/                # Equity curve, trades, metrics
│   └── live/                     # Daily signals CSV
├── scripts/
│   ├── preprocess.py             # Data pipeline
│   ├── gridsearch.py             # Parameter optimization
│   ├── backtest.py               # Walk-forward backtest
│   └── live.py                   # Signal generation
└── src/
    ├── core/                     # Config loader, logger
    ├── dataio/                   # Fetch (yfinance), prep (returns, cleaning)
    ├── modeling/                 # Windows, similarity, voting
    ├── evals/                    # Metrics, correlation analysis
    └── trading/                  # Backtest engine with costs/slippage
```

## Architecture

### Data Pipeline (dataio)
- `fetch.py`: Downloads S&P 500 tickers and adjusted OHLCV from Yahoo Finance
- `prep.py`: Cleans prices, computes returns, aligns calendars

### Modeling (modeling)
- `windows.py`: Builds rolling windows of length X, normalizes (z-score/rank/vol), labels next-day direction
- `similarity.py`: Computes similarity (Pearson/Spearman/cosine) between windows, finds top Y analogs
- `vote.py`: Majority or similarity-weighted voting, emits UP/DOWN/ABSTAIN signal based on threshold Z

### Evaluation (evals)
- `metrics.py`: Hit rate, precision/recall, Sharpe, max drawdown, profit factor
- `correlation_matrix.py`: EDA - correlation heatmaps and distribution

### Trading (trading)
- `engine.py`: Walk-forward backtester with costs/slippage, position sizing, daily open→close trades

## Key Parameters (configs/default.yaml)

- `data.start/end`: Date range for historical data
- `data.top_n`: Number of S&P 500 stocks to use (default: 50)
- `windows.length`: X - window length (default: 10)
- `windows.normalization`: zscore|rank|vol
- `similarity.metric`: pearson|spearman|cosine
- `similarity.top_k`: Y - number of analogs (default: 25)
- `vote.threshold`: Z - vote threshold 0-1 (default: 0.70)
- `vote.scheme`: majority|similarity_weighted
- `backtest.costs_bps`: Transaction costs in basis points
- `backtest.max_positions`: Max concurrent positions

## Algorithm Flow

1. **Preprocess**: Download prices → compute returns → build windows → save windows.pkl
2. **Grid Search** (optional): Test combinations of X/Y/Z to find optimal parameters
3. **Backtest**: Walk-forward simulation with strict no-look-ahead bias
4. **Live**: Generate signals for today's trading

Each window is normalized within itself to capture pattern shape (not scale). Similarity is computed using the configured metric. Voting aggregates analog outcomes with optional similarity weighting.

## Important Design Patterns

- **No Look-Ahead Bias**: All analog searches filter to windows with `end_date < target_date`
- **Window Normalization**: Applied per-window (z-score/rank/vol) to focus on pattern shape
- **Walk-Forward Testing**: Backtest only uses data available up to each simulation date
- **Data Classes**: Window, Signal, Trade classes in modeling/trading modules
- **Configuration-Driven**: Single YAML controls all parameters

## Common Tasks

### Adding a New Similarity Metric
Edit `src/modeling/similarity.py::compute_similarity()` to add new metric, then update config validation in `src/core/config.py`

### Changing Universe
Edit `configs/default.yaml::data.universe` - use "sp500" or path to CSV with ticker column

### Adjusting Costs
Edit `configs/default.yaml::backtest.costs_bps` and `slippage_bps`

### Testing New Parameters
Run `python scripts/gridsearch.py` with modified param_grid in the script

## Data Files

- `data/raw/adj_close_prices.parquet`: Adjusted close prices (dates × tickers)
- `data/processed/returns.parquet`: Daily returns
- `data/processed/windows.pkl`: Pickled list of Window objects
- `results/backtests/trades.csv`: Trade log
- `results/backtests/equity_curve.csv`: Equity over time
- `results/live/signals_YYYYMMDD.csv`: Daily signals

## Notes for Development

- Windows are stored as numpy arrays (features) with metadata (symbol, dates, label)
- All dates use pandas Timestamp for consistency
- Logger is centralized via `src/core/logger.py::get_logger()`
- Config validation happens at load time in `src/core/config.py`
- All financial data is adjusted for splits/dividends via yfinance
