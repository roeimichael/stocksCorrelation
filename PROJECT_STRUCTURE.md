# Project Structure

## Overview

All code is now unified under the `src/` directory with clear separation of concerns.

## Directory Structure

```
stocksCorrelation/
├── run.py                      # Convenience wrapper (calls src.run.main())
├── config.yaml                 # Main configuration file
├── requirements.txt            # Python dependencies
│
├── src/                        # Main source code directory
│   ├── __init__.py
│   ├── run.py                  # Main CLI entry point
│   │
│   ├── core/                   # Core infrastructure
│   │   ├── config.py           # Configuration loading & validation
│   │   ├── constants.py        # Centralized constants & paths
│   │   ├── data_loader.py      # Data loading utilities
│   │   └── logger.py           # Logging setup (loguru)
│   │
│   ├── dataio/                 # Data input/output
│   │   ├── fetch.py            # Yahoo Finance data fetching
│   │   ├── prep.py             # Returns preparation
│   │   └── live_append.py      # Live data ingestion
│   │
│   ├── modeling/               # ML/pattern matching logic
│   │   ├── similarity.py       # Analog ranking (Pearson/Spearman/Cosine)
│   │   ├── vote.py             # Voting aggregation
│   │   └── windows.py          # Window construction & normalization
│   │
│   ├── evals/                  # Performance evaluation
│   │   ├── metrics.py          # Sharpe, drawdown, etc.
│   │   └── correlation_matrix.py  # Correlation analysis
│   │
│   ├── trading/                # Trading systems
│   │   ├── engine.py           # Backtest engine
│   │   ├── paper_trading.py    # PaperTradingPortfolio class
│   │   ├── monitor.py          # Position monitoring
│   │   └── hedging.py          # Risk management & hedging
│   │
│   ├── cli/                    # Command-line interface scripts
│   │   ├── __init__.py
│   │   ├── preprocess.py       # Data download & processing
│   │   ├── backtest.py         # Historical backtesting
│   │   ├── gridsearch.py       # Parameter optimization
│   │   ├── live_daily.py       # Live signal generation
│   │   ├── paper_trade_daily.py   # Paper trading
│   │   ├── run_multi_strategy_paper_trading.py  # Multi-strategy testing
│   │   ├── daily_runner.py     # Main daily workflow
│   │   └── export_to_excel.py  # Excel report generation
│   │
│   └── api/                    # FastAPI backend
│       ├── __init__.py
│       ├── main.py             # FastAPI application
│       │
│       ├── models/             # Pydantic schemas
│       │   ├── positions.py
│       │   ├── correlations.py
│       │   ├── trades.py
│       │   ├── signals.py
│       │   └── monitoring.py
│       │
│       ├── routers/            # API endpoints
│       │   ├── positions.py
│       │   ├── correlations.py
│       │   ├── trades.py
│       │   ├── signals.py
│       │   ├── monitoring.py   # WebSocket support
│       │   └── operations.py   # Background operations
│       │
│       └── services/           # Business logic
│           ├── positions_service.py
│           ├── correlations_service.py
│           ├── trades_service.py
│           ├── signals_service.py
│           └── monitoring_service.py
│
├── data/                       # Data storage
│   ├── raw/                    # Downloaded price data
│   ├── processed/              # Returns & windows
│   ├── live_ingest/            # Daily data input
│   └── paper_trading/          # Paper trading state
│
├── results/                    # Output and results
│   ├── backtests/
│   ├── experiments/
│   ├── live/
│   ├── paper_trading/
│   └── plots/
│
├── configs/                    # Configuration templates
│   └── default.yaml
│
└── docs/                       # Documentation
    ├── AUTOMATION_GUIDE.md
    └── PAPER_TRADING_GUIDE.md
```

## Usage

### Running Commands

All commands are accessed through the unified CLI:

```bash
# Using the convenience wrapper
python run.py [command] [options]

# Or directly
python -m src.run [command] [options]
```

### Available Commands

#### Data Preprocessing
```bash
python run.py preprocess [--config CONFIG] [--force-full]
```

#### Backtesting
```bash
python run.py backtest [--config CONFIG]
```

#### Parameter Optimization
```bash
python run.py gridsearch [--config CONFIG]
```

#### Live Signal Generation
```bash
python run.py live-signals [--config CONFIG] [--date YYYY-MM-DD]
```

#### Paper Trading
```bash
python run.py paper-trade [--config CONFIG]
```

#### Multi-Strategy Testing
```bash
python run.py multi-strategy [--config CONFIG]
```

#### Daily Workflow (Main Production)
```bash
python run.py daily [--config CONFIG] [--date YYYY-MM-DD] [--skip-signals] [--skip-monitor] [--skip-close]
```

#### FastAPI Server
```bash
python run.py api [--host HOST] [--port PORT] [--reload]
```

## Import Structure

All imports should follow these patterns:

### From Core Modules
```python
from src.core.logger import get_logger
from src.core.constants import Paths
from src.core.config import load_config
```

### From Data I/O
```python
from src.dataio.fetch import fetch_universe
from src.dataio.prep import prepare_returns
```

### From Modeling
```python
from src.modeling.similarity import rank_analogs
from src.modeling.vote import vote
from src.modeling.windows import build_windows
```

### From Trading
```python
from src.trading.engine import run_backtest
from src.trading.paper_trading import PaperTradingPortfolio
from src.trading.monitor import classify_alert
```

### From API
```python
from src.api.main import app
from src.api.services.positions_service import PositionsService
```

### From CLI (when needed)
```python
from src.cli.daily_runner import main as daily_runner
```

## Key Improvements

1. **Unified Structure**: Everything under `src/` for consistency
2. **Clear Separation**: API, CLI, core logic all in dedicated folders
3. **Single Entry Point**: `run.py` for all operations
4. **No Duplication**: Removed redundant batch/shell scripts
5. **Clean Imports**: All imports use `src.` prefix
6. **Modular Design**: Each component in its logical location

## Migration Guide

### Old vs New

| Old Location | New Location |
|-------------|-------------|
| `scripts/preprocess.py` | `src/cli/preprocess.py` |
| `scripts/backtest.py` | `src/cli/backtest.py` |
| `daily_runner.py` | `src/cli/daily_runner.py` |
| `api/main.py` | `src/api/main.py` |
| `run_paper_trading.bat/sh` | `python run.py paper-trade` |
| `run_multi_strategy.bat/sh` | `python run.py multi-strategy` |

### Import Changes

**Old:**
```python
from scripts.preprocess import main
from api.services.positions_service import PositionsService
```

**New:**
```python
from src.cli.preprocess import main
from src.api.services.positions_service import PositionsService
```

## API Server

The FastAPI server is now located at `src/api/main.py` and can be started with:

```bash
python run.py api
```

Access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development

### Adding New CLI Commands

1. Create new file in `src/cli/`
2. Add import to `src/cli/__init__.py`
3. Add command handler in `src/run.py`

### Adding New API Endpoints

1. Create new router in `src/api/routers/`
2. Create corresponding service in `src/api/services/`
3. Create Pydantic models in `src/api/models/`
4. Register router in `src/api/main.py`

### Adding New Core Functionality

Add to appropriate module:
- Data operations → `src/dataio/`
- Pattern matching → `src/modeling/`
- Trading logic → `src/trading/`
- Evaluation → `src/evals/`
- Infrastructure → `src/core/`
