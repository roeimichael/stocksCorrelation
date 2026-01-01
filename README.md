# Stock Correlation Trading System

Pattern-matching trading system with FastAPI backend and correlation analysis.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Start API Server
```bash
python run.py api
```

Access API at: http://localhost:8000/docs

## Structure

```
src/
├── run.py           # Main entry point
├── api/             # FastAPI backend
├── cli/             # Command-line tools
├── core/            # Core infrastructure
├── dataio/          # Data operations
├── modeling/        # Pattern matching
├── evals/           # Metrics
└── trading/         # Trading logic
```

## Commands

```bash
python run.py preprocess      # Download & prepare data
python run.py backtest        # Historical backtest
python run.py gridsearch      # Optimize parameters
python run.py live-signals    # Generate signals
python run.py paper-trade     # Paper trading
python run.py multi-strategy  # Multi-strategy test
python run.py daily           # Daily workflow
python run.py api             # Start API server
```

## API Endpoints

### Positions
- `GET /api/positions` - View all positions
- `GET /api/positions/{symbol}` - Get position details
- `GET /api/positions/summary/stats` - Portfolio summary

### Correlations
- `GET /api/correlations/matrix` - Correlation matrix
- `GET /api/correlations/portfolio` - Portfolio analysis
- `GET /api/correlations/pairs/{symbol1}/{symbol2}` - Pair correlation

### Trades
- `GET /api/trades` - Trade history
- `GET /api/trades/recent` - Recent trades
- `GET /api/trades/performance/by-symbol` - Performance stats

### Signals
- `GET /api/signals/latest` - Latest signals
- `GET /api/signals/date/{date}` - Signals by date
- `GET /api/signals/history/{symbol}` - Signal history

### Monitoring
- `GET /api/monitoring/alerts/latest` - Latest alerts
- `WS /api/monitoring/ws` - WebSocket updates

### Operations
- `POST /api/operations/daily` - Run daily workflow
- `POST /api/operations/backtest` - Run backtest
- `POST /api/operations/gridsearch` - Parameter optimization
- `GET /api/operations/status/{task_id}` - Task status

## Configuration

Edit `config.yaml` to customize:
- Data parameters (window length, lookback period)
- Similarity metrics (Pearson, Spearman, Cosine)
- Trading rules (thresholds, position limits)
- Risk management (hedging, monitoring)

## Development

### Run with auto-reload
```bash
python run.py api --reload
```

### Module import
```python
from src.api.main import app
from src.cli.preprocess import main
from src.core.config import load_config
```

## License

MIT
