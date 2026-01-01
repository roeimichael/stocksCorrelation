# Stock Correlation Trading System

API-first pattern-matching trading system with correlation analysis.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

API available at: **http://localhost:8000**

Documentation: **http://localhost:8000/docs**

## API Endpoints

All operations are controlled through the API. No CLI commands.

### View Data

**Positions**
- `GET /api/positions` - All open positions
- `GET /api/positions/{symbol}` - Position details
- `GET /api/positions/summary/stats` - Portfolio summary

**Correlations**
- `GET /api/correlations/matrix?symbols=AAPL,MSFT` - Correlation matrix
- `GET /api/correlations/portfolio` - Portfolio analysis
- `GET /api/correlations/pairs/{symbol1}/{symbol2}` - Pair correlation

**Trades**
- `GET /api/trades` - Trade history
- `GET /api/trades/recent?n=20` - Recent trades
- `GET /api/trades/performance/by-symbol` - Performance by symbol

**Signals**
- `GET /api/signals/latest` - Latest signals
- `GET /api/signals/date/{date}` - Signals by date
- `GET /api/signals/history/{symbol}` - Signal history

**Monitoring**
- `GET /api/monitoring/alerts/latest` - Latest alerts
- `WS /api/monitoring/ws` - WebSocket real-time updates

### Run Operations

**All operations run in background and return a task_id for tracking.**

**Data & Setup**
- `POST /api/operations/preprocess` - Download and prepare data

**Trading Operations**
- `POST /api/operations/daily` - Run daily workflow
- `POST /api/operations/backtest` - Run backtest
- `POST /api/operations/gridsearch` - Optimize parameters

**Task Management**
- `GET /api/operations/status/{task_id}` - Check task status
- `GET /api/operations/tasks` - List all tasks

## Example Usage

### Start a Backtest
```bash
curl -X POST http://localhost:8000/api/operations/backtest \
  -H "Content-Type: application/json" \
  -d '{"config": "config.yaml"}'

# Response: {"status": "started", "task_id": "backtest_20250101_120000"}
```

### Check Task Status
```bash
curl http://localhost:8000/api/operations/status/backtest_20250101_120000

# Response: {"status": "completed", "started_at": "...", "completed_at": "..."}
```

### View Positions
```bash
curl http://localhost:8000/api/positions?sort_by=pnl

# Returns all positions sorted by P&L
```

### WebSocket Real-time Updates
```javascript
const ws = new WebSocket('ws://localhost:8000/api/monitoring/ws');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Update:', data);
};
```

## Configuration

Edit `config.yaml` to customize:
- Window length and normalization
- Similarity metrics (Pearson, Spearman, Cosine)
- Vote thresholds
- Position limits and risk management

## Structure

```
src/
├── api/          # FastAPI endpoints
├── cli/          # Internal CLI modules
├── core/         # Core infrastructure
├── dataio/       # Data operations
├── modeling/     # Pattern matching
├── evals/        # Metrics
└── trading/      # Trading logic
```

## Development

```bash
# Run with auto-reload
python -c "import uvicorn; uvicorn.run('src.api.main:app', reload=True)"
```

## License

MIT
