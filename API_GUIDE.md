# FastAPI Backend Guide

## Overview

The Stock Correlation Trading System now includes a comprehensive FastAPI backend that provides:

- **RESTful API** for accessing positions, trades, signals, and correlations
- **WebSocket support** for real-time monitoring updates
- **Background task execution** for running daily workflows, backtests, and optimizations
- **Organized data views** with sorting, filtering, and correlation analysis

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
# Using the unified CLI runner
python run.py api

# Or with custom host/port
python run.py api --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the API

- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc (ReDoc)
- **API Root**: http://localhost:8000

## Unified CLI Runner

All batch/shell scripts have been replaced with a single Python entry point: `run.py`

### Available Commands

```bash
# Data preprocessing
python run.py preprocess [--config CONFIG] [--force-full]

# Historical backtest
python run.py backtest [--config CONFIG]

# Parameter optimization
python run.py gridsearch [--config CONFIG]

# Generate live signals
python run.py live-signals [--config CONFIG] [--date DATE]

# Paper trading
python run.py paper-trade [--config CONFIG]

# Multi-strategy testing
python run.py multi-strategy [--config CONFIG]

# Daily workflow (main production workflow)
python run.py daily [--config CONFIG] [--date DATE] [--skip-signals] [--skip-monitor] [--skip-close]

# Start API server
python run.py api [--host HOST] [--port PORT] [--reload]
```

## API Endpoints

### Positions

Get organized views of current investments with P&L, alerts, and metrics.

#### Get All Positions

```http
GET /api/positions
GET /api/positions?sort_by=pnl&filter_alert=GREEN
```

**Query Parameters:**
- `sort_by` (optional): Sort field - `pnl`, `pnl_pct`, `days_held`, `symbol`, `confidence`
- `filter_alert` (optional): Filter by alert level - `GREEN`, `YELLOW`, `RED`

**Response:**
```json
{
  "status": "success",
  "data": {
    "total_positions": 8,
    "total_notional": 80000.0,
    "total_pnl": 1250.50,
    "total_pnl_pct": 1.56,
    "alert_breakdown": {
      "GREEN": 5,
      "YELLOW": 2,
      "RED": 1
    },
    "positions": [
      {
        "symbol": "AAPL",
        "entry_date": "2025-12-15",
        "entry_price": 195.50,
        "current_price": 198.75,
        "shares": 50,
        "notional": 9775.0,
        "pnl": 162.50,
        "pnl_pct": 1.66,
        "signal": "UP",
        "confidence": 0.75,
        "alert_level": "GREEN",
        "days_held": 3
      }
    ]
  }
}
```

#### Get Position by Symbol

```http
GET /api/positions/{symbol}
```

**Example:**
```http
GET /api/positions/AAPL
```

#### Get Positions Summary

```http
GET /api/positions/summary/stats
```

Returns high-level metrics without individual position details.

---

### Correlations

Analyze correlations between stocks and portfolio diversification.

#### Get Correlation Matrix

```http
GET /api/correlations/matrix
GET /api/correlations/matrix?symbols=AAPL,MSFT,GOOGL&lookback_days=60
```

**Query Parameters:**
- `symbols` (optional): Comma-separated list of symbols. If omitted, uses portfolio positions
- `lookback_days` (optional): Number of days to look back (default: 60, range: 10-365)

**Response:**
```json
{
  "status": "success",
  "data": {
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "matrix": [
      [1.0, 0.85, 0.72],
      [0.85, 1.0, 0.78],
      [0.72, 0.78, 1.0]
    ],
    "lookback_days": 60,
    "top_positive": [
      {
        "symbol1": "AAPL",
        "symbol2": "MSFT",
        "correlation": 0.85,
        "lookback_days": 60
      }
    ],
    "top_negative": []
  }
}
```

#### Get Portfolio Correlation Analysis

```http
GET /api/correlations/portfolio
GET /api/correlations/portfolio?lookback_days=60
```

Returns correlation matrix, diversification score, concentration risk, and hedge suggestions for current portfolio.

**Response:**
```json
{
  "status": "success",
  "data": {
    "portfolio_symbols": ["AAPL", "MSFT", "GOOGL"],
    "correlation_matrix": { ... },
    "diversification_score": 0.65,
    "concentration_risk": 0.35,
    "suggested_hedges": [
      {
        "symbol": "TLT",
        "correlation": -0.35
      }
    ]
  }
}
```

#### Get Pair Correlation

```http
GET /api/correlations/pairs/{symbol1}/{symbol2}
```

**Example:**
```http
GET /api/correlations/pairs/AAPL/MSFT?lookback_days=60
```

---

### Trades

Access trade history with performance statistics.

#### Get Trade History

```http
GET /api/trades
GET /api/trades?start_date=2025-01-01&end_date=2025-12-31&symbol=AAPL&min_pnl=0
```

**Query Parameters:**
- `start_date` (optional): Filter trades after this date (YYYY-MM-DD)
- `end_date` (optional): Filter trades before this date (YYYY-MM-DD)
- `symbol` (optional): Filter by symbol
- `min_pnl` (optional): Minimum P&L filter
- `max_pnl` (optional): Maximum P&L filter

**Response:**
```json
{
  "status": "success",
  "data": {
    "total_trades": 100,
    "winning_trades": 65,
    "losing_trades": 35,
    "win_rate": 65.0,
    "total_pnl": 12500.0,
    "avg_win": 350.0,
    "avg_loss": -180.0,
    "profit_factor": 1.94,
    "trades": [...]
  }
}
```

#### Get Recent Trades

```http
GET /api/trades/recent
GET /api/trades/recent?n=20
```

Returns N most recent trades (default: 20, max: 100).

#### Get Performance by Symbol

```http
GET /api/trades/performance/by-symbol
```

Returns aggregate performance metrics grouped by symbol.

---

### Signals

View trading signals and their history.

#### Get Latest Signals

```http
GET /api/signals/latest
```

Returns the most recent signal generation.

**Response:**
```json
{
  "status": "success",
  "date": "2025-12-31",
  "total_signals": 50,
  "up_signals": 12,
  "down_signals": 8,
  "abstain_signals": 30,
  "signals": [
    {
      "symbol": "AAPL",
      "signal": "UP",
      "confidence": 0.75,
      "p_up": 0.75,
      "p_down": 0.25,
      "n_analogs": 25,
      "date": "2025-12-31"
    }
  ]
}
```

#### Get Signals by Date

```http
GET /api/signals/date/{date}
```

**Example:**
```http
GET /api/signals/date/2025-12-31
```

#### Get Signal History for Symbol

```http
GET /api/signals/history/{symbol}
GET /api/signals/history/AAPL?days=30
```

---

### Monitoring & Alerts

Real-time position monitoring with alerts.

#### Get Latest Alerts

```http
GET /api/monitoring/alerts/latest
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "date": "2025-12-31",
    "total_alerts": 10,
    "green": 6,
    "yellow": 3,
    "red": 1,
    "error": 0,
    "alerts": [
      {
        "symbol": "AAPL",
        "alert_level": "YELLOW",
        "date": "2025-12-31",
        "message": "Similarity retention below threshold",
        "recommended_action": "Consider reducing position by 50%",
        "metrics": {
          "similarity_retention": 0.18,
          "directional_concordance": 0.65
        }
      }
    ]
  }
}
```

#### Get Alerts by Date

```http
GET /api/monitoring/alerts/date/{date}
```

#### Get Alert History for Symbol

```http
GET /api/monitoring/alerts/history/{symbol}?days=30
```

#### WebSocket (Real-time Updates)

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/api/monitoring/ws');

ws.onopen = () => {
  console.log('Connected to real-time monitoring');

  // Request status
  ws.send('status');

  // Request alerts
  ws.send('alerts');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'connection') {
    console.log('Connection established');
  }

  if (data.type === 'alert_update') {
    console.log('New alerts:', data.data);
  }

  if (data.type === 'position_update') {
    console.log('Position updated:', data.data);
  }
};

// Keep-alive ping
setInterval(() => {
  ws.send('ping');
}, 30000);
```

---

### Operations

Run background operations (daily workflow, backtest, etc.).

#### Run Daily Workflow

```http
POST /api/operations/daily
Content-Type: application/json

{
  "date": "2025-12-31",
  "skip_signals": false,
  "skip_monitor": false,
  "skip_close": false,
  "config": "config.yaml"
}
```

**Response:**
```json
{
  "status": "started",
  "task_id": "daily_20251231_153045",
  "message": "Daily workflow started in background",
  "command": "python run.py daily --config config.yaml"
}
```

#### Run Backtest

```http
POST /api/operations/backtest
Content-Type: application/json

{
  "config": "config.yaml"
}
```

#### Run Grid Search

```http
POST /api/operations/gridsearch
Content-Type: application/json

{
  "config": "config.yaml"
}
```

**Note:** Grid search is a long-running operation (30+ minutes).

#### Run Preprocessing

```http
POST /api/operations/preprocess
Content-Type: application/json

{
  "config": "config.yaml",
  "force_full": false
}
```

#### Get Task Status

```http
GET /api/operations/status/{task_id}
```

**Example:**
```http
GET /api/operations/status/daily_20251231_153045
```

**Response:**
```json
{
  "task_id": "daily_20251231_153045",
  "status": "completed",
  "started_at": "2025-12-31T15:30:45",
  "completed_at": "2025-12-31T15:35:20",
  "command": "python run.py daily --config config.yaml",
  "returncode": 0,
  "stdout": "...",
  "stderr": ""
}
```

#### List All Tasks

```http
GET /api/operations/tasks
```

Returns summary of all background tasks.

---

## Frontend Integration

### CORS Configuration

The API is configured to accept requests from any origin (`allow_origins=["*"]`). For production, update this in `api/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Example Frontend Usage (React)

```javascript
// Fetch positions
const fetchPositions = async () => {
  const response = await fetch('http://localhost:8000/api/positions?sort_by=pnl');
  const data = await response.json();
  return data.data;
};

// Fetch correlation matrix
const fetchCorrelations = async (symbols) => {
  const symbolsQuery = symbols.join(',');
  const response = await fetch(
    `http://localhost:8000/api/correlations/matrix?symbols=${symbolsQuery}&lookback_days=60`
  );
  const data = await response.json();
  return data.data;
};

// Run daily workflow
const runDailyWorkflow = async () => {
  const response = await fetch('http://localhost:8000/api/operations/daily', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      skip_signals: false,
      skip_monitor: false,
      skip_close: false,
    }),
  });
  const data = await response.json();
  return data.task_id;
};

// Check task status
const checkTaskStatus = async (taskId) => {
  const response = await fetch(`http://localhost:8000/api/operations/status/${taskId}`);
  const data = await response.json();
  return data;
};
```

---

## Architecture

```
api/
├── main.py                    # FastAPI application setup
├── models/                    # Pydantic schemas
│   ├── positions.py
│   ├── correlations.py
│   ├── trades.py
│   ├── signals.py
│   └── monitoring.py
├── routers/                   # API endpoints
│   ├── positions.py           # Position management
│   ├── correlations.py        # Correlation analysis
│   ├── trades.py              # Trade history
│   ├── signals.py             # Trading signals
│   ├── monitoring.py          # Alerts + WebSocket
│   └── operations.py          # Background operations
└── services/                  # Business logic
    ├── positions_service.py
    ├── correlations_service.py
    ├── trades_service.py
    ├── signals_service.py
    └── monitoring_service.py
```

---

## Migration from Old Scripts

### Old Way (Batch/Shell Scripts)

```bash
# Windows
run_paper_trading.bat
run_multi_strategy.bat

# Linux/Mac
./run_paper_trading.sh
./run_multi_strategy.sh
```

### New Way (Unified CLI)

```bash
# CLI
python run.py paper-trade
python run.py multi-strategy

# Or via API
curl -X POST http://localhost:8000/api/operations/daily \
  -H "Content-Type: application/json" \
  -d '{"config": "config.yaml"}'
```

---

## Development

### Running in Development Mode

```bash
# With auto-reload
python run.py api --reload

# Or directly with uvicorn
uvicorn api.main:app --reload --port 8000
```

### Testing Endpoints

```bash
# Using curl
curl http://localhost:8000/api/positions

# Using httpie
http GET http://localhost:8000/api/positions

# Using Python requests
import requests
response = requests.get('http://localhost:8000/api/positions')
print(response.json())
```

---

## Production Deployment

### Using Gunicorn + Uvicorn Workers

```bash
pip install gunicorn

gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300
```

### Using Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

```bash
# .env file
API_HOST=0.0.0.0
API_PORT=8000
CONFIG_PATH=config.yaml
LOG_LEVEL=INFO
```

---

## Next Steps

1. **Build Your Frontend**: Use React, Vue, or any framework to consume the API
2. **Customize CORS**: Update allowed origins for production
3. **Add Authentication**: Implement JWT or API key authentication if needed
4. **Monitor Performance**: Use FastAPI's built-in metrics or add Prometheus
5. **Database Integration**: Optionally migrate from file storage to PostgreSQL/MongoDB

---

## Support

For issues or questions:
- Check API documentation: http://localhost:8000/docs
- Review this guide
- Check project README.md
