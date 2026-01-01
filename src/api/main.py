"""
FastAPI Backend for Stock Correlation Trading System

Provides RESTful API and WebSocket support for:
- Viewing positions and investments
- Correlation analysis
- Trade history
- Real-time monitoring
- Running daily workflows
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.logger import get_logger
from src.api.routers import (
    positions,
    correlations,
    trades,
    signals,
    monitoring,
    operations,
)

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Stock Correlation Trading System API",
    description="RESTful API for pattern matching trading system with correlation analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(positions.router, prefix="/api/positions", tags=["Positions"])
app.include_router(correlations.router, prefix="/api/correlations", tags=["Correlations"])
app.include_router(trades.router, prefix="/api/trades", tags=["Trades"])
app.include_router(signals.router, prefix="/api/signals", tags=["Signals"])
app.include_router(monitoring.router, prefix="/api/monitoring", tags=["Monitoring"])
app.include_router(operations.router, prefix="/api/operations", tags=["Operations"])


@app.get("/")
async def root():
    """Root endpoint - API status."""
    return {
        "status": "running",
        "service": "Stock Correlation Trading System API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "trading-api"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8084, log_level="error")
