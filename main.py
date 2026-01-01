#!/usr/bin/env python3
"""Stock Correlation Trading System - API Server"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.1",
        port=8080,
        reload=False,
        log_level="error"
    )
