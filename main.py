#!/usr/bin/env python3
"""Stock Correlation Trading System - API Server"""

if __name__ == "__main__":
    import uvicorn
    print("Starting API server...")
    print("API: http://localhost:8084")
    print("Docs: http://localhost:8084/docs")
    print("Press Ctrl+C to stop")
    print("-" * 40)
    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.1",
        port=8084,
        reload=False,
        log_level="error"
    )
