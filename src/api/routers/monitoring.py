"""API endpoints for position monitoring and real-time updates."""

from fastapi import APIRouter, Query, HTTPException, Path, WebSocket, WebSocketDisconnect
from typing import Optional, List
import asyncio
import json

from src.api.services.monitoring_service import MonitoringService
from src.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
monitoring_service = MonitoringService()


# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept and store new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


@router.get("/alerts/latest")
async def get_latest_alerts():
    """
    Get the most recent position alerts.

    Returns latest alert generation with all alerts and summary statistics.
    """
    try:
        alerts_data = monitoring_service.get_latest_alerts()

        return {
            "status": "success",
            "data": alerts_data,
        }
    except Exception as e:
        logger.error(f"Error fetching latest alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/date/{date}")
async def get_alerts_by_date(
    date: str = Path(..., description="Date in YYYY-MM-DD format"),
):
    """
    Get alerts for a specific date.

    Returns all alerts generated on the specified date.
    """
    try:
        alerts_data = monitoring_service.get_alerts_by_date(date=date)

        return {
            "status": "success",
            "data": alerts_data,
        }
    except Exception as e:
        logger.error(f"Error fetching alerts for date {date}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/history/{symbol}")
async def get_alert_history(
    symbol: str = Path(..., description="Stock ticker symbol"),
    days: int = Query(30, description="Number of days to look back", ge=1, le=365),
):
    """
    Get alert history for a specific symbol.

    Returns chronological alert history for the symbol over the specified period.
    """
    try:
        history = monitoring_service.get_alert_history(symbol=symbol, days=days)

        return {
            "status": "success",
            "symbol": symbol.upper(),
            "lookback_days": days,
            "count": len(history),
            "data": history,
        }
    except Exception as e:
        logger.error(f"Error fetching alert history for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time monitoring updates.

    Sends real-time updates for:
    - New alerts generated
    - Position P&L changes
    - Signal updates
    - System status

    Usage:
        ws://localhost:8000/api/monitoring/ws
    """
    await manager.connect(websocket)

    try:
        # Send initial connection message
        await websocket.send_json(
            {
                "type": "connection",
                "status": "connected",
                "message": "Real-time monitoring WebSocket connected",
            }
        )

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Receive message from client (for ping/pong or commands)
                data = await websocket.receive_text()

                # Handle ping
                if data == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": str(asyncio.get_event_loop().time())})

                # Handle status request
                elif data == "status":
                    alerts_data = monitoring_service.get_latest_alerts()
                    await websocket.send_json(
                        {
                            "type": "status",
                            "data": alerts_data,
                        }
                    )

                # Handle alerts request
                elif data == "alerts":
                    alerts_data = monitoring_service.get_latest_alerts()
                    await websocket.send_json(
                        {
                            "type": "alerts",
                            "data": alerts_data,
                        }
                    )

            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket loop: {e}")
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": str(e),
                    }
                )

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)


async def broadcast_alert_update(alerts_data: dict):
    """
    Broadcast alert updates to all connected WebSocket clients.

    Called by monitoring service when new alerts are generated.
    """
    await manager.broadcast(
        {
            "type": "alert_update",
            "data": alerts_data,
        }
    )


async def broadcast_position_update(positions_data: dict):
    """
    Broadcast position updates to all connected WebSocket clients.

    Called when position P&L or status changes.
    """
    await manager.broadcast(
        {
            "type": "position_update",
            "data": positions_data,
        }
    )
