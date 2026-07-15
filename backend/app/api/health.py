"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/health", tags=["health"])


@router.get("/status")
def health_status():
    """Check API health status."""
    return {"status": "healthy", "version": "1.0.0"}