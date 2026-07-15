"""Inference endpoints."""

from fastapi import APIRouter, File, UploadFile

router = APIRouter(prefix="/api/v1/inference", tags=["inference"])


@router.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    """Enhance satellite image."""
    return {"status": "processing", "filename": file.filename}


@router.post("/colorize")
async def colorize(file: UploadFile = File(...)):
    """Colorize infrared image."""
    return {"status": "processing", "filename": file.filename}


@router.post("/segment")
async def segment(file: UploadFile = File(...)):
    """Segment image."""
    return {"status": "processing", "filename": file.filename}