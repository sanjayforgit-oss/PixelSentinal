"""File upload endpoints."""

from fastapi import APIRouter, File, UploadFile
from typing import List

router = APIRouter(prefix="/api/v1/upload", tags=["upload"])


@router.post("/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload a satellite image."""
    return {"filename": file.filename}


@router.post("/batch")
async def upload_batch(files: List[UploadFile] = File(...)):
    """Upload multiple images."""
    return {"files": [f.filename for f in files]}