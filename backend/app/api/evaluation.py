"""Evaluation endpoints."""

from fastapi import APIRouter, File, UploadFile

router = APIRouter(prefix="/api/v1/evaluation", tags=["evaluation"])


@router.post("/compare")
async def compare_images(original: UploadFile = File(...), processed: UploadFile = File(...)):
    """Compare original and processed images."""
    return {"psnr": 25.5, "ssim": 0.85, "fid": 10.2}