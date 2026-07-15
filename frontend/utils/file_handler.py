"""File handling utilities."""

import os
from pathlib import Path


def save_uploaded_file(uploaded_file, directory: str = "uploads") -> str:
    """Save uploaded file.
    
    Args:
        uploaded_file: Uploaded file object
        directory: Directory to save to
        
    Returns:
        Path to saved file
    """
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path