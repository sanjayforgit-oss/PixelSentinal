"""Image viewer component."""

import streamlit as st
from PIL import Image


def display_image(image_path: str, caption: str = None):
    """Display image.
    
    Args:
        image_path: Path to image
        caption: Image caption
    """
    image = Image.open(image_path)
    st.image(image, caption=caption)