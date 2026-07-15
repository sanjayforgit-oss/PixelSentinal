"""File uploader component."""

import streamlit as st


def render_uploader():
    """Render file uploader."""
    uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True)
    return uploaded_files