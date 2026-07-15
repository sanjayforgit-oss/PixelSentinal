"""Sidebar component."""

import streamlit as st


def render_sidebar():
    """Render sidebar menu."""
    with st.sidebar:
        st.title("Menu")
        page = st.radio("Select Page", ["Home", "Upload", "Results", "Evaluation"])
    return page