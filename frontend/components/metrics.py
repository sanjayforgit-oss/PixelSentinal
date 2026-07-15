"""Metrics display component."""

import streamlit as st


def display_metrics(metrics: dict):
    """Display evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
    """
    cols = st.columns(len(metrics))
    for col, (key, value) in zip(cols, metrics.items()):
        with col:
            st.metric(key, f"{value:.4f}")