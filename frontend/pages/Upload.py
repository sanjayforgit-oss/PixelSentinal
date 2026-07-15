"""Upload page."""

import streamlit as st

st.title("Upload Images")
st.write("Upload satellite imagery for processing")

uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True)