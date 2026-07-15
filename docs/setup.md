# Setup Guide

## Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU support)
- 8GB+ RAM
- 50GB+ disk space

## Installation

### Option 1: pip

```bash
pip install -r requirements.txt
```

### Option 2: conda

```bash
conda env create -f environment.yml
conda activate pixelsentinel
```

### Option 3: Docker

```bash
docker-compose up -d
```

## Running the Application

### Backend

```bash
uvicorn backend.app.main:app --reload
```

### Frontend

```bash
streamlit run frontend/streamlit_app.py
```