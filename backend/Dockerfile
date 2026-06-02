# Multi-stage Dockerfile for the Tumor Sense FastAPI backend.
#
# Use this when you'd rather build once locally (or in CI) and push to a
# registry, instead of letting Render rebuild the heavy torch/langchain stack
# on every deploy. See backend/DEPLOY.md → "Option B" for the full recipe.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_VERBOSITY=error \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TUMORSENSE_MODELS_DIR=/app/backend/models

# System deps needed by faiss-cpu, Pillow, and some torch ops.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libstdc++6 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first to keep the layer cache useful between code edits.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r backend/requirements.txt

# Copy the backend package (server + svm + image_models + rag + docs + scripts).
COPY backend/ ./backend/

# Pre-download embedding model so cold start doesn't pay the 90 MB hit on the
# first /api/explain call. Comment out to keep the image small.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8000

# Render expects the service to bind to $PORT; default to 8000 for local docker.
CMD ["sh", "-c", "python backend/scripts/download_weights.py && uvicorn backend.server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
