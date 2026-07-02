FROM python:3.11-slim

WORKDIR /app

# System deps needed by faiss / trafilatura / pymupdf at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch build: the default PyPI wheel bundles CUDA libs that are
# dead weight (and extra memory) on a CPU-only host like Render.
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding + reranker models into the image so the first
# real request isn't slow (and doesn't need network access at runtime).
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY . .

# Render injects $PORT at runtime; default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

# Shell form so $PORT is substituted at container start, not build time.
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
