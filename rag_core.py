"""Shared RAG pipeline: web search + file ingestion + hybrid retrieval + Gemini generation."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from ddgs import DDGS
import trafilatura
import fitz  # PyMuPDF
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from google import genai

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEN_MODEL = "gemini-2.5-flash"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RETRIEVE = 15
TOP_K_RERANK = 5
NUM_SEARCH_RESULTS = 6

# Models are expensive to load; load once and reuse across requests.
_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def web_search(query: str, max_results: int = NUM_SEARCH_RESULTS) -> list[dict]:
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def scrape(url: str) -> str | None:
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        return trafilatura.extract(downloaded)
    except Exception:
        return None


def parse_file(path: str) -> str | None:
    ext = Path(path).suffix.lower()
    try:
        if ext == ".pdf":
            doc = fitz.open(path)
            return "\n\n".join(page.get_text("text") for page in doc)
        elif ext in (".txt", ".md"):
            return Path(path).read_text(encoding="utf-8", errors="replace")
        return None
    except Exception:
        return None


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return [c for c in chunks if len(c.strip()) > 50]


def build_index(chunks: list[str], embedder: SentenceTransformer):
    embeddings = embedder.encode(chunks, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings, dtype="float32"))
    bm25 = BM25Okapi([c.split() for c in chunks])
    return index, bm25


def hybrid_retrieve(query, chunks, index, bm25, embedder, top_k=TOP_K_RETRIEVE):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    _, dense_ids = index.search(np.array(q_emb, dtype="float32"), min(top_k, len(chunks)))

    bm25_scores = bm25.get_scores(query.split())
    sparse_ids = np.argsort(bm25_scores)[::-1][:top_k]

    seen, merged = set(), []
    for i in list(dense_ids[0]) + list(sparse_ids):
        if i not in seen and i < len(chunks):
            seen.add(i)
            merged.append(chunks[i])
    return merged


def rerank(query: str, candidates: list[str], reranker: CrossEncoder, top_k=TOP_K_RERANK) -> list[str]:
    pairs = [(query, c) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = [c for _, c in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]
    return ranked[:top_k]


def generate_answer(query: str, context_chunks: list[str], client: genai.Client) -> str:
    context = "\n\n".join(context_chunks)[:8000]
    prompt = (
        f"Answer the question using only the context below. "
        f"If the context doesn't contain the answer, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    response = client.models.generate_content(model=GEN_MODEL, contents=prompt)
    return response.text


def run_pipeline(query: str, file_paths: list[str] | None = None) -> dict:
    """End-to-end: search + files -> retrieve -> rerank -> generate. Returns dict for API/CLI use."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    client = genai.Client(api_key=api_key)

    results = web_search(query)
    urls = [r["href"] for r in results if "href" in r]

    all_chunks: list[str] = []
    for url in urls:
        text = scrape(url)
        if text:
            all_chunks.extend(chunk_text(text))

    filenames = [Path(p).name for p in (file_paths or [])]
    for path in (file_paths or []):
        text = parse_file(path)
        if text:
            all_chunks.extend(chunk_text(text))

    if not all_chunks:
        return {"answer": "No content could be retrieved for this query.", "sources": [], "files": filenames}

    embedder = get_embedder()
    index, bm25 = build_index(all_chunks, embedder)

    candidates = hybrid_retrieve(query, all_chunks, index, bm25, embedder)
    reranker = get_reranker()
    top_chunks = rerank(query, candidates, reranker)

    answer = generate_answer(query, top_chunks, client)

    return {"answer": answer, "sources": urls, "files": filenames}
