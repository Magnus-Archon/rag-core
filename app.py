"""FastAPI wrapper around rag_core. Serves the UI + /api/ask endpoint."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rag_core import run_pipeline

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Local RAG")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/ask")
async def ask(query: str = Form(...), files: list[UploadFile] = File(default=[])):
    tmp_dir = tempfile.mkdtemp()
    file_paths = []
    try:
        for f in files:
            dest = Path(tmp_dir) / f.filename
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            file_paths.append(str(dest))

        result = run_pipeline(query, file_paths)
        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
