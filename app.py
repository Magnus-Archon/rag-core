"""FastAPI wrapper around rag_core. Serves the UI + /api/ask endpoint.

Note: rag_core is imported here, but it only imports light stdlib/typing
at module level -- all heavy ML libraries load lazily on first use. This
keeps app startup (and therefore port binding) fast, which Render requires.
"""
from __future__ import annotations

import shutil
import tempfile
import traceback
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from rag_core import run_pipeline

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Local RAG")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/ask")
async def ask(
    query: str = Form(...),
    scope: str = Form("auto"),
    files: list[UploadFile] = File(default=[]),
):
    tmp_dir = tempfile.mkdtemp()
    file_paths = []
    try:
        for f in files:
            if not f.filename:
                continue
            dest = Path(tmp_dir) / f.filename
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            file_paths.append(str(dest))

        result = run_pipeline(query, file_paths, scope=scope)
        return result
    except Exception as e:
        # Always return JSON on failure -- never let an unhandled exception
        # produce a bare/partial response that breaks the frontend's res.json().
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
