"""FastAPI wrapper around rag_core. Serves the UI + /api/ask endpoint."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

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
        # Ensure query is not empty
        if not query or not query.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "Query cannot be empty", "answer": "Error: Query cannot be empty", "sources": [], "files": []}
            )
        
        # Save uploaded files
        if files:
            for f in files:
                try:
                    dest = Path(tmp_dir) / f.filename
                    with dest.open("wb") as out:
                        shutil.copyfileobj(f.file, out)
                    file_paths.append(str(dest))
                except Exception as e:
                    print(f"Error saving file {f.filename}: {e}")

        # Run the pipeline
        result = run_pipeline(query, file_paths)
        
        # Ensure result has all required fields
        if not isinstance(result, dict):
            result = {"answer": str(result), "sources": [], "files": []}
        
        result.setdefault("answer", "No answer generated")
        result.setdefault("sources", [])
        result.setdefault("files", file_paths)
        
        return result
    except Exception as e:
        print(f"API error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "answer": f"Error: {str(e)}", "sources": [], "files": []}
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
