# veriface/api/main.py
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# -----------------------------------------------------------------------------
# External pipeline import (with good error if missing)
# -----------------------------------------------------------------------------
try:
    from adk_project.agents.coordinator.agent import run_all
except Exception as e:
    # Raise a clear import-time error so you see it in logs immediately
    raise RuntimeError(
        "Failed to import run_all from adk_project.agents.coordinator.agent. "
        "Verify your PYTHONPATH and that the package is installed. "
        f"Original error: {e}"
    )

# -----------------------------------------------------------------------------
# App init & CORS
# -----------------------------------------------------------------------------
APP_NAME = "VeriFace DIA API"
APP_VERSION = "0.1.0"

app = FastAPI(title=APP_NAME, version=APP_VERSION)

# Permissive for dev; tighten allow_origins to your UI URL for prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # e.g., ["https://your-ui-xyz.run.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("veriface.api")

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
SAFE_EXTS = {".mp4", ".mov", ".mkv", ".webm"}

def _sanitize_filename(name: str) -> str:
    # Drop any path components and keep a simple filename
    return Path(name).name.replace("..", "_").replace("/", "_").replace("\\", "_")

def _to_jsonable(obj: Any) -> Any:
    """
    Try to ensure the result is JSON serializable.
    """
    try:
        json.dumps(obj)
        return obj
    except Exception:
        # Fallback: convert to dict/str where possible
        if hasattr(obj, "dict"):
            try:
                return obj.dict()
            except Exception:
                pass
        if hasattr(obj, "__dict__"):
            try:
                return dict(obj.__dict__)
            except Exception:
                pass
        try:
            return str(obj)
        except Exception:
            return {"_nonserializable": True}

def _error_response(detail: str, status_code: int = 400, extra: Dict[str, Any] | None = None):
    payload = {"ok": False, "error": detail}
    if extra:
        payload.update(extra)
    return JSONResponse(status_code=status_code, content=payload)

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True, "status": "healthy"}

@app.get("/version")
def version():
    return {"ok": True, "name": APP_NAME, "version": APP_VERSION}

@app.get("/ping")
def ping():
    return {"ok": True, "pong": True}

@app.post("/analyze")
async def analyze(request: Request, file: UploadFile = File(...)):
    t0 = time.time()
    client = request.client.host if request.client else "unknown"
    fname = _sanitize_filename(file.filename or "upload.bin")
    ext = Path(fname).suffix.lower()

    log.info(f"[/analyze] from={client} filename={fname} size=? ext={ext}")

    if ext not in SAFE_EXTS:
        return _error_response(
            f"Unsupported file type '{ext}'. Allowed: {sorted(SAFE_EXTS)}",
            status_code=400,
        )

    # Allow FPS override via env var (defaults to 25 if not set or invalid)
    try:
        fps = int(os.getenv("VERIFACE_FPS", "25"))
    except ValueError:
        fps = 25

    # Save to temp and run pipeline
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = os.path.join(td, fname)
            # Stream copy to avoid loading in memory
            with open(tmp_path, "wb") as out:
                shutil.copyfileobj(file.file, out)

            log.info(f"[/analyze] saved to {tmp_path}, starting run_all(fps={fps})")
            try:
                result = run_all(tmp_path, fps=fps)  # <-- your pipeline
            except Exception as e:
                log.exception("run_all failed")
                return _error_response("analysis_failed", status_code=500, extra={"detail": str(e)})

    except Exception as e:
        log.exception("Failed to store temp file / pre-process")
        return _error_response("upload_failed", status_code=500, extra={"detail": str(e)})

    duration = round(time.time() - t0, 3)
    log.info(f"[/analyze] done in {duration}s")

    # Make sure it’s JSON serializable
    payload = _to_jsonable(result)
    return JSONResponse(
        content={
            "ok": True,
            "duration_s": duration,
            "filename": fname,
            "result": payload,
        }
    )
