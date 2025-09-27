# api/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile, shutil, os

from adk_project.agents.coordinator.agent import run_all

app = FastAPI(title="VeriFace DIA API", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # basic guard
    if not file.filename.lower().endswith((".mp4", ".mov", ".mkv", ".webm", ".wav", ".mp3")):
        raise HTTPException(status_code=400, detail="Please upload a media file (.mp4/.mov/.mkv/.webm/.wav/.mp3)")

    # save to a temp path
    with tempfile.TemporaryDirectory() as td:
        tmp_path = os.path.join(td, file.filename)
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            result = run_all(tmp_path, fps=25)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"analysis_failed: {e}")

    return JSONResponse(content=result)
