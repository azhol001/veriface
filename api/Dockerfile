# veriface/api/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---- System deps often needed for media/CV/audio pipelines ----
# - ffmpeg: video/audio decoding
# - libgl1: OpenCV headless needs an OpenGL runtime
# - libsndfile1: common for audio libs (soundfile/torchaudio)
# - git: some ADK models/tools may pip-install from git
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libsndfile1 git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first to leverage layer caching
COPY api/requirements.txt .
RUN pip install -r requirements.txt

# Copy app code
COPY api /app/api

# (Optional) If your ADK code lives outside /api, bring it in here:
# COPY adk_project /app/adk_project

# Health & runtime
EXPOSE 8080

# Uvicorn FastAPI entrypoint
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
