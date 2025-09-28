FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# System libs needed by your deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libsndfile1 git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (cache-friendly)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app code
COPY api /app/api
# IMPORTANT: include your pipeline package if it lives at repo root
# If your repo has this folder, the next line MUST be present:
COPY adk_project /app/adk_project

# debug: list contents during build (remove later)
RUN ls -la /app && ls -la /app/adk_project

EXPOSE 8080
CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8080"]

