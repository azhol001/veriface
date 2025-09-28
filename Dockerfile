# Root-level Dockerfile for VeriFace API (Cloud Run)
FROM python:3.12-slim

# System libs for mediapipe/opencv/audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy all project code (api/, adk_project/, etc.)
COPY . /app

# Cloud Run uses $PORT
ENV PORT=8080

# Start FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
