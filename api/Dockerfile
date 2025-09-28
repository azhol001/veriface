# veriface/api/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies for OpenCV, ffmpeg, librosa, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libsndfile1 git \
 && rm -rf /var/lib/apt/lists/*

# Install ALL deps from root
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy only API source code
COPY api /app/api

EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
