# api/Dockerfile
FROM python:3.11-slim
WORKDIR /app

# system deps for your packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libsndfile1 git \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY api /app/api
EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
