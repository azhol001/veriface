# Dockerfile at repo root (optional helper)
# syntax=docker/dockerfile:1.7
ARG SERVICE=api
ARG PY_BASE=python:3.11-slim

FROM ${PY_BASE} as base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Common system packages (tuned for api needs; harmless for ui)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libsndfile1 git \
 && rm -rf /var/lib/apt/lists/*

# --- Install deps per service (expects requirements.txt in subfolder) ---
ARG SERVICE
COPY ${SERVICE}/requirements.txt requirements.txt
RUN pip install -r requirements.txt

# --- Copy service code ---
COPY ${SERVICE} /app/${SERVICE}

# --- Final image/command per service ---
# We use a tiny launch script so CMD can switch at runtime based on SERVICE
RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -e' \
    'if [ "$SERVICE" = "api" ]; then' \
    '  exec uvicorn api.main:app --host 0.0.0.0 --port 8080' \
    'elif [ "$SERVICE" = "ui" ]; then' \
    '  exec streamlit run ui/streamlit_app.py --server.port=8080 --server.address=0.0.0.0' \
    'else' \
    '  echo "Unknown SERVICE=$SERVICE (use api|ui)"; exit 2' \
    'fi' \
    > /usr/local/bin/entrypoint.sh \
 && chmod +x /usr/local/bin/entrypoint.sh

ENV SERVICE=${SERVICE}
EXPOSE 8080
CMD ["/usr/local/bin/entrypoint.sh"]
