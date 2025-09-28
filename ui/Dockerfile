# veriface/ui/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# Install deps first for caching
COPY ui/requirements.txt .
RUN pip install -r requirements.txt

# Copy app code
COPY ui /app/ui

# Port for Cloud Run
EXPOSE 8080

# Streamlit entrypoint (Cloud Run will set $PORT=8080; we pin to 8080)
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
