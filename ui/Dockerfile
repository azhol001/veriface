FROM python:3.12-slim

WORKDIR /app

# minimal deps for streamlit + requests
RUN pip install --no-cache-dir streamlit==1.37.* requests==2.*

# copy just the UI code
COPY streamlit_app.py /app/ui/streamlit_app.py

# Cloud Run listens on $PORT; Streamlit must bind to 0.0.0.0 and that port
ENV PORT=8080
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Default API_BASE; will override at deploy
ENV API_BASE="http://127.0.0.1:8080"

EXPOSE 8080

CMD ["streamlit", "run", "/app/ui/streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
