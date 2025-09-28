# ui/Dockerfile
FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ui /app/ui
EXPOSE 8080
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
