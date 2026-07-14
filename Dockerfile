FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer between builds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and the trained model artifact
COPY app/ app/
COPY src/ src/
COPY models/ models/

ENV MODEL_PATH=models/model.joblib
ENV MODEL_VERSION=v1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
