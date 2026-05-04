FROM python:3.11-slim

WORKDIR /app

# System deps for sentence-transformers / faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install episodic-memory library (editable mount → copy for production image)
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e ".[dev]" && \
    pip install --no-cache-dir flask>=3.0 gunicorn>=21.0

# Service entry point
COPY service/api.py ./service/api.py

EXPOSE 8099

# Gunicorn: 1 worker (BGE model is not fork-safe), 120s timeout for model load
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8099", \
     "--workers", "1", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "service.api:app"]
