# Multi-stage Dockerfile for AI Plagiarism Detection System
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create directories with proper permissions BEFORE switching users
RUN mkdir -p /app /app/data /app/cache /home/appuser/.cache \
    && groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app /home/appuser

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
USER appuser

# Set cache environment variables for model downloads (FIXED!)
ENV TRANSFORMERS_CACHE=/app/cache \
    HF_HOME=/app/cache \
    TOKENIZERS_PARALLELISM=false

# Production stage
FROM base AS production

# Copy application code with proper ownership
COPY --chown=appuser:appuser apps/ ./apps/
COPY --chown=appuser:appuser tests/ ./tests/
COPY --chown=appuser:appuser sample_docs/ ./sample_docs/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment variables with secure defaults
ENV API_KEY="demo-secret" \
    MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2" \
    DATA_DIR="/app/data" \
    SIMILARITY_THRESHOLD="0.88" \
    CHUNK_SIZE="300" \
    OVERLAP="50"

# Run the application
CMD ["uvicorn", "apps.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
