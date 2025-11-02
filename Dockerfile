# Multi-stage build for production optimization
FROM python:3.10-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim

# Create non-root user for security
RUN groupadd -r churnapi && useradd -r -g churnapi churnapi

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY api/ ./api/
COPY models/ ./models/
COPY logs/ ./logs/

# Create necessary directories and set permissions
RUN mkdir -p /app/logs && \
    chown -R churnapi:churnapi /app

# Switch to non-root user
USER churnapi

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Security: Run with limited privileges
EXPOSE 8000

# Use production-optimized settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CHURN_API_LOG_LEVEL=INFO
ENV CHURN_API_DEBUG=false

# Production command with process management
CMD ["uvicorn", "api.deployment_api:api", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--access-log", \
     "--log-level", "info"]