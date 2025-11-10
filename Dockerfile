# Multi-stage Dockerfile for COVID-19 Chest X-Ray Classification

# Stage 1: Base image with CUDA support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Stage 2: Development image
FROM base as development

WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-dev.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY tests/ ./tests/
COPY setup.py .
COPY pyproject.toml .
COPY README.md .

# Install package in editable mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p logs checkpoints mlruns wandb data

# Expose MLflow UI port
EXPOSE 5000

# Default command
CMD ["/bin/bash"]

# Stage 3: Production image (minimal)
FROM base as production

WORKDIR /app

# Copy only necessary files
COPY requirements.txt .

# Install only production dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only source code and configs
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY setup.py .
COPY README.md .

# Install package
RUN pip install --no-cache-dir .

# Create necessary directories
RUN mkdir -p logs checkpoints mlruns data

# Expose MLflow UI port
EXPOSE 5000

# Run training by default
CMD ["python", "-m", "scripts.train_mlflow"]

# Stage 4: Testing image
FROM development as testing

# Run tests on build
RUN pytest tests/ -v --cov=src

CMD ["pytest", "tests/", "-v"]
