# syntax=docker/dockerfile:1.7
# ──────────────────────────────────────────────────────────────────────────────
# Phone Number Detection Microservice
# Base: NVIDIA CUDA 12.4.1 + cuDNN runtime on Ubuntu 22.04
# ──────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        python3.11-distutils \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libmagic1 \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && python -m pip install --upgrade pip setuptools wheel

# ── Create non-root user ───────────────────────────────────────────────────────
RUN groupadd --gid 1001 appgroup \
 && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

# ── Model cache volume mount point ─────────────────────────────────────────────
# PaddleOCR models will be downloaded here on first run, then persist via volume
RUN mkdir -p /models && chown -R appuser:appgroup /models

# ── Install Python dependencies ────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .

# Install GPU paddle first, then rest
RUN pip install paddlepaddle-gpu==2.6.1.post120 \
        -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html \
 && pip install -r requirements.txt

# ── Copy application source ────────────────────────────────────────────────────
COPY --chown=appuser:appgroup app/ ./app/
COPY --chown=appuser:appgroup gunicorn.conf.py .

# ── Switch to non-root user ───────────────────────────────────────────────────
USER appuser

# ── Runtime environment ────────────────────────────────────────────────────────
ENV HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=2 \
    LOG_LEVEL=info \
    DETECT_USE_GPU=true \
    DETECT_OCR_LANG=en \
    DETECT_ENABLE_DOCS=false \
    PADDLE_OCR_MODEL_DIR=/models \
    # Prevent Python from writing .pyc files to the read-only fs
    PYTHONDONTWRITEBYTECODE=1 \
    TMPDIR=/tmp

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["gunicorn", "app.main:app", "-c", "gunicorn.conf.py"]
