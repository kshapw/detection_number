# CUDA + cuDNN base (stable for Paddle)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libmagic1 \
    libffi-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

# -----------------------------
# Python setup
# -----------------------------
RUN pip install --no-cache-dir --upgrade pip

# Install Paddle GPU (compatible with CUDA 11.8)
RUN pip install --no-cache-dir paddlepaddle-gpu==2.5.2

# 🔥 Fix lmdb + cffi issues (VERY IMPORTANT)
RUN pip install --no-cache-dir cffi
RUN pip uninstall -y lmdb || true
RUN pip install --no-cache-dir lmdb==1.4.1

# -----------------------------
# App setup
# -----------------------------
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install app dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app ./app
COPY gunicorn.conf.py .

# -----------------------------
# Runtime
# -----------------------------
EXPOSE 8000

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app.main:app"]