# CUDA + cuDNN base (REQUIRED for GPU)
FROM nvidia/cuda:12.3.1-cudnn8-runtime-ubuntu22.04
# FROM nvidia/cuda:12.3.2-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libmagic1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Paddle GPU (IMPORTANT)
RUN pip install --no-cache-dir paddlepaddle-gpu==2.6.1

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app ./app
COPY gunicorn.conf.py .

# Expose port
EXPOSE 8000

# Start app
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app.main:app"]