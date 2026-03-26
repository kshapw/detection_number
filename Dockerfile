FROM python:3.8-slim

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libgomp1 \
    libmagic1 \
    libffi-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Python setup
# -----------------------------
RUN pip install --no-cache-dir --upgrade pip

# Install Paddle CPU for inference on non-GPU hosts.
RUN pip install --no-cache-dir paddlepaddle==2.6.2

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
