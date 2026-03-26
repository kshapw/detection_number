"""Gunicorn configuration for production deployment."""

import multiprocessing
import os

# Bind
host = os.getenv("HOST", "0.0.0.0")
port = os.getenv("PORT", "8000")
bind = f"{host}:{port}"

# Worker class — Uvicorn async workers
worker_class = "uvicorn.workers.UvicornWorker"

# OCR startup is heavyweight, so keep the default modest unless CPU capacity
# and request volume justify more workers.
workers = int(os.getenv("WORKERS", 2))

# Timeouts
timeout = int(os.getenv("GUNICORN_TIMEOUT", 120))  # large images can be slow
graceful_timeout = 30
keepalive = 5

# Logging
loglevel = os.getenv("LOG_LEVEL", "info").lower()
accesslog = "-"   # stdout
errorlog = "-"    # stdout
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(D)sµs'

# Limits
worker_connections = 1000
max_requests = 500          # recycle workers to prevent memory leaks
max_requests_jitter = 50
