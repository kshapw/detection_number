# Phone Number Detection Microservice

Fully offline REST API that extracts phone numbers from uploaded images using GPU-accelerated OCR.

---

## First-time deploy

```bash
git pull
docker compose up --build -d
docker compose ps
```

> First run downloads PaddleOCR models (~300 MB) into a named volume. Subsequent restarts are instant.

---

## Check logs

```bash
# Live logs — both services
docker compose logs -f

# App only
docker compose logs -f app

# Nginx only
docker compose logs -f nginx
```

---

## Test it

```bash
# Readiness (wait until you see "ready": true)
curl http://localhost:8080/ready

# Detect a phone number
curl -F "file=@your_image.jpg" http://localhost:8080/detect

# Liveness
curl http://localhost:8080/health
```

---

## Update after code change

```bash
git pull
docker compose build --no-cache && docker compose up -d
docker compose logs -f app
```

---

## Restart without rebuild

```bash
docker compose down
docker compose up -d
docker compose logs -f app
```

---

## Stop everything

```bash
docker compose down
```

---

## Pick a specific GPU (multi-GPU server)

Edit `docker-compose.yml` under the `app` service:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ["1"]      # ← change to GPU index you want (0, 1, 2 …)
          capabilities: [gpu]
```

Then:
```bash
docker compose up --build -d
```

---

## Check GPU usage inside container

```bash
docker exec -it detection-app-1 nvidia-smi
```

---

## API

### `POST /detect`

```bash
curl -F "file=@card.jpg" http://localhost:8080/detect
```

**Response:**
```json
{
  "phone_numbers": ["8722359047"],
  "raw_text": "Veeresh K\n51Y\nMALE\n8722359047\n...",
  "confidence_scores": [0.97, 0.99, 0.95],
  "latency_ms": 42.3
}
```

| Endpoint | Purpose |
|---|---|
| `POST /detect` | Upload image → get phone numbers |
| `GET /health` | Liveness probe |
| `GET /ready` | Readiness probe (model loaded?) |
| `GET /metrics` | Prometheus (internal network only) |

---

## Configuration (env vars, prefix `DETECT_`)

| Variable | Default | Description |
|---|---|---|
| `DETECT_USE_GPU` | `true` | Use CUDA for inference |
| `DETECT_OCR_LANG` | `en` | PaddleOCR language (`en`, `hi`, `kn`) |
| `DETECT_OCR_CONFIDENCE_THRESHOLD` | `0.6` | Min OCR confidence per text token |
| `DETECT_DEFAULT_PHONE_REGION` | `IN` | Country hint for phone parsing |
| `DETECT_MAX_UPLOAD_BYTES` | `10485760` | Upload size cap (10 MB) |
| `DETECT_ENABLE_DOCS` | `false` | Set `true` to expose `/docs` Swagger UI |
| `DETECT_CORS_ORIGINS` | `["*"]` | Restrict to specific origins in prod |
| `WORKERS` | `2` | Gunicorn worker count |
