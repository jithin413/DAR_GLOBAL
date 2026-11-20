# ---- DarGlobal & Wasalt Property Assistant ----
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBED_MODEL_DIR=/app/models/bge-small \
    PORT=7860

# non-root user (uid 1000 is what Hugging Face Spaces expects)
RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# bake the embedding model into the image (no download at start-up)
COPY app/embedder.py app/embedder.py
RUN touch app/__init__.py && python -c "from app.embedder import ensure_model; ensure_model()"

COPY app/ app/
COPY data/kb/ data/kb/
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/api/health',timeout=4)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
