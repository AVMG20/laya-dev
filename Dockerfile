# decider local API on CPU, for a server without a GPU (e.g. Coolify on a VPS).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Model files are downloaded on first start into this directory; mount a volume here
    # so a redeploy doesn't fetch the ~3.5 GB again.
    HF_HOME=/data/huggingface \
    DECIDER_HOST=0.0.0.0 \
    DECIDER_PORT=8000 \
    DECIDER_DEVICE=cpu

# CPU-only PyTorch: the default wheel bundles CUDA and is several GB larger.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install "decider-ai==1.2.1" "fastapi>=0.110" "uvicorn>=0.29"

RUN useradd --create-home --uid 1000 decider \
    && mkdir -p /data/huggingface \
    && chown -R decider:decider /data
USER decider
WORKDIR /app

COPY --chown=decider:decider decider_server.py index.html ./

EXPOSE 8000
# The first start downloads the model before the server listens, hence the long start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["python", "decider_server.py"]
