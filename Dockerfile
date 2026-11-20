# Offline IT Helpdesk API + web UI.
# The embedding model (23 MB) and the index are baked into the image, so the container
# needs no network at runtime. The LLM runs in a separate llama.cpp container (see
# docker-compose.yml) or is replaced by the deterministic extractive mode.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HELPDESK_CACHE=/opt/helpdesk/models \
    HELPDESK_KB_DIR=/app/kb \
    HELPDESK_INDEX_DIR=/app/index \
    HELPDESK_LLM=extractive

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[server,cpu]"

COPY kb ./kb
RUN helpdesk download && helpdesk index \
    && useradd --system --uid 10001 helpdesk && chown -R helpdesk /app
USER helpdesk

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
# 0.0.0.0 is needed inside a container; publish the port on 127.0.0.1 only (see compose).
CMD ["helpdesk", "serve", "--host", "0.0.0.0", "--port", "8000"]
