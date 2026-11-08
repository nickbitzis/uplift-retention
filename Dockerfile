# Build stage: resolve and install dependencies with uv into a virtualenv.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Runtime stage: slim image, non-root user, only the venv and app code.
FROM python:3.12-slim
RUN useradd --create-home --uid 10001 uplift
WORKDIR /app
COPY --from=build --chown=uplift:uplift /app /app
USER uplift
ENV PATH="/app/.venv/bin:$PATH" HOST=0.0.0.0 PORT=8830 UPLIFT_STUDY=/app/data/study.pkl
EXPOSE 8830
# The study (200,000 subscribers, 12 models) trains at first start: about 30 to 60 seconds.
HEALTHCHECK --interval=15s --timeout=3s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8830/healthz', timeout=2)"
CMD ["uplift", "serve"]
