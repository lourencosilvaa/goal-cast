FROM python:3.12-slim AS base

ARG GIT_SHA=unknown
ARG BUILD_TIME=unknown
ENV GIT_SHA=${GIT_SHA}
ENV BUILD_TIME=${BUILD_TIME}

RUN pip install uv

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock* .

RUN uv sync --frozen --no-dev

COPY config/ config/
COPY src/__init__.py src/__init__.py
COPY src/backend/ src/backend/
COPY src/models/ src/models/
COPY src/analysis/ src/analysis/
COPY src/scrapers/ src/scrapers/
COPY src/evaluation/ src/evaluation/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
