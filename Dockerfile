FROM python:3.12-slim AS base

ARG GIT_SHA=unknown
ARG BUILD_TIME=unknown
ENV GIT_SHA=${GIT_SHA}
ENV BUILD_TIME=${BUILD_TIME}

RUN pip install uv

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock* .

# Install ONLY the backend dependency group (no ML libs like pandas/xgboost/sklearn)
RUN uv sync --frozen --no-dev --only-group backend

COPY config/ config/
COPY src/__init__.py src/__init__.py
COPY src/backend/ src/backend/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
