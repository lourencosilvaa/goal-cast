# ── Stage 1: Build frontend ────────────────────────────────────────────────────
FROM node:22-alpine AS frontend-build

WORKDIR /app

COPY src/frontend/package.json src/frontend/package-lock.json* ./

RUN npm install

COPY src/frontend/ .

ARG VITE_API_URL=""
ARG VITE_SUPABASE_URL
ARG VITE_SUPABASE_ANON_KEY

RUN npm run build

# ── Stage 2: Python backend (serves API + frontend static files) ───────────────
FROM python:3.12-slim

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

# Copy built frontend — served by FastAPI at /
COPY --from=frontend-build /app/dist /app/static

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
