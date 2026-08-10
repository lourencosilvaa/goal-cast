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

# Run straight from the environment built above. Using `uv run` in CMD would
# re-sync against the DEFAULT dependency groups on every container start,
# re-downloading the full ML + dev toolchain and adding minutes to cold starts.
ENV PATH="/app/.venv/bin:$PATH"

COPY config/ config/
COPY src/__init__.py src/__init__.py
COPY src/backend/ src/backend/
# The wire contract the results service speaks. Imported by the gateway on the
# import path of the whole app, so its absence is not a broken feature — it is
# a container that will not start.
COPY src/contracts/ src/contracts/
# The 1X2 probability triple, shared with the offline models. Two files, not
# the package: everything else under src/models/ imports pandas, sklearn or
# xgboost, and copying it wholesale would put the ML stack back in an image
# built without it.
COPY src/models/__init__.py src/models/__init__.py
COPY src/models/outcome_model.py src/models/outcome_model.py
# Canonical team-name resolution is shared with the offline pipeline; the
# backend needs it to serve the admin alias-review screen.
COPY src/teams/ src/teams/

# Copy built frontend — served by FastAPI at /
COPY --from=frontend-build /app/dist /app/static

EXPOSE 8000

CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
