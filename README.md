# Football Prediction Agent

A full-stack web application that combines a **ML ensemble model** with **AI-powered contextual analysis** (via Google Gemini) to generate football match predictions and identify value betting opportunities across Europe's top leagues.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   FRONTEND (React/Vite)                  │
│  Dashboard │ Value Bets │ Settings │ Admin Panel         │
│  Supabase Auth (email/password) │ Per-user Gemini key   │
└────────────────────────┬────────────────────────────────┘
                         │ HTTPS
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  BACKEND (FastAPI)                        │
│  /api/predictions │ /api/leagues │ /api/ai/analyze       │
│  /api/admin │ /api/keys │ /api/export │ /api/status      │
└──────────┬──────────────────────┬───────────────────────┘
           │                      │
           ▼                      ▼
┌──────────────────┐   ┌──────────────────────────────────┐
│  Hugging Face    │   │            Supabase               │
│  Private Repo    │   │  Auth │ user_profiles             │
│  ML model files  │   │  user_api_keys │ app_settings     │
│  + datasets/     │   └──────────────────────────────────┘
│  (Parquet files) │
└──────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│               ML PIPELINE (5 layers)                     │
│  HuggingFace datasets/ (8 seasons × 6 leagues)          │
│  Feature Engineering → Ensemble Model → Value Detection  │
└─────────────────────────────────────────────────────────┘
```

---

## ML Pipeline

### Data Ingestion

Historical match data covers **8 seasons** across **6 leagues**, sourced from [football-data.co.uk](https://www.football-data.co.uk/) and stored on **Hugging Face** as per-league Parquet files (no local disk required on Render):

| League | Code | Seasons |
|--------|------|---------|
| Premier League | `E0` | 2018/19 → 2025/26 |
| La Liga | `SP1` | 2018/19 → 2025/26 |
| Bundesliga | `D1` | 2018/19 → 2025/26 |
| Serie A | `I1` | 2018/19 → 2025/26 |
| Ligue 1 | `F1` | 2018/19 → 2025/26 |
| Liga Portugal | `P1` | 2018/19 → 2025/26 |

**Data source priority:** HuggingFace Parquet → local CSV cache (24 h TTL) → football-data.co.uk download. On Render (ephemeral filesystem), all data comes from HuggingFace. The weekly GitHub Action refreshes data from the web and re-uploads to HF.

**Columns retained**: Date, teams, full-time & half-time goals/results, shots, fouls, corners, cards, and Bet365 odds (B365H/D/A).

**Training metadata** is stored in `training_results.json` (uploaded to HF alongside the model) and includes the `seasons_trained` field — the exact list of seasons that contributed data to the current model.

### Feature Engineering

| Feature Group | Description |
|---------------|-------------|
| Rolling stats | 5-match rolling averages: goals, shots, corners, fouls, points |
| ELO ratings | K=32, home advantage +65, initial rating 1500 |
| xG proxy | Historical conversion rates (SoT 30%, shots 3%), no same-match leakage |
| Fatigue | Rest days between matches, fatigue flag (≤3 days), midweek flag |
| Head-to-Head | Last 5 meetings: win %, draw %, avg total goals |
| Draw features | Historical draw rates, form gap, attack/defense similarity |

### Ensemble Model

Four classifiers combined via soft voting (weights `[1, 1, 2]`):

| Model | Key Hyperparameters |
|-------|---------------------|
| Logistic Regression | `C=0.5`, `max_iter=1000`, `class_weight=balanced` |
| Random Forest | `n_estimators=200`, `max_depth=8`, `min_samples_leaf=10` |
| XGBoost (×2 weight) | `n_estimators=200`, `max_depth=5`, `lr=0.05`, `subsample=0.8` |
| Gradient Boosting | `n_estimators=150`, `max_depth=4`, `lr=0.08` (comparison only) |

Cross-validation uses `TimeSeriesSplit` with 5 folds to prevent data leakage. Output: 3-class probabilities (Home Win / Draw / Away Win).

### Value Bet Detection

A value bet is flagged when `ML probability > bookmaker implied probability + 3%`.

- **Kelly Criterion** stake sizing: `f = (b×p − q) / b`, capped at 25%
- **KL divergence** between ML and bookmaker distributions
- **Blended probabilities**: 50% ML + 30% bookmaker average + 20% best odds

---

## Application Features

### Dashboard
- Daily match predictions for all 6 leagues
- Confidence levels and blended probabilities
- AI analysis per match (Google Gemini)
- Export predictions to CSV or Excel

### Value Bets
- Auto-detected value opportunities (edge ≥ 3%)
- Kelly stake recommendations and KL divergence scores

### Settings
- Per-user Google Gemini API key (stored encrypted in Supabase)
- Gemini model selection (2.5 Flash / 2.5 Pro / 2.0 Flash / 2.0 Flash Lite)

### Admin Panel
- Create new users (invite-only)
- Approve or revoke user access
- Real-time user list with status badges (Admin / Active / Revoked)

### Auth
- Email/password login via Supabase
- Self-registration — new accounts require admin approval before access
- Pending approval screen shown to unapproved users

### Retraining Banner
- Amber banner shown automatically when the ML model is being retrained
- Polls backend every 30 seconds; dismissible per session

---

## Project Structure

```
football-prediction-agent/
├── .github/
│   └── workflows/
│       ├── retrain.yml        # Weekly model retraining + HF upload + Render redeploy
│       └── render.yaml        # Render deployment config (backend + frontend)
├── config/
│   ├── config.yaml            # Centralized configuration (no hardcoded values)
│   └── config_loader.py       # Pydantic config loader (HuggingFaceConfig, DataConfig, …)
├── src/
│   ├── backend/
│   │   ├── api/
│   │   │   ├── admin.py       # User management (list, create, approve/revoke)
│   │   │   ├── ai.py          # Gemini match analysis
│   │   │   ├── evaluation.py  # Model evaluation stats
│   │   │   ├── exports.py     # CSV / Excel export
│   │   │   ├── keys.py        # Per-user Gemini key CRUD
│   │   │   ├── leagues.py     # Available leagues
│   │   │   ├── predictions.py # Match predictions
│   │   │   ├── profile.py     # User profile + self-registration
│   │   │   └── status.py      # Retraining status flag
│   │   ├── core/
│   │   │   ├── auth.py              # JWT validation via Supabase
│   │   │   ├── encryption.py        # Fernet symmetric encryption
│   │   │   └── supabase_client.py   # Supabase service-role client singleton
│   │   ├── services/
│   │   │   ├── api_key_service.py       # Encrypted key storage
│   │   │   ├── app_settings_service.py  # app_settings table CRUD
│   │   │   ├── model_loader.py          # HuggingFace model + dataset download
│   │   │   ├── prediction_service.py    # Core prediction logic
│   │   │   └── user_service.py          # user_profiles table CRUD
│   │   └── main.py            # FastAPI app with lifespan (model + datasets preload on startup)
│   ├── frontend/              # React + TypeScript + Vite
│   │   └── src/
│   │       ├── components/    # GlassCard, NeonButton, RetrainingBanner, layout, …
│   │       ├── contexts/      # AuthContext (Supabase session + profile)
│   │       ├── lib/           # api.ts (all backend calls), supabase.ts
│   │       ├── pages/         # Dashboard, ValueBets, Settings, Admin, Login
│   │       └── App.tsx        # Routes (ProtectedRoute, AdminRoute)
│   ├── models/                # ML pipeline (data_loader, feature_engineer, trainer, predictor)
│   ├── scrapers/              # Odds scrapers (Betclic, Betano, Solverde)
│   └── analysis/              # Value detection, KL divergence, Poisson match stats
├── scripts/
│   ├── train_model.py         # Train ML ensemble locally (writes seasons_trained to results)
│   ├── upload_to_hf.py        # Upload model artefacts + per-league Parquet datasets to HF
│   ├── find_value_bets.py     # Full prediction + value pipeline
│   └── predict_match.py       # Predict a single match
├── tests/                     # Test suite mirroring src/ (100% coverage on data_loader)
├── datasets/cache/            # Local CSV cache from football-data.co.uk (ephemeral on Render)
├── output/                    # Reports, trained models, exports
├── Dockerfile                 # Backend Docker image (Python 3.12-slim + uv)
└── pyproject.toml             # Python deps managed by uv
```

### HuggingFace Repo Layout

After each weekly retrain, the HF repo contains:

```
{HF_REPO_ID}/
├── ensemble_model.joblib       # Trained ensemble
├── scaler.joblib               # StandardScaler
├── feature_names.joblib        # List of 59 feature names
├── training_results.json       # CV scores, accuracy, seasons_trained, last_match_date
└── datasets/
    ├── E0.parquet              # Premier League — all seasons merged
    ├── SP1.parquet             # La Liga
    ├── D1.parquet              # Bundesliga
    ├── I1.parquet              # Serie A
    ├── F1.parquet              # Ligue 1
    └── P1.parquet              # Liga Portugal
```

Parquet format is ~3–5× smaller than CSV and loads in a single read per league on backend startup.

---

## Deployment

The app runs as two separate services on **Render**, backed by **Supabase** (auth + DB) and **Hugging Face** (model + dataset storage).

### Services at a glance

| Service | Type | Runtime |
|---------|------|---------|
| Backend API | Render Web Service | Docker (`Dockerfile` at repo root) |
| Frontend | Render Static Site | Node build from `src/frontend` |
| Database & Auth | Supabase | — |
| ML Model + Datasets | Hugging Face Hub | Private repo (`datasets/` subfolder) |

---

### Step 1 — Supabase

Create a Supabase project and run this SQL in the **SQL Editor**:

```sql
-- User profiles (access control)
CREATE TABLE public.user_profiles (
  user_id    UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email      TEXT NOT NULL,
  approved   BOOLEAN NOT NULL DEFAULT false,
  is_admin   BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE public.user_profiles ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_role_full_access" ON public.user_profiles
  FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "users_read_own_profile" ON public.user_profiles
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

-- Encrypted per-user Gemini API keys
CREATE TABLE public.user_api_keys (
  user_id    UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  service    TEXT NOT NULL,
  key_enc    TEXT NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE public.user_api_keys ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_role_full_access" ON public.user_api_keys
  FOR ALL TO service_role USING (true) WITH CHECK (true);

-- App settings (retraining flag, etc.)
CREATE TABLE public.app_settings (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
ALTER TABLE public.app_settings ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_role_full_access" ON public.app_settings
  FOR ALL TO service_role USING (true) WITH CHECK (true);

-- Grant yourself admin access (run after your first login)
INSERT INTO public.user_profiles (user_id, email, approved, is_admin)
SELECT id, email, true, true
FROM auth.users
WHERE email = 'your-email@example.com'
ON CONFLICT (user_id) DO UPDATE SET approved = true, is_admin = true;
```

From **Supabase → Settings → API**, collect:

| Key | Used by |
|-----|---------|
| Project URL | Backend (`SUPABASE_URL`) + Frontend (`VITE_SUPABASE_URL`) |
| `service_role` secret | Backend only (`SUPABASE_SERVICE_KEY`) — never expose |
| `anon` public key | Frontend (`VITE_SUPABASE_ANON_KEY`) |

---

### Step 2 — Hugging Face

1. Create a **private** model repository at [huggingface.co](https://huggingface.co/)
2. Generate a **User Access Token** with write permissions
3. Train and upload the initial model and datasets:

```bash
uv run python scripts/train_model.py
HF_TOKEN=hf_... HF_REPO_ID=username/repo uv run python scripts/upload_to_hf.py
```

The upload script will:
- Push model artefacts (`.joblib`, `.json`) to the repo root
- Convert all CSV cache files to per-league Parquet files and push to `datasets/`

Collect:
- Token → `HF_TOKEN`
- Repo ID (`username/repo-name`) → `HF_REPO_ID`

---

### Step 3 — Backend (Render Web Service)

Create a **Web Service** on Render with runtime **Docker** pointing to the repo root.

| Environment Variable | Description | How to obtain |
|----------------------|-------------|---------------|
| `SUPABASE_URL` | Supabase project URL | Supabase → Settings → API |
| `SUPABASE_SERVICE_KEY` | Service role secret | Supabase → Settings → API |
| `ENCRYPTION_KEY` | Fernet key for encrypting Gemini keys at rest | `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `HF_TOKEN` | Hugging Face access token | HF → Settings → Tokens |
| `HF_REPO_ID` | HF repo ID (`user/repo`) | Your HF repo page |
| `RETRAIN_API_KEY` | Secret for the retraining webhook | `openssl rand -hex 32` |

Set the **Health check path** to `/api/leagues`.

On startup the backend downloads the entire HF repo (model + `datasets/` Parquet files) to `/tmp/hf_models`, then serves predictions directly from the Parquet files — no local CSV cache needed.

After the service is created, copy its public URL — you'll need it for the frontend and GitHub Actions.

---

### Step 4 — Frontend (Render Static Site)

Create a **Static Site** on Render:
- **Build command**: `cd src/frontend && npm install && npm run build`
- **Publish directory**: `src/frontend/dist`
- **Rewrite rule**: `/* → /index.html`

| Environment Variable | Value |
|----------------------|-------|
| `VITE_API_URL` | Backend Render URL (e.g. `https://football-agent-api.onrender.com`) |
| `VITE_SUPABASE_URL` | Supabase project URL |
| `VITE_SUPABASE_ANON_KEY` | Supabase anon public key |

---

### Step 5 — GitHub Actions Secrets

Add these in **GitHub → Settings → Secrets and variables → Actions**:

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | Hugging Face access token |
| `HF_REPO_ID` | HF repo ID (`user/repo`) |
| `RENDER_DEPLOY_HOOK_URL` | Render deploy hook (backend service → Settings → Deploy Hook) |
| `RENDER_BACKEND_URL` | Backend public URL |
| `RETRAIN_API_KEY` | Same value as the backend env var |

The workflow (`.github/workflows/retrain.yml`) runs **every Monday at 06:00 UTC** and can be triggered manually. It:
1. Sets the retraining banner flag → users see the amber banner
2. Downloads fresh data from football-data.co.uk (current season updated weekly)
3. Retrains the model across all 8 seasons
4. Uploads model artefacts + updated Parquet datasets to Hugging Face
5. Triggers a Render redeploy → backend restarts and loads everything from HF
6. Clears the retraining banner flag

---

### Deployment overview

```
GitHub Actions (every Monday 06:00 UTC)
  └─► sets retraining=true (Supabase app_settings)
  └─► downloads fresh data from football-data.co.uk
  └─► retrains model (8 seasons × 6 leagues)
  └─► uploads model + datasets/ Parquet to Hugging Face
  └─► triggers Render deploy hook
  └─► sets retraining=false

Render (Backend)
  └─► Docker image built from Dockerfile
  └─► on startup: downloads model + datasets/ from Hugging Face → /tmp/hf_models
  └─► FootballDataLoader reads from /tmp/hf_models/datasets/*.parquet
  └─► reads SUPABASE_URL / SUPABASE_SERVICE_KEY
  └─► uses ENCRYPTION_KEY to decrypt per-user Gemini keys

Render (Frontend)
  └─► static build from src/frontend
  └─► calls backend via VITE_API_URL
  └─► Supabase auth via VITE_SUPABASE_URL + VITE_SUPABASE_ANON_KEY

Supabase
  └─► user_profiles  (approved, is_admin)
  └─► user_api_keys  (encrypted Gemini keys)
  └─► app_settings   (retraining flag)
```

---

## Local Development

### Prerequisites
- Python ≥ 3.12 with [uv](https://docs.astral.sh/uv/)
- Node.js ≥ 18

### Backend

```bash
uv sync
uv run uvicorn src.backend.main:app --reload --port 8000
```

Create a `.env` file (or export directly):

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
ENCRYPTION_KEY=your-fernet-key
# Optional — skips HF download if absent; data is fetched from football-data.co.uk instead
HF_TOKEN=hf_...
HF_REPO_ID=username/repo
RETRAIN_API_KEY=any-local-secret
```

### Frontend

```bash
cd src/frontend
npm install
npm run dev
```

Create `src/frontend/.env.local`:

```
VITE_API_URL=http://localhost:8000
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
```

### ML Pipeline (standalone)

```bash
# Train the model (downloads data, writes seasons_trained to training_results.json)
uv run python scripts/train_model.py

# Upload model + Parquet datasets to Hugging Face
HF_TOKEN=hf_... HF_REPO_ID=user/repo uv run python scripts/upload_to_hf.py

# Value bets pipeline
uv run python scripts/find_value_bets.py --league "Liga Portugal"

# Single match prediction
uv run python scripts/predict_match.py --home "Benfica" --away "Porto" --league P1
```

### Tests

```bash
# Full suite
uv run pytest tests/

# With coverage
uv run pytest tests/ --cov=src --cov-report=term-missing
```

### Code Quality

```bash
uv run black src/
uv run ruff check src/
uv run mypy src/
```
