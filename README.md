# Football Prediction Agent

A full-stack web application that combines a **ML ensemble model** with **AI-powered contextual analysis** (via Google Gemini) to generate football match predictions and identify value betting opportunities across Europe's top leagues.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   FRONTEND (React/Vite)                  │
│  Dashboard │ Value Bets │ Settings │ Admin Panel         │
│  Supabase Auth │ Per-user Gemini + NVIDIA keys           │
│  "Calcular ao vivo" — on-demand inference button         │
└────────────────────────┬────────────────────────────────┘
                         │ HTTPS
                         ▼
┌─────────────────────────────────────────────────────────┐
│              BACKEND (FastAPI — lightweight)              │
│  /api/predictions │ /api/leagues │ /api/ai/analyze       │
│  /api/admin │ /api/keys │ /api/export │ /api/status      │
│  /api/predictions/infer  ◄── on-demand inference proxy   │
│  Reads pre-computed predictions from Supabase (~50 MB)   │
└──────────────┬──────────────────────────┬───────────────┘
               │ pre-computed             │ on-demand
               ▼                          ▼
┌──────────────────────┐   ┌─────────────────────────────┐
│       Supabase        │   │   HuggingFace Space          │
│  Auth                 │   │   Docker FastAPI service     │
│  user_profiles        │   │   POST /infer                │
│  user_api_keys        │   │   Loads model at startup     │
│  app_settings         │   │   Fetches fixtures → feature │
│  predictions (JSONB)  │   │   engineering → ensemble     │
└──────────▲────────────┘   └──────────────▲──────────────┘
           │                               │
┌──────────┴──────────┐     ┌──────────────┴──────────────┐
│   GitHub Actions     │     │      Hugging Face Hub        │
│  daily + on retrain  │     │  Private repo                │
│  run_inference.py    │     │  ML model (.joblib)          │
│  uploads JSONB to    │     │  datasets (.parquet)         │
│  Supabase            │     │  Updated weekly by retrain   │
└──────────────────────┘     └─────────────────────────────┘
```

**Two inference paths:**
- **Offline (default):** GitHub Actions runs `run_inference.py` daily at 06:30 UTC, uploads pre-computed predictions to Supabase. The backend just reads them — no ML at runtime, under 50 MB.
- **On-demand ("Calcular ao vivo"):** The backend proxies a request to the **HuggingFace Space**, which holds the model in memory, fetches live fixtures, runs the full pipeline, and returns predictions in seconds.

---

## ML Pipeline

### Data Ingestion

Historical match data covers **10 seasons** (2017/18 → 2026/27) across **21 divisions**, ~64,000 matches, sourced from [football-data.co.uk](https://www.football-data.co.uk/) and stored on **Hugging Face** as per-league Parquet files (no local disk required on Render):

| Country | Codes | Divisions |
|---------|-------|-----------|
| England | `E0` `E1` `E2` `E3` | Premier League, Championship, League One, League Two |
| Scotland | `SC0` `SC1` `SC2` `SC3` | Premiership, Championship, League One, League Two |
| Spain | `SP1` `SP2` | La Liga, La Liga 2 |
| Germany | `D1` `D2` | Bundesliga, 2. Bundesliga |
| Italy | `I1` `I2` | Serie A, Serie B |
| France | `F1` `F2` | Ligue 1, Ligue 2 |
| Netherlands | `N1` | Eredivisie |
| Belgium | `B1` | Jupiler Pro League |
| Portugal | `P1` | Liga Portugal |
| Turkey | `T1` | Super Lig |
| Greece | `G1` | Super League Greece |

`EC` (National League) is deliberately excluded: it ships no shot, foul or corner columns, so `FeatureEngineer.build_match_features` would drop every one of its rows while its teams still appeared on team pages.

Second tiers earn their place beyond raw volume — a promoted side arrives carrying a real ELO rating instead of entering at the 1500 default. Measured on an identical 2,040-fixture holdout, expanding from 6 to 21 divisions changed prediction quality in the original six by **−0.0020 log-loss, 95% CI [−0.0062, +0.0023]** — no detectable effect either way.

**Two data-integrity guards** live in `FootballDataLoader`, both learned the hard way:
- *Division verification.* football-data.co.uk 301-redirects a not-yet-published season's file onto a different division (`2627/SP1.csv` → `2627/SC1.csv`). Payloads are checked against the `Div` column before being cached, or Scottish Championship matches end up labelled "La Liga".
- *Tolerant decoding.* Some season files carry a stray cp1252 byte in an odds column (1819/SC0 and 1819/I2 both do). A strict UTF-8 read aborts and silently drops the whole season, so reads fall back to latin-1.

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

Three classifiers combined via soft voting (weights `[1, 1, 2]`):

| Model | Key Hyperparameters |
|-------|---------------------|
| Logistic Regression | `C=0.5`, `max_iter=1000`, `class_weight=balanced` |
| Random Forest | `n_estimators=200`, `max_depth=8`, `min_samples_leaf=10` |
| XGBoost (×2 weight) | `n_estimators=400`, `max_depth=3`, `lr=0.01`, `subsample=0.8`, `colsample_bytree=0.9` |

*(Gradient Boosting was evaluated and dropped — it was dominated by XGBoost in `TimeSeriesSplit` CV and is redundant with it.)*

Training uses `TimeSeriesSplit` (5 folds) to prevent leakage, **exponential time-decay sample weighting** (half-life 540 days — recent matches count more), and leakage-safe median imputation fit on the training fold only.

**Probability calibration** — the fitted ensemble is wrapped in `CalibratedClassifierCV` (via `FrozenEstimator`) on a held-out chronological slice, so reported probabilities match observed frequencies. This is what makes value-bet edges trustworthy; on this dataset it is roughly log-loss-neutral but improves reliability. Toggle via `model.calibration` (`sigmoid`/`isotonic`).

**Hyperparameter search** — `scripts/tune_xgboost.py` runs a bounded, leakage-safe (`TimeSeriesSplit`) randomized search minimising log-loss. Report-only: it prints the best params, never mutates config.

Output: 3-class probabilities (Home Win / Draw / Away Win), blended with the Dixon-Coles model below.

### Dixon-Coles Score Model

A **Dixon-Coles bivariate-Poisson** model (`src/models/poisson/`) models goals directly — MLE-fit attack/defense strengths per team, a global home advantage, and the low-score `rho` correction, with exponential time-decay weighting. It provides:

- **Calibrated score markets** — scorelines, Over/Under (1.5/2.5/3.5), BTTS and expected goals — replacing the earlier naive independent-Poisson calculator (kept only as a fallback when the model doesn't know a team).
- **A 1X2 distribution blended into the ensemble** — `model.poisson.blend_weight` (default `0.4`, empirically optimal via `scripts/tune_blend_weight.py`; the blend beats both the ensemble and the Poisson model alone).

### Value Bet Detection

A value bet is flagged when `ML probability > bookmaker implied probability + 3%`, where the ML probability is the calibrated, Poisson-blended 1X2 output.

- **Kelly Criterion** stake sizing: `f = (b×p − q) / b`, capped at 25%
- **KL divergence** between ML and bookmaker distributions
- **Blended probabilities**: 50% ML + 30% bookmaker average + 20% best odds

---

## Application Features

### Dashboard
- Daily match predictions for all 21 divisions
- Confidence levels and blended probabilities
- AI analysis per match (Google Gemini)
- **On-demand inference** ("Calcular ao vivo") — calls the HF Space directly for live predictions
- Export predictions to CSV / Excel
- Date picker to browse predictions for upcoming fixtures

### Value Bets
- Auto-detected value opportunities (edge ≥ 3%)
- Kelly stake recommendations and KL divergence scores

### Settings
- Per-user Google Gemini API key (stored encrypted in Supabase)
- Per-user NVIDIA API key (same encrypted storage, separate `service` record)
- Gemini model selection (2.5 Flash / 2.5 Pro / 2.0 Flash / 2.0 Flash Lite)

### Admin Panel
- Create new users (invite-only)
- Approve or revoke user access
- Real-time user list with status badges (Admin / Active / Revoked)

### Auth
- Email/password login via Supabase
- Self-registration — new accounts require admin approval before access
- Pending approval screen shown to unapproved users
- **Logout button** available in the sidebar and mobile navigation

### Retraining Banner
- Amber banner shown automatically when the ML model is being retrained
- Polls backend every 30 seconds; dismissible per session

---

## Project Structure

```
football-prediction-agent/
├── .github/
│   └── workflows/
│       ├── retrain.yml            # Weekly retrain (skips upload/redeploy if no new data)
│       ├── run-inference.yml      # Daily inference → Supabase upload
│       ├── deploy.yml             # Build Docker image → Render redeploy
│       └── deploy-hf-space.yml    # Sync hf_space/ → HuggingFace Space
├── config/
│   ├── config.yaml                # Centralized configuration
│   └── config_loader.py           # Pydantic config loader (env var overrides)
├── hf_space/                      # Self-contained HuggingFace Space (Docker)
│   ├── Dockerfile                 # Python 3.11, uvicorn, port 7860
│   ├── requirements.txt           # FastAPI + full ML stack
│   ├── app.py                     # FastAPI: /health + POST /infer
│   ├── config/
│   │   ├── config.yaml            # Space-specific config
│   │   └── config_loader.py       # Minimal Pydantic loader + env var injection
│   └── src/models/                # Copies of predictor, feature_engineer, data_loader
├── src/
│   ├── backend/                   # Lightweight FastAPI (no ML deps at runtime)
│   │   ├── api/
│   │   │   ├── admin.py           # User management (list, create, approve/revoke)
│   │   │   ├── ai.py              # Gemini match analysis
│   │   │   ├── evaluation.py      # Model evaluation stats
│   │   │   ├── exports.py         # CSV/Excel export (reads from Supabase)
│   │   │   ├── inference.py       # POST /api/predictions/infer (proxy to HF Space)
│   │   │   ├── keys.py            # Per-user Gemini + NVIDIA key CRUD
│   │   │   ├── leagues.py         # Available leagues
│   │   │   ├── predictions.py     # Match predictions (reads from Supabase)
│   │   │   ├── profile.py         # User profile + self-registration
│   │   │   └── status.py          # Retraining status flag
│   │   ├── core/
│   │   │   ├── auth.py            # JWT validation via Supabase
│   │   │   ├── encryption.py      # Fernet symmetric encryption
│   │   │   └── supabase_client.py # Supabase service-role client singleton
│   │   ├── services/
│   │   │   ├── api_key_service.py       # Encrypted key storage (Gemini + NVIDIA)
│   │   │   ├── app_settings_service.py  # app_settings table CRUD
│   │   │   ├── inference_service.py     # HTTP client → HF Space /infer
│   │   │   ├── model_loader.py          # HuggingFace download (inference script)
│   │   │   ├── prediction_service.py    # Reads predictions from Supabase
│   │   │   └── user_service.py          # user_profiles table CRUD
│   │   └── main.py                # FastAPI app (lightweight startup, no ML)
│   ├── frontend/                  # React + TypeScript + Vite
│   │   └── src/
│   │       ├── components/        # GlassCard, NeonButton, RetrainingBanner, …
│   │       │   └── layout/        # Sidebar + MobileNav (with logout button)
│   │       ├── contexts/          # AuthContext (Supabase session + profile)
│   │       ├── lib/               # api.ts (all backend calls), supabase.ts
│   │       ├── pages/             # Dashboard, ValueBets, Settings, Admin, Login
│   │       └── App.tsx            # Routes (ProtectedRoute, AdminRoute)
│   ├── models/                    # ML pipeline (data_loader, feature_engineer, trainer, predictor)
│   ├── scrapers/
│   │   ├── fixtures_fetcher.py    # football-data.co.uk CSV → OddsAPI → FlashScore
│   │   ├── flashscore/            # FlashScore scraper (HTTP + Playwright fallback)
│   │   └── ...                    # Odds scrapers (Betclic, Betano, Solverde)
│   └── analysis/                  # Value detection, KL divergence, Poisson match stats
├── scripts/
│   ├── train_model.py             # Train ensemble + calibration + Dixon-Coles
│   ├── tune_xgboost.py            # Offline XGBoost log-loss search (report-only)
│   ├── tune_blend_weight.py       # Offline ensemble/Poisson blend-weight sweep
│   ├── tune_european_weight.py    # Offline European Dixon-Coles/ELO blend sweep
│   ├── upload_to_hf.py            # Upload model + Parquet datasets to HF
│   ├── restart_hf_space.py        # Restart the HF Space so it reloads the new data
│   ├── run_inference.py           # Offline inference → Supabase upload
│   ├── find_value_bets.py         # Full prediction + value pipeline (local)
│   └── predict_match.py           # Predict a single match (local)
├── supabase/migrations/           # SQL migrations for Supabase tables
├── tests/                         # Test suite mirroring src/
├── datasets/cache/                # Local CSV cache (ephemeral on Render)
├── output/                        # Reports, trained models, exports
├── Dockerfile                     # Slim backend image (no ML deps)
└── pyproject.toml                 # Python deps managed by uv
```

### HuggingFace Repo Layout

After each weekly retrain, the HF repo contains:

```
{HF_REPO_ID}/
├── ensemble_model.joblib       # Trained (calibrated) ensemble
├── poisson_model.joblib        # Dixon-Coles score model (1X2 blend + markets)
├── scaler.joblib               # Imputer + StandardScaler pipeline
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
| On-demand Inference | HuggingFace Space | Docker (`hf_space/`) |
| ML Model + Datasets | Hugging Face Hub | Private repo (`datasets/` subfolder) |

---

### Step 0 — External APIs and keys

Everything the project talks to, and whether you need to register for it.

#### Required

| Service | Key | Free tier | Used for | Set in |
|---|---|---|---|---|
| [Supabase](https://supabase.com/) | Project URL, `service_role` secret, `anon` key | Free plan is sufficient | Auth, `predictions`, `team_aliases`, encrypted per-user API keys | Render backend, Render frontend, GitHub Actions |
| [Hugging Face](https://huggingface.co/settings/tokens) | User Access Token (write) | Free | Model + dataset storage, Space hosting | HF Space secrets, GitHub Actions |

#### No key required

| Source | Used for | Notes |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/) | Domestic results (21 leagues) and `fixtures.csv` with Bet365 odds | No registration, no rate limit. The primary data source. |
| [openfootball](https://github.com/openfootball/champions-league) | European results corpus (CL/EL/UECL) | Public domain, plain `git clone`. See [docs/european-competitions.md](docs/european-competitions.md). |

#### Optional

| Service | Key | Free tier | Used for | Set in |
|---|---|---|---|---|
| [The Odds API](https://the-odds-api.com/) | `THE_ODDS_API_KEY` | 500 credits/month | Fixture + odds **fallback** when `fixtures.csv` is empty. Also the only configured odds source for CL/EL/UECL (`odds_api.competitions`). | GitHub Actions (`run-inference.yml`) |
| [Google Gemini](https://aistudio.google.com/apikey) | — | — | AI match commentary | **Per user, not an env var.** Each user pastes their own key into Settings; it is encrypted at rest with `ENCRYPTION_KEY` and never leaves the backend. |

> **Both fixture keys are wired into `run-inference.yml` and are now required.**
> They were not: the workflow set neither, so every provider skipped itself and
> each scheduled run discovered zero European fixtures without saying so. With
> `european.required: true` an empty provider chain now raises
> `NoFixtureProvidersError` instead. **Add both as GitHub Actions secrets
> (Step 5) before the next scheduled run**, or set `european.required: false`
> to go back to skipping quietly.

#### European fixture discovery

openfootball supplies *historical* results, not upcoming fixtures, so those come
from an API. Every claim below was verified against the live services with real
keys in August 2026 — the published tiers do not tell the whole story.

| Service | Env var | Free tier | Serves upcoming CL/EL/UECL? |
|---|---|---|---|
| [The Odds API](https://the-odds-api.com/) | `THE_ODDS_API_KEY` | 500 credits/month; **`/events` costs 0** | ✅ **Yes** — the primary source |
| [football-data.org](https://www.football-data.org/) | `FOOTBALL_DATA_API_KEY` | 10 req/min | ⏳ CL only, and its schedule lags a season |
| [API-Football](https://www.api-football.com/) | `API_FOOTBALL_API_KEY` | 100 req/day | ❌ **No** — free plan is capped at seasons 2022–2024 |

Two findings worth knowing before you rely on any of them:

- **API-Football cannot serve upcoming fixtures on the free plan at all.** It
  answers `HTTP 200` with `results: 0` and the real reason buried in
  `body["errors"]`: *"Free plans do not have access to this season, try from
  2022 to 2024."* A status-code-only check reads that as "no matches scheduled".
  It is therefore not configured as a fixture provider — it is used only by
  `scripts/verify_corpus_against_api_football.py`, over the seasons it does
  cover.
- **football-data.org is dormant, not broken.** Its free tier covers the
  Champions League (Europa returns 403, Conference 404) but was still serving
  the completed 2025-26 season, so `status=SCHEDULED` returned nothing. It
  should start answering once 2026-27 loads, which is why it stays configured.

The chain returns the first non-empty answer, so provider order is a correctness
property rather than a preference — it lives in `european.provider_order`.

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

-- Canonical team-name aliases (admin-validated mapping for scraped names).
-- FlashScore spells teams its own way ("Sporting CP" vs football-data's
-- "Sp Lisbon"). The inference pipeline inserts unrecognised names as 'pending';
-- an admin confirms each one in the Admin page, which flips it to 'approved'.
-- Only 'approved' rows are ever used to resolve a name.
CREATE TABLE public.team_aliases (
  league_code    TEXT NOT NULL,
  raw_name       TEXT NOT NULL,
  canonical_name TEXT,
  status         TEXT NOT NULL DEFAULT 'pending'
                 CHECK (status IN ('pending', 'approved')),
  approved_by    TEXT,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (league_code, raw_name)
);
ALTER TABLE public.team_aliases ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_role_full_access" ON public.team_aliases
  FOR ALL TO service_role USING (true) WITH CHECK (true);

-- Grant yourself admin access (run after your first login)
INSERT INTO public.user_profiles (user_id, email, approved, is_admin)
SELECT id, email, true, true
FROM auth.users
WHERE email = 'your-email@example.com'
ON CONFLICT (user_id) DO UPDATE SET approved = true, is_admin = true;

-- Pre-computed predictions (written by inference job, read by backend)
CREATE TABLE IF NOT EXISTS public.predictions (
  id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  league_code TEXT        NOT NULL,
  match_date  TEXT        NOT NULL,
  payload     JSONB       NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (league_code, match_date)
);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON public.predictions (match_date);
ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_role_full_access" ON public.predictions
  FOR ALL TO service_role USING (true) WITH CHECK (true);
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

### Step 2b — HuggingFace Space (on-demand inference)

1. Create a **Docker Space** at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Note its repo ID (e.g. `username/football-prediction`)
3. The `deploy-hf-space.yml` workflow will sync `hf_space/` automatically on every push to `main`

Set the following **Space secrets** (Space → Settings → Variables and secrets):

| Secret | Description |
|--------|-------------|
| `HF_REPO_ID` | Your model repo ID (same as above) |
| `HF_TOKEN` | HuggingFace access token |

The Space downloads the model at startup and keeps it warm in memory — no per-request download.

---

### Step 3 — Backend (Render Web Service)

Create a **Web Service** on Render with runtime **Docker** pointing to the repo root.

| Environment Variable | Description | How to obtain |
|----------------------|-------------|---------------|
| `SUPABASE_URL` | Supabase project URL | Supabase → Settings → API |
| `SUPABASE_SERVICE_KEY` | Service role secret | Supabase → Settings → API |
| `ENCRYPTION_KEY` | Fernet key for encrypting API keys at rest | `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `RETRAIN_API_KEY` | Secret for the retraining webhook | `openssl rand -hex 32` |
| `HF_SPACE_URL` | Public URL of your HuggingFace Space | Space → `https://username-space-name.hf.space` |

Set the **Health check path** to `/api/health` — it is unauthenticated and returns 200 as soon as the app is up. Do not use an authenticated route such as `/api/leagues`: it returns 401 to Render, so the health check never passes.

The backend is lightweight — it does **not** load any ML model or historical data. It reads pre-computed predictions from the Supabase `predictions` table and serves them via the API. Startup is near-instant and memory usage stays under 50 MB.

**No additional environment variables are needed for the European competitions.**
The admin alias-review screen reuses `SUPABASE_URL` / `SUPABASE_SERVICE_KEY`, and
the openfootball corpus never reaches Render — it is built offline and shipped to
Hugging Face inside `datasets/european.parquet`.

Two files the image needs are committed to the repo rather than generated at
build time, because the backend container installs only the `backend` dependency
group and has no pandas to rebuild them with:

| File | Purpose |
|---|---|
| `src/backend/data/teams_registry.json` | Current-season teams — fixture pickers, `/teams` |
| `src/backend/data/teams_registry_historical.json` | Every cached season — alias resolution |

Both live under `src/backend/`, which the `Dockerfile` copies wholesale. Regenerate
and **commit** them whenever new season data lands (see
[docs/european-competitions.md](docs/european-competitions.md) §5).

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
| `HF_REPO_ID` | HF model repo ID (`user/repo`) |
| `HF_SPACE_REPO_ID` | HF Space repo ID (`user/space-name`) — used by `deploy-hf-space.yml` and by the post-retrain Space restart |
| `SUPABASE_URL` | Supabase project URL (for inference job) |
| `SUPABASE_SERVICE_KEY` | Service role secret (for inference job) |
| `RENDER_DEPLOY_HOOK_URL` | Render deploy hook (backend service → Settings → Deploy Hook) |
| `RENDER_BACKEND_URL` | Backend public URL |
| `RETRAIN_API_KEY` | Same value as the backend env var |
| `THE_ODDS_API_KEY` | **Required.** Enables the Odds API fallback for domestic fixtures **and** is the primary source of upcoming European fixtures. Absent, with no other provider keyed → the inference job fails under `european.required`. |
| `FOOTBALL_DATA_API_KEY` | Champions League fixture fallback. Dormant until football-data.org loads the current season, but it counts as a provider: with this set and the Odds API key absent, the run proceeds. |

#### Retrain workflow (`.github/workflows/retrain.yml`)
Runs **every Monday at 06:00 UTC** (or manually via *Run workflow*, with an optional `force` toggle):
1. Sets the retraining banner flag → users see the amber banner
2. Downloads fresh data from football-data.co.uk
3. Retrains the model (ensemble + calibration + Dixon-Coles) across all seasons — **only when enough new data has accumulated** (see the two signals below; `--force`/the `force` toggle overrides)
4. Uploads Parquet datasets, plus model artefacts when there are new ones
5. Restarts the Hugging Face Space (`scripts/restart_hf_space.py`)
6. Triggers the Render redeploy — **only when the model was refit**
7. Clears the retraining banner flag (always)

> **Why step 5 exists:** the Space builds its match history once, in `lifespan`, from the Parquet snapshot on the Hub. A running Space therefore keeps answering `/team-insights` from the frame it booted with, so uploading newer results changes nothing until it reboots — team pages silently freeze at the last restart. The step fails the job on error rather than warning, because silent staleness is the failure mode it exists to prevent.

#### Two independent signals

`train_model.py` emits two GitHub Actions step outputs, and conflating them is what once froze the deployment for three months:

| Signal | Means | Gates |
|--------|-------|-------|
| `data_changed` | The loaded data holds matches the model has not seen | Dataset upload (`--datasets-only`) + Space restart |
| `retrained` | The model was actually refit | Artefact upload + Render redeploy |

They disagree whenever a refit is held back, and on those weeks the **data still ships** — otherwise the Space keeps serving a stale snapshot while the workflow reports success.

The refit itself is governed by `retrain_check.min_new_matches` (default **800**). Time-decay weighting makes a single round of matches negligible against a ~64,000-row corpus, so refitting on one buys a redeploy and a Space restart to move probabilities in the third decimal. One weekly round across the configured divisions is ~190 matches, so 800 refits roughly monthly. Set it to `0` to refit on any new data, or `retrain_check.enabled: false` to refit only via `--force`.

> **Reading `training_results.json` after the league expansion:** the headline accuracy and log-loss come from the model's own held-out slice, which now includes lower divisions that are inherently harder to predict. Those numbers dropped (≈0.53 → 0.49 accuracy) purely because the test population changed — it is not a regression in the original six leagues, where the measured effect was nil.

#### Inference workflow (`.github/workflows/run-inference.yml`)
Runs **daily at 06:30 UTC**, after retraining completes, or manually:
1. Downloads ML model from Hugging Face
2. Fetches today's fixtures from football-data.co.uk
3. Runs the full ML pipeline (feature engineering → ELO → ensemble prediction → Poisson stats → value detection)
4. Uploads pre-computed predictions to the Supabase `predictions` table

---

### Deployment overview

```
GitHub Actions — Retrain (every Monday 06:00 UTC)
  └─► retrains model → uploads to Hugging Face model repo
  └─► triggers inference workflow

GitHub Actions — Inference (daily 06:30 UTC + after retrain)
  └─► downloads model from Hugging Face
  └─► runs ML pipeline for all leagues
  └─► uploads predictions JSONB to Supabase predictions table

GitHub Actions — Deploy HF Space (on push to main, hf_space/** changed)
  └─► pushes hf_space/ to HuggingFace Space via huggingface_hub

HuggingFace Space (Docker — on-demand inference)
  └─► loads model from HF model repo at startup (held in memory)
  └─► POST /infer: fetches fixtures → features → ensemble → predictions
  └─► called by backend InferenceService when user clicks "Calcular ao vivo"

Render (Backend — lightweight, ~50 MB)
  └─► Docker image with backend deps only (no pandas/sklearn/xgboost)
  └─► reads predictions from Supabase (no ML at runtime)
  └─► proxies on-demand requests to HF Space via InferenceService
  └─► serves auth, admin, AI analysis, exports, API keys (Gemini + NVIDIA)
  └─► uses ENCRYPTION_KEY to decrypt per-user API keys

Render (Frontend)
  └─► static build from src/frontend
  └─► calls backend via VITE_API_URL
  └─► Supabase auth via VITE_SUPABASE_URL + VITE_SUPABASE_ANON_KEY

Supabase
  └─► predictions   (pre-computed ML predictions per league+date)
  └─► user_profiles  (approved, is_admin)
  └─► user_api_keys  (encrypted Gemini + NVIDIA keys, keyed by user_id+service)
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
# On-demand inference — set to your HF Space URL (or omit to disable the feature)
HF_SPACE_URL=https://username-space-name.hf.space
# Optional — used only by offline inference scripts
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

### Local login bypass (optional)

To skip Supabase login while developing locally, enable the dev auth bypass on
**both** sides. It uses a synthetic dev user (no Supabase account required):

Backend — in `.env`:

```
DEV_AUTH_BYPASS=true
# optional overrides
DEV_USER_ID=dev-user
DEV_USER_EMAIL=dev@localhost
```

Frontend — in `src/frontend/.env.local`:

```
VITE_DEV_AUTH_BYPASS=true
# optional overrides (must match the backend values)
VITE_DEV_USER_ID=dev-user
VITE_DEV_USER_EMAIL=dev@localhost
```

> ⚠️ This is **local-only**. The frontend gate is compiled out of production
> builds (`import.meta.env.DEV`), and the backend flag must never be set in a
> deployed environment. When enabled, the backend logs a warning at startup.

### ML Pipeline (standalone)

```bash
# Train the model (ensemble + calibration + Dixon-Coles Poisson; downloads
# data, writes seasons_trained to training_results.json). Only retrains when
# there is new data unless --force is passed.
uv run python scripts/train_model.py

# --- Offline, report-only tuning (never mutate config automatically) ---
# Bounded, leakage-safe (TimeSeriesSplit) XGBoost log-loss search
uv run python scripts/tune_xgboost.py
# Sweep the ensemble/Poisson blend weight by held-out log-loss
uv run python scripts/tune_blend_weight.py
# Sweep the European Dixon-Coles/ELO blend, rolling-origin over recent seasons
uv run python scripts/tune_european_weight.py

# Upload model + Parquet datasets to Hugging Face
HF_TOKEN=hf_... HF_REPO_ID=user/repo uv run python scripts/upload_to_hf.py

# Publish only the datasets, leaving the model artefacts alone
HF_TOKEN=hf_... HF_REPO_ID=user/repo uv run python scripts/upload_to_hf.py --datasets-only

# Restart the Space so it reloads the uploaded snapshot (--dry-run to preview)
HF_TOKEN=hf_... HF_SPACE_REPO_ID=user/space uv run python scripts/restart_hf_space.py

# Run inference and upload predictions to Supabase
SUPABASE_URL=... SUPABASE_SERVICE_KEY=... HF_TOKEN=... HF_REPO_ID=... \
  uv run python scripts/run_inference.py

# Value bets pipeline (local HTML report)
uv run python scripts/find_value_bets.py --league "Liga Portugal"

# Single match prediction
uv run python scripts/predict_match.py --home "Benfica" --away "Porto" --league P1
```

### European club competitions

Champions / Europa / Conference League. These exist to **calibrate ratings
across leagues**, not merely to add fixtures: each domestic league is a
near-closed pool, so without real cross-league matches its ELO ratings are not
comparable with any other league's.

```bash
# Team registries — the historical one is what alias resolution matches against
uv run python scripts/build_teams_registry.py
uv run python scripts/build_teams_registry.py --all-seasons

# Fetch and parse the openfootball corpus into datasets/european/
uv run python scripts/build_european_corpus.py

# Reconcile openfootball names with the canonical registry, then review them
uv run python scripts/queue_european_team_names.py --push
uv run python scripts/review_team_names.py          # http://127.0.0.1:8765

# Export the approvals into the committed seed, then commit it. CI is given no
# Supabase credentials, so this is the only route by which approved aliases
# reach training — skip it and the calibration silently does nothing there.
uv run python scripts/export_team_aliases.py
```

Name reconciliation is a prerequisite for the calibration to have any effect —
ELO is keyed by the team-name string, so an unresolved name creates a separate
rating and the pools never link.

Full documentation: **[docs/european-competitions.md](docs/european-competitions.md)**

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
