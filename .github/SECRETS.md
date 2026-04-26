# Secrets Reference

## GitHub Repository Secrets

Go to: **Repository → Settings → Secrets and variables → Actions → New repository secret**

---

### `VITE_API_URL`

**Used by:** `deploy-frontend.yml` (baked into frontend build)

**What it is:** The public URL of your Render backend service.

**How to get it:** Render → your backend service → top of the page shows the URL (e.g. `https://football-prediction-xxxx.onrender.com`).

---

### `VITE_SUPABASE_URL`

**Used by:** `deploy-frontend.yml` (baked into frontend build)

**What it is:** Your Supabase project URL.

**How to get it:** Supabase dashboard → your project → Settings → API → **Project URL**.

---

### `VITE_SUPABASE_ANON_KEY`

**Used by:** `deploy-frontend.yml` (baked into frontend build)

**What it is:** The Supabase anon/public key (safe to use in the browser).

**How to get it:** Supabase dashboard → your project → Settings → API → **anon public** key.

---

### `RENDER_FRONTEND_DEPLOY_HOOK_URL`

**Used by:** `deploy-frontend.yml` (triggers Render to redeploy the frontend after a new image is pushed)

**What it is:** A unique Render webhook URL that redeploys the frontend service when POSTed to.

**How to get it:** Render → your **frontend** service → Settings → scroll to **Deploy Hook** → copy the URL.

---

### `RENDER_DEPLOY_HOOK_URL`

**Used by:** `retrain.yml` (triggers Render to redeploy the backend after a new model is trained and a new image is pushed)

**What it is:** A unique Render webhook URL that redeploys the backend service when POSTed to.

**How to get it:** Render → your **backend** service → Settings → scroll to **Deploy Hook** → copy the URL.

---

### `RENDER_BACKEND_URL`

**Used by:** `retrain.yml` (signals the backend to set/clear the retraining flag during model training)

**What it is:** The public URL of your Render backend service (same value as `VITE_API_URL`).

**How to get it:** Render → your backend service → top of the page shows the URL (e.g. `https://football-prediction-xxxx.onrender.com`).

---

### `RETRAIN_API_KEY`

**Used by:** `retrain.yml` (authenticates requests to `/api/retrain-status`)

**What it is:** A random secret shared between GitHub Actions and the backend to protect the retrain endpoint.

**How to get it:** Generate a new one locally:
```bash
openssl rand -hex 32
```
Use the same value here **and** in Render's backend environment variables (see below).

---

### `HF_TOKEN`

**Used by:** `retrain.yml` (uploads the trained model to Hugging Face Hub)

**What it is:** A Hugging Face personal access token with write permissions to your model repo.

**How to get it:** [huggingface.co](https://huggingface.co) → your profile → Settings → Access Tokens → **New token** (role: Write).

---

### `HF_REPO_ID`

**Used by:** `retrain.yml` (identifies which Hugging Face repo to upload the model to)

**What it is:** The Hugging Face repo ID in the format `username/repo-name`.

**How to get it:** The repo ID is shown on your Hugging Face model repo page (e.g. `lourenco/football-prediction-model`).

---

## Render — Backend Service Environment Variables

Go to: **Render → your backend service → Environment**

| Variable | Required | Value | How to get it |
|----------|----------|-------|---------------|
| `SUPABASE_URL` | ✅ Yes | Your Supabase project URL | Supabase → project → Settings → API → Project URL |
| `SUPABASE_SERVICE_KEY` | ✅ Yes | Supabase service role key | Supabase → project → Settings → API → **service_role** key (keep secret) |
| `ENCRYPTION_KEY` | ✅ Yes | A Fernet symmetric key — backend crashes without it | Generate with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `HF_TOKEN` | ✅ Yes | Same as GitHub secret `HF_TOKEN` | See above |
| `HF_REPO_ID` | ✅ Yes | Same as GitHub secret `HF_REPO_ID` | See above |
| `RETRAIN_API_KEY` | ✅ Yes | Same as GitHub secret `RETRAIN_API_KEY` | Generate with `openssl rand -hex 32` |
| `HF_LOCAL_DIR` | ❌ Optional | Local path to cache downloaded HF model | Defaults to `/tmp/hf_models` — only set if you need a different path |

---

## Render — Frontend Service Environment Variables

**None required.** All `VITE_*` variables are baked into the static JS bundle at build time via GitHub Actions. Render only serves the pre-built files.
