---
title: Football Prediction API
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Football Prediction API

FastAPI inference service for on-demand football match outcome predictions.

## Endpoints

- `GET /health` — liveness check
- `POST /infer` — run predictions for a given date and league selection

### POST /infer

```json
{
  "date": "28/04/2025",
  "league_codes": ["E0", "SP1"]
}
```

Both fields are optional. `date` defaults to today. `league_codes` defaults to all supported leagues.

## Environment variables

| Variable | Description |
|---|---|
| `HF_REPO_ID` | HuggingFace model repo (e.g. `user/football-model`) |
| `HF_TOKEN` | HuggingFace access token (required for private repos) |
| `HF_LOCAL_DIR` | Local path for downloaded model files (default: `/tmp/hf_models`) |
