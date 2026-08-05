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
- `GET /teams` — team names grouped by league code
- `POST /infer` — run predictions for a given date and league selection
- `POST /predict-custom` — one-off prediction for any two teams
- `POST /match-insights` — head-to-head history and both team profiles
- `GET /team-insights` — full statistical profile of a single team

### POST /infer

```json
{
  "date": "28/04/2025",
  "league_codes": ["E0", "SP1"]
}
```

Both fields are optional. `date` defaults to today. `league_codes` defaults to all supported leagues.

### POST /match-insights

```json
{
  "home_team": "Sporting",
  "away_team": "Porto",
  "league_code": "P1"
}
```

Returns the head-to-head record and meeting list, a full profile for each team
and — when the Dixon-Coles model knows both teams — the goal markets (xG,
over/under, BTTS, most likely scorelines). Unknown teams yield `404`.

### GET /team-insights

`/team-insights?team=Sporting&league_code=P1`

Returns all-time and home/away win-draw-loss records, goal averages, recent
form, market rates (clean sheets, BTTS, over 2.5), per-match averages (shots,
corners, cards) and the recent results list. Unknown teams yield `404`.

Reporting horizons come from the `insights` block of `config/config.yaml`.

## Environment variables

| Variable | Description |
|---|---|
| `HF_REPO_ID` | HuggingFace model repo (e.g. `user/football-model`) |
| `HF_TOKEN` | HuggingFace access token (required for private repos) |
| `HF_LOCAL_DIR` | Local path for downloaded model files (default: `/tmp/hf_models`) |
