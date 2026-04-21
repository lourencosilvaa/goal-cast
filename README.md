# Football Prediction Agent

A GitHub Copilot Agent that combines ML ensemble predictions with real-time contextual intelligence (injuries, team news, suspensions, lineups) to produce adjusted probabilities and identify value betting opportunities across Europe's top leagues.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              LAYER 1 — DATA INGESTION                    │
│  football-data.co.uk │ football-data.org API             │
│  5 seasons × 6 leagues │ 24h cache TTL                   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│           LAYER 2 — FEATURE ENGINEERING                  │
│  Rolling stats (5-match) │ ELO ratings (K=32)            │
│  xG proxy │ Fatigue │ H2H │ Draw features               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│            LAYER 3 — ML ENSEMBLE MODEL                   │
│  Logistic Regression │ Random Forest │ XGBoost           │
│  Soft Voting [1:1:2] │ TimeSeriesSplit CV (5-fold)       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│          LAYER 4 — ODDS & MATCH STATISTICS               │
│  B365 odds (football-data.co.uk fixtures CSV)            │
│  Poisson xG │ Over/Under │ BTTS │ Top scorelines         │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              LAYER 5 — ANALYSIS                          │
│  Value bet detection (edge ≥ 3%)                         │
│  KL divergence │ Kelly Criterion (capped 25%)            │
│  Probability blending (50% ML / 30% BK / 20% best)      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│           LAYER 6 — COPILOT AGENT                        │
│  Context research (injuries, suspensions, lineups)       │
│  Probability adjustment │ HTML report (PT-PT)            │
│  Natural-language insights │ Value bet recommendations   │
└─────────────────────────────────────────────────────────┘
```

---

## Pipeline — Layer by Layer

### Layer 1: Data Ingestion

Historical match data is loaded from [football-data.co.uk](https://www.football-data.co.uk/) covering **5 seasons** across **6 leagues**:

| League | Code | Seasons |
|--------|------|---------|
| Premier League | `E0` | 2020/21 → 2024/25 |
| La Liga | `SP1` | 2020/21 → 2024/25 |
| Bundesliga | `D1` | 2020/21 → 2024/25 |
| Serie A | `I1` | 2020/21 → 2024/25 |
| Ligue 1 | `F1` | 2020/21 → 2024/25 |
| Liga Portugal | `P1` | 2020/21 → 2024/25 |

Data is cached locally in `datasets/cache/` with a **24-hour TTL** to avoid redundant downloads.

**Columns retained**: Date, HomeTeam, AwayTeam, full-time & half-time goals/results, shots/shots on target, fouls, corners, cards, and Bet365 odds (B365H/D/A).

A secondary loader fetches data from the [football-data.org API](https://api.football-data.org/v4) for cup competitions (Champions League, Europa League, Conference League, FA Cup, Copa del Rey), also cached with 24h TTL.

### Layer 2: Feature Engineering

Raw match data is transformed into predictive features through the following pipeline:

#### Rolling Statistics (5-match window)
Per-team rolling averages computed over the last 5 matches:
- Goals for / Goals against
- Shots / Shots on target
- Corners / Fouls
- Points per game (form)

#### ELO Rating System
FIFA-style rating using the formula: `R_new = R_old + K × M × (S - E)`

| Parameter | Value |
|-----------|-------|
| K-Factor | `32` |
| Home Advantage | `+65` ELO points |
| Initial Rating | `1500` |
| Margin Multiplier | `log(goal_diff + 1)` |
| Expected Score | `E = 1 / (1 + 10^((R_opp - R_team) / 400))` |

Features generated: `elo_home`, `elo_away`, `elo_diff`, `elo_expected_home`, `elo_expected_away`.

#### xG Proxy (No Same-Match Leakage)
Rolling expected goals built from historical conversion rates:
- Shots on target → xG: **30%** conversion rate
- Regular shots → xG: **3%** conversion rate

Features: `home_xG_rolling`, `away_xG_rolling`, `home_xGA_rolling`, `away_xGA_rolling`, `xG_diff`.

#### Fatigue Features
- Rest days between matches (capped at 30 days, default 14 for first match)
- Fatigue flag: `≤3 days` rest
- Midweek flag: Tuesday (1) / Wednesday (2)

#### Head-to-Head Features (last 5 meetings)
- `h2h_home_wins` — Win fraction from the home team's perspective
- `h2h_draws` — Draw fraction
- `h2h_total_goals_avg` — Average total goals

#### Draw-Specific Features
- `home_draw_pct`, `away_draw_pct`, `avg_draw_pct` — Historical draw rates
- `form_gap` — Absolute form difference (similar form → more draws)
- `attack_similarity` — `1 / (1 + |GF_diff|)`
- `defense_similarity` — `1 / (1 + |GA_diff|)`
- `combined_defensive` — `1/(1+home_GA) + 1/(1+away_GA)` (both teams defensive → draws)

### Layer 3: ML Ensemble Model

Four classifiers are trained and combined into a soft-voting ensemble:

| Model | Key Hyperparameters |
|-------|-------------------|
| **Logistic Regression** | `C=0.5`, `max_iter=1000`, `class_weight="balanced"` |
| **Random Forest** | `n_estimators=200`, `max_depth=8`, `min_samples_leaf=10`, `class_weight="balanced"` |
| **XGBoost** | `n_estimators=200`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8` |
| **Gradient Boosting** | `n_estimators=150`, `max_depth=4`, `learning_rate=0.08` |

**Ensemble**: Soft voting with weights **[1, 1, 2]** (LR, RF, XGB) — XGBoost gets double weight due to superior individual performance. Gradient Boosting is trained for comparison but not included in the final ensemble.

**Cross-validation**: `TimeSeriesSplit` with **5 folds** to prevent future data leakage. Metrics reported: accuracy and log loss (mean ± std). Data is scaled with `StandardScaler` and split 80/20 for train/test.

**Feature exclusions**: Odds-derived features (`norm_prob_*`, `odds_prob_*`, `odds_spread`) and same-match xG features (`home_xG_proxy`, `away_xG_proxy`, `home_xG_overperf`, `away_xG_overperf`) are excluded to prevent the model from simply copying bookmaker odds.

**Output**: 3-class probabilities — Away Win (0), Draw (1), Home Win (2). The predicted outcome is the class with the highest probability; confidence equals that maximum probability.

### Layer 4: Odds & Match Statistics

#### Odds
Bet365 odds are fetched from football-data.co.uk's fixtures CSV and converted to implied probabilities. These serve as the market baseline to compare against ML predictions.

#### Poisson Match Statistics
Match-level statistics are computed using a **Poisson distribution** model:

- **xG (Expected Goals)**: `home_attack × away_defense / league_avg` (clamped to 0.3–4.0 for home, 0.2–3.5 for away)
- **Over/Under**: `P(Over N) = 1 - P(≤N)` from Poisson CDF
- **BTTS**: `1 - P(home scores 0) - P(away scores 0) + P(0-0)`
- **Top scorelines**: Top 8 most probable scorelines ranked by `P(h) × P(a)` from independent Poisson distributions

Team stats are computed from the most recent **10 matches** per team (goals scored/conceded, shots, corners, clean sheet %, BTTS %, over 2.5 %, form points).

### Layer 5: Analysis

#### Value Bet Detection
A value bet exists when:
```
ML_Probability > Bookmaker_Implied_Probability + min_edge
```

| Threshold | Value |
|-----------|-------|
| Minimum edge to report | **3%** (`0.03`) |
| Value divergence threshold | **5%** (`0.05`) |

#### Kelly Criterion
Optimal bet sizing: `f = (b × p - q) / b` where `b = odds - 1`, `p = ML probability`, `q = 1 - p`. **Capped at 25%** of bankroll.

#### Confidence Levels

| Level | Criteria |
|-------|----------|
| **HIGH / ALTA** | edge ≥ 10% AND probability ≥ 50% |
| **MEDIUM / MÉDIA** | edge ≥ 5% AND probability ≥ 35% |
| **LOW / BAIXA** | Everything else |

#### KL Divergence
Measures information-theoretic distance between ML and bookmaker distributions:
```
KL(ML ‖ BK) = Σ ML_prob × ln(ML_prob / BK_prob)
```

Additional divergence features: signed/absolute divergence per outcome, max divergence, sources agreement (whether ML and bookmaker agree on the favourite).

#### Probability Blending
Final blended probabilities combine multiple signals:
- **50%** ML model predictions
- **30%** bookmaker average implied probabilities
- **20%** best available odds (from aggregator)

### Layer 6: Copilot Agent (Context & Reporting)

The Copilot Agent adds a reasoning layer on top of the statistical pipeline:

1. **Context research** — Searches the web for injuries, suspensions, lineup news, managerial changes, and motivation factors from official club sites, Transfermarkt, and national/international sports press
2. **Probability adjustment** — Applies contextual adjustments to ML base probabilities (±1–15% depending on severity)
3. **HTML report generation** — Produces a full report in **Português de Portugal (PT-PT)** with match cards, value bets, Poisson stats, and NLP-generated analysis
4. **Value bet re-evaluation** — Context may invalidate ML-flagged value bets (e.g., star player injured) or reveal new ones the model missed

---

## Quick Start

```bash
# Install dependencies
uv sync

# Install Playwright browsers (for scraping)
uv run playwright install chromium

# Train the model (from football-data.co.uk)
uv run python scripts/train_model.py

# Or train from local data
uv run python scripts/train_model.py --local-data datasets/joined_data.csv

# Scrape current odds
uv run python scripts/scrape_odds.py --league "Liga Portugal"

# Find value bets (ML + odds)
uv run python scripts/find_value_bets.py --league "Liga Portugal"

# Run tests
uv run pytest tests/ -v
```

## Copilot Agent Usage

When using GitHub Copilot in this project, the agent is configured via `.github/copilot-instructions.md` to act as a Football Prediction Analyst. You can ask it things like:

- "What are today's value bets for Liga Portugal?"
- "Train the model and show me the results"
- "Scrape odds from all Portuguese betting sites"
- "Predict Arsenal vs Chelsea"
- "Compare odds across Betclic, Betano, and Solverde"

The agent will run the appropriate CLI scripts and interpret the results.

## Project Structure

```
football-prediction-agent/
├── .github/
│   └── copilot-instructions.md   # Copilot agent brain
├── config/
│   ├── config.yaml               # Centralized config
│   └── config_loader.py          # Pydantic config loader
├── src/
│   ├── models/                   # ML pipeline
│   │   ├── data_loader.py        # Historical data (football-data.co.uk)
│   │   ├── football_data_org_loader.py  # Cup data (football-data.org API)
│   │   ├── data_cleaner.py       # Data cleaning
│   │   ├── feature_engineer.py   # Feature engineering (rolling, ELO, xG, H2H)
│   │   ├── elo.py                # ELO rating system (K=32)
│   │   ├── trainer.py            # Model training (4 classifiers + ensemble)
│   │   └── predictor.py          # Match prediction (3-class proba)
│   ├── scrapers/                 # Odds sources
│   │   ├── base_scraper.py       # Abstract base
│   │   ├── betclic_scraper.py    # Betclic.pt
│   │   ├── betano_scraper.py     # Betano.pt
│   │   ├── solverde_scraper.py   # Solverde.pt
│   │   ├── fixtures_fetcher.py   # B365 odds from fixtures CSV
│   │   └── odds_aggregator.py    # Multi-source aggregator
│   ├── analysis/                 # Analysis tools
│   │   ├── value_detector.py     # Value bet detection (edge ≥ 3%)
│   │   ├── divergence.py         # KL divergence & probability blending
│   │   ├── match_stats.py        # Poisson model (xG, O/U, BTTS, scorelines)
│   │   └── report_generator.py   # Report generation
│   └── visualization/            # Output
│       ├── html_report.py        # HTML report generator (PT-PT)
│       └── plots.py              # Matplotlib charts
├── scripts/                      # CLI entry points
│   ├── train_model.py            # Train ML ensemble
│   ├── scrape_odds.py            # Fetch B365 odds from fixtures CSV
│   ├── predict_match.py          # Predict a single match
│   └── find_value_bets.py        # Full pipeline: predict + stats + value + HTML
├── tests/                        # Test suite (mirrors src/)
├── datasets/                     # Historical data + cache
├── output/                       # Reports (JSON + HTML), trained models
└── pyproject.toml
```

## Key Concepts

### Value Bet Detection
A value bet exists when the ML model estimates a higher probability for an outcome than what the bookmaker odds imply. The **edge** = ML probability − bookmaker implied probability. Minimum edge to report: **3%**. Divergence threshold: **5%**.

### Kelly Criterion
Optimal bet sizing formula: `f = (b×p - q) / b` where `b = odds - 1`, `p = ML probability`, `q = 1 - p`. Capped at **25%** of bankroll to limit risk.

### Divergence Analysis
Uses KL-divergence to measure how much the ML model and bookmakers disagree. Higher divergence = potential opportunity (or model error). Blended probabilities combine **50% ML**, **30% bookmaker average**, and **20% best odds**.

### ELO Ratings
FIFA-style rating system with K-factor of **32** and a home advantage of **+65 ELO points**. Teams start at **1500**, gain/lose points based on results adjusted for opponent strength and margin of victory (`log(goal_diff + 1)`).
