# European Club Competitions

Support for the Champions League, Europa League and Conference League.

These competitions have no football-data.co.uk feed, so their results come from
the public-domain [openfootball](https://github.com/openfootball/champions-league)
project. They exist in this project for one reason, and it is not fixture
coverage — it is **rating calibration**.

---

## 1. Why this exists

Each domestic league is a near-closed pool: teams only ever play opponents from
their own division. ELO is zero-sum within a pool, so every league's ratings
drift around the same starting value regardless of how strong the league really
is. Measured on the real corpus, five leagues have a mean ELO of *exactly*
1500.0 — Liga Portugal, Eredivisie, Süper Lig, Jupiler Pro League and Super
League Greece. Not approximately; the arithmetic pins them, because those
countries have only one tier in `data.leagues` and their pool is perfectly
sealed.

The consequence is concrete. Before calibration the best-rated side in each
league looked like this:

| League | Team | ELO |
|---|---|---|
| Bundesliga | Bayern Munich | 1985 |
| Premier League | Arsenal | 1930 |
| Scottish Premiership | Celtic | 1925 |
| Liga Portugal | Sp Lisbon | 1906 |
| Super League Greece | Olympiakos | 1840 |

Celtic and Arsenal within five points of each other implies a 49.2% coin flip
between them. That is not a judgement the model formed from evidence; there is
no evidence either way in domestic-only data.

Dixon-Coles is worse than miscalibrated: cross-league attack and defence
strengths are **unidentified**. With no match connecting the pools there is no
unique answer to fit, so the optimiser settles on an arbitrary one. More
domestic data can never fix it.

**The fix is not speculative.** The same mechanism already works inside
England, where promotion and relegation move teams between tiers every year:

| | mean ELO |
|---|---|
| Premier League | 1719 |
| Championship | 1542 |
| League One | 1480 |
| League Two | 1359 |

A clean 360-point gradient, because those crossings carry rating information
with them. A Champions League tie does exactly that between Portugal and
England.

---

## 2. How the data flows

```
openfootball repo  ──build_european_corpus──▶  datasets/european/*.csv
                                                        │
                          queue_european_team_names ─────┤
                                     │                   │
                              Supabase queue             │
                                     │                   │
                             review_team_names           │
                                     │                   │
                            approved aliases  ───────────┤
                                                         ▼
                                              training / calibration
```

Two rules hold throughout:

- **The network is never in the training path.** Fetching and parsing is an
  explicit CLI step that writes a CSV cache; training reads only that cache. A
  GitHub outage can never silently produce a model without cross-league links.
- **Nothing is matched automatically on similarity.** A name resolves exactly,
  resolves through a human-approved alias, or stays unresolved. Suggestions are
  proposals only.

---

## 3. Building the corpus

```bash
uv run python scripts/build_european_corpus.py
```

Shallow-clones openfootball (~600 KB) into `datasets/openfootball/`, parses
every configured competition-season, and writes two caches:

| File | Contents |
|---|---|
| `datasets/european/results.csv` | Main draws — 2,636 matches |
| `datasets/european/results_with_qualifiers.csv` | Plus qualifiers — 3,490 matches |

Both are gitignored: they are caches, regenerable from the CLI.

Useful flags: `--no-sync` (parse the existing checkout offline), `--seasons
2526,2425`, `--dry-run`.

### Upstream coverage is uneven

| Competition | Seasons available |
|---|---|
| Champions League | 2011-12 → 2025-26 |
| Europa League | 2020-21 onwards |
| Conference League | 2021-22 onwards |
| Qualifiers | 2024-25 onwards |

Champions League is therefore the backbone of the linkage; the other two
reinforce the recent half. A configured season with no upstream file is skipped
silently, because a missing competition-season is the normal case, not an error.

### Parsing traps

The `football.txt` format has three conventions that silently corrupt data if
misread. All are handled and regression-tested:

1. **Scorelines put the decisive number first.** `3-4 pen. 1-1 a.e.t. (1-1, 0-0)`
   is a penalty shootout — the bracket holds `(90 minutes, half time)`. Reading
   the leading pair would record the 2012 final as Bayern 3-4 Chelsea rather
   than the 1-1 draw it was. The corpus keeps the **90-minute** score, matching
   football-data's `FTHG`/`FTAG` semantics.
2. **The calendar restarts per group.** openfootball lists each group in full
   before starting the next, so a bare `Wed Sep 14` regularly follows a
   `Wed Dec 7`. Inferring the year from the previous line walks the 2011-12
   season forward into 2018. Years come from the season, not the previous line.
3. **Awarded and cancelled matches are excluded.** A forfeit 3-0 carries no
   performance signal; feeding it to the ratings would assert that one side
   outplayed the other.

Verification: 4,240 matches parsed + 6 deliberately excluded = 4,246, exactly
the total openfootball declares across all 30 files.

---

## 4. Team name reconciliation

openfootball spells clubs in full (`Sport Lisboa e Benfica`) while every model,
dataset and artefact here is keyed by football-data's short form (`Benfica`).
Only 17 of 382 spellings match outright.

**Canonical keys never change.** Migrating to openfootball's spelling would mean
remapping the whole corpus and retraining — and openfootball is not internally
consistent anyway (`Real Madrid` and `Real Madrid CF`, `SL Benfica` and `Sport
Lisboa e Benfica` both appear). Its names are treated as aliases instead.

### Queue the names

```bash
uv run python scripts/queue_european_team_names.py --push
```

Sorts every distinct spelling into three outcomes:

| Outcome | Count (main draws) | Action |
|---|---|---|
| Matched outright | 15 | None — the resolver recognises these natively |
| Needs review | 99 | Queued for the review tool |
| Untracked country | 104 | None — nothing to match against |

The third bucket is deliberately **not** a backlog. Shakhtar, Slavia Praha,
Salzburg and Bodø/Glimt come from leagues absent from `data.leagues`, so they
have no domestic history to link to. They keep their openfootball names and
build ratings of their own from European play — which is what makes a
Benfica-vs-Shakhtar result informative rather than noise.

### Review them

```bash
uv run python scripts/review_team_names.py     # http://127.0.0.1:8765
```

A local review tool that reads and writes the same Supabase `team_aliases`
table the deployed admin panel uses, so approvals are the real thing and need
no redeploy. Localhost-only, no login, uses the service key from `.env`.

The top suggestion is pre-selected but nothing saves without a click, because
suggestions are wrong often enough to matter: `Sport Lisboa e Benfica` ranks
`Sp Lisbon` — which is **Sporting**, not Benfica — above the correct answer.

### Country scoping

Queued names are stored under a scope like `EU-POR`, not the competition code.
Two reasons:

- **Suggestions stay safe.** Searching all 21 leagues offers `Sparta Rotterdam`
  for `AC Sparta Praha`, a near-miss that would attribute a Czech club's history
  to a Dutch one. Scoping to the club's own country rules it out, and a country
  with no tracked league gets no candidates at all.
- **One approval covers every competition.** A club plays the Champions League
  one year and the Europa League the next; keying the alias to the competition
  would demand approval three times.

The country is encoded in the existing `league_code` column, so no Supabase
migration is needed. `country_leagues` in `config.yaml` maps openfootball's
three-letter codes to league codes — note `MCO: [F1, F2]`, since Monaco is its
own country to openfootball but plays in France.

---

## 5. The two team registries

| File | Built from | Teams | Used by |
|---|---|---|---|
| `teams_registry.json` | Newest season only | 372 | Fixture pickers, `/teams` fallback |
| `teams_registry_historical.json` | Every cached season | 725 | Alias resolution |

```bash
uv run python scripts/build_teams_registry.py                # current season
uv run python scripts/build_teams_registry.py --all-seasons  # historical
```

**Why two.** The default registry answers "who plays this season", which is what
a fixture picker wants. Alias resolution wants the opposite: Vitesse has seven
Eredivisie seasons in the corpus and Sivasspor eight in the Süper Lig, but both
were relegated out of their top division, so neither appears in the newest
season file. Both were unmatchable until the historical registry existed —
despite being *exact* canonical spellings that needed no alias at all.

That gap was structural, not incidental: the openfootball corpus spans nine
seasons while the current registry spans one.

> **Regenerate the historical registry whenever new season data lands.**
> Re-running the name queue after a season rolls over without doing so
> reproduces the Vitesse problem for that season's relegated clubs.

Both files ship inside `src/backend/` so the deployed image carries them without
needing `datasets/` or pandas — commit them when they change.

---

## 6. Configuration

All of it lives under `european:` in `config/config.yaml`:

| Key | Purpose |
|---|---|
| `repo_url`, `checkout_path` | Where openfootball is cloned |
| `competitions` | Competition code → openfootball file stem |
| `qualifier_competitions` | Kept separate so including them stays a measured choice |
| `seasons` | Mirrors `data.seasons`, so time-decay treats both corpora alike |
| `cache_path`, `qualifiers_cache_path` | The two parsed caches |
| `country_leagues` | Country code → league codes; scopes name matching |
| `alias_scope` | Prefix for European alias scopes (`EU`) |

---

## 7. Operational order

Fresh setup, or after new season data:

```bash
# 1. Domestic registries
uv run python scripts/build_teams_registry.py
uv run python scripts/build_teams_registry.py --all-seasons

# 2. European corpus
uv run python scripts/build_european_corpus.py

# 3. Reconcile names
uv run python scripts/queue_european_team_names.py --push
uv run python scripts/review_team_names.py

# 3b. Export approvals to the committed seed — CI cannot reach Supabase, so
#     without this the calibration silently does nothing there (§10)
uv run python scripts/export_team_aliases.py

# 4. Retrain (reads the caches; never touches the network for this data)
uv run python scripts/train_model.py --force

# 5. Confirm the calibration actually worked (report-only)
uv run python scripts/evaluate_calibration.py
```

Step 3 is a prerequisite for step 4 having any effect: ELO is keyed by the
team-name string, so until `Sport Lisboa e Benfica` resolves to `Benfica` the
European result creates a *separate* rating and the pools never link. The
calibration would run cleanly and achieve nothing.

The `team_aliases` table must exist in Supabase — its SQL is in
[README.md](../README.md) under Deployment. Both the queue writer and the sink
swallow write failures by design (right for the fixture pipeline, which meets
the same unknown name every run), so the queue script reads its writes back and
reports what actually landed rather than trusting the call.


---

## 8. Train/serve parity

Training and all three inference paths — `run_inference.py`,
`find_value_bets.py` and the HF Space — build ELO through the *same*
`CrossCompetitionEloBuilder` over the *same* corpus, loaded by the one shared
`load_european_corpus`.

This is not tidiness. A fixture's `elo_home` feature is read from that team's
**last historical row** — the rating it carried into its most recent match, not
its current rating. Reproducing that number means replaying the identical
chronological walk. A model trained on calibrated ratings and served
uncalibrated ones is *worse* than one never calibrated, because the features
stop meaning what the model learned them to mean — and nothing fails loudly.

`tests/models/test_train_serve_parity.py` pins the property directly, including
a byte-for-byte check that the Space's vendored copy of the builder has not
drifted from the original.

The Space reads `datasets/european.parquet`, uploaded by `upload_to_hf.py`
**already translated** to canonical keys — so it needs neither Supabase nor the
alias registry. An older snapshot without that file still works; the ratings
are simply uncalibrated, exactly as before.

---

## 9. Does it work?

```bash
uv run python scripts/evaluate_calibration.py
```

Measured on the real corpus after 99 approvals:

| League | before | after | shift |
|---|---|---|---|
| Premier League | 1724 | 1783 | +59 |
| Bundesliga | 1635 | 1664 | +29 |
| Scottish Premiership | 1688 | 1636 | **−52** |
| Super League Greece | 1500 | 1459 | **−41** |

**Leagues pinned at exactly 1500.0: 5 → 0.**

The Scottish Premiership result is the clearest evidence. It was rated *above*
the Bundesliga and La Liga because Celtic dominated a weak closed pool and ELO
had no way to know the pool was weak. European results told it.

| | log loss | Brier |
|---|---|---|
| uncalibrated | 1.0065 | 0.5977 |
| calibrated | **0.9636** | **0.5706** |

A +0.0428 log-loss improvement across 2,636 real European matches. These
ratings are fitted on the same matches they are scored against, so this
measures whether the linkage carries signal, not out-of-sample skill.

---

## 10. Deployment and CI

### What needs no setup

Nothing here reaches Render. The backend never loads the corpus — it serves
pre-computed predictions and the admin alias screen, and the latter reuses the
existing `SUPABASE_URL` / `SUPABASE_SERVICE_KEY`. **No new environment variable
is required on Render for European competitions.** No new API key either: the
corpus is a `git clone` of a public-domain repo.

The HF Space gets the corpus as `datasets/european.parquet`, uploaded already
translated to canonical keys, so it needs neither Supabase nor the registry.

The one thing that must be **committed**: `src/backend/data/teams_registry_historical.json`.
The backend image installs only the `backend` dependency group — no pandas — so
it cannot rebuild the registry at boot.

### Approvals must be exported to the seed

CI is given **no Supabase credentials**, deliberately — the network has no
business in the training path. So an approval that exists only in the
`team_aliases` table is invisible to every automated run.

After approving names, export and commit:

```bash
uv run python scripts/export_team_aliases.py
```

This writes the approvals into `config/team_aliases.yaml` under their `EU-*`
scopes. Supabase stays the **review** surface; the seed becomes the **record**.
The merge is additive and idempotent — existing hand-written entries survive,
and re-running produces no diff — so the file's history is a readable log of
what was approved and when.

The export refuses to write nothing. A Supabase read that fails and one that
legitimately returns zero rows look identical from outside, and only one of
them is safe, so both exit non-zero rather than reporting success.

### How this was going wrong

Both CI workflows silently produced uncalibrated output, for two independent
reasons, either of which alone erased the entire effect measured in §9:

| # | Cause | Effect |
|---|---|---|
| 1 | Neither `retrain.yml` nor `run-inference.yml` built the corpus, and `datasets/european/` is gitignored | `load_european_corpus` found no cache and returned empty — no European rows reached ELO at all |
| 2 | No Supabase credentials in CI, and `config/team_aliases.yaml` held **zero** `EU-` entries | Even with the corpus, `build_translator` saw none of the 99 approvals, so openfootball spellings stayed untranslated and linked nothing |

The runs still went green. `load_european_corpus` degrades by design — an
outage must never take predictions down — and printed a warning into a log
nobody reads. The Monday retrain would have replaced a calibrated model with an
uncalibrated one and reported success; the daily inference job would then have
served uncalibrated features to a model trained on calibrated ones, which §8
explains is worse than never calibrating at all.

Three things now prevent it:

1. **`scripts/export_team_aliases.py`** puts the approvals in git, where CI can
   read them without a credential.
2. **Both workflows build the corpus** before running, with no
   `continue-on-error`.
3. **`european.required: true`** in `config/config.yaml` makes the loader raise
   instead of degrading. The two guards cover each other: if the build step is
   ever removed, training refuses rather than quietly regressing.

`required` is off by default in `EuropeanConfig`, so an older config file, a
fresh checkout that has not built the corpus yet, and the HF Space (which
carries its own `SpaceConfig`) all keep working unchanged.

### Verifying it end to end

The check that catches every variant of this — run it after any change to the
corpus, the aliases or the workflows:

```bash
uv run python scripts/evaluate_calibration.py
```

If any league is back at *exactly* 1500.0, the corpus or the aliases did not
arrive. That number is not a coincidence; §1 explains why a sealed pool lands
there arithmetically.

---

## 11. Finding upcoming fixtures

openfootball supplies history. Fixtures come from an API, behind
`FixtureProvider` — one file per source, chained, first non-empty answer wins.

### What the free tiers actually do

Verified against the live services with real keys, August 2026. The published
tiers do not tell the whole story:

| Provider | Endpoint | Upcoming CL/EL/UECL? |
|---|---|---|
| **The Odds API** | `/v4/sports/{key}/events` | ✅ **Yes.** The one that works |
| **football-data.org** | `/v4/competitions/{code}/matches` | ⏳ CL only, and lagging a season |
| **API-Football** | — | ❌ Not usable; see below |

**The Odds API leads for a measured reason: `/events` costs zero credits.**
Fetching it left `x-requests-used` at 1 and `x-requests-remaining` at 499. Only
`/odds` draws on the 500/month budget, at one credit per region per market. Its
limit is inherent rather than contractual — it lists what bookmakers are
pricing, so the horizon is a week or two, and a competition that has not kicked
off returns nothing.

Every call is made by our own `requests` code, copied from the shapes in
[the-odds-api/samples-python](https://github.com/the-odds-api/samples-python)
and football-data's quickstart — same endpoints, same parameter names
(`api_key`, not `apiKey`), same quota headers. No vendor SDK is involved, so
what goes on the wire is visible in this repository.

**API-Football is excluded, and the way it fails is the point.** Asked for the
current season it returns:

```
HTTP 200   results: 0   errors: {"plan": "Free plans do not have access to
                                  this season, try from 2022 to 2024."}
```

A status-code check reads that as "no matches scheduled". A provider built on
one would have returned empty forever while looking perfectly healthy. This is
why providers here inspect response *bodies*, not just statuses.

**football-data.org is dormant, not dead**, and the distinction is kept in code.
A 403 (Europa) or 404 (Conference) means the plan excludes that competition, so
it is recorded in `uncovered` and never requested again — retrying would spend
the 10/minute allowance to learn nothing. A 200 with zero matches means the
season has not loaded yet, and must *not* disable the provider.

### Team names, again

Each provider spells clubs its own way, and none match openfootball —
`FC Kairat` from the Odds API is `Kairat Almaty` elsewhere. A fixture also
carries **no country code**, so the `EU-POR` scoping that makes corpus
suggestions safe is unavailable: candidates must be drawn from all 21 tracked
leagues, where "AC Sparta Praha" finds "Sparta Rotterdam".

So acceptance stays exact-match-or-approved-alias, and unresolved names queue
under the plain `EU` scope. But **reading** aliases spans every `EU-*` scope
too, which matters more than it sounds: without it the 99 clubs already
approved from the corpus would come back unresolved the moment they appeared in
a real fixture, and every one would need approving a second time. That is what
`TeamNameQuery.alias_search_scopes` provides — the same read/write split the
query already makes between `league_code` and `candidate_league_codes`.

Measured on a live fetch of the 2026-27 CL qualifying round: 10 fixtures, of
which `Union Saint-Gilloise → St. Gilloise` and `Olympiakos Piraeus →
Olympiakos` resolved straight from corpus approvals. Most of the rest are
genuinely unresolvable — Kairat, Levski, Bodø/Glimt, Sturm Graz and Sparta
Prague come from leagues this project does not track, so they have no domestic
history to link to, exactly as §4 describes.

---

## 12. Validating the corpus against an independent source

```bash
uv run python scripts/verify_corpus_against_api_football.py
```

Regression tests prove the openfootball parser agrees with itself. This checks
it against someone else, over the 2022–2024 seasons API-Football's free plan
covers — the one thing that plan is good for.

Result: **753 matches compared, 745 identical, 8 differing — every one a tie
that went past 90 minutes.** The corpus records the 90-minute score to match
football-data's `FTHG`/`FTAG` semantics; the API reports the score after extra
time:

```
UECL 2023  Olympiakos Piraeus vs ACF Fiorentina: 90' 0-0, after extra time 1-0
UECL 2024  Legia Warszawa vs Molde FK:           90' 1-0, after extra time 2-0
```

The script classifies those separately and exits non-zero only on an
*unexplained* disagreement, so it answers the question rather than handing a
list to a human. There are currently none — independent confirmation that the
parser's three traps in §3 are handled correctly.

Matching is scoped by kickoff date before names are compared loosely, and a
pairing is only accepted when exactly one candidate matches. Without the date
scoping only 8 of 125 matches paired; with it, 79. Requiring uniqueness is what
stops a loose name match inventing a discrepancy that does not exist.

---

## 13. Predicting a European fixture

```
fixture discovered (§11) → both teams resolved → Dixon-Coles + ELO → prediction
```

### The ensemble is deliberately not used

It is trained only on domestic rows. European results carry no shot, foul or
corner columns, so they never enter its training matrix — meaning it has never
seen a fixture whose two teams come from different leagues, and carries no
feature that would tell it one had arrived. Asking it anyway is extrapolation
presented as a prediction.

Dixon-Coles and ELO are usable *because of* the calibration in §1–§9:
Dixon-Coles is fitted on `combined_goals()`, so its attack and defence
parameters are identified across pools rather than arbitrary; ELO ratings sit
on one scale. Neither statement was true before the corpus existed.

Every prediction carries the model that produced it, so a UI can never present
one as though the ensemble had. The label follows `dixon_coles_weight`, which
§14 measured and set to zero — so in the shipped configuration it reads
`model: "elo"`, and Dixon-Coles contributes only the gate described below.

### Refusing rather than guessing

Three gates, each returning a stated reason: both names must resolve (§11),
Dixon-Coles must know both teams, and each must clear `min_matches_per_team`.
A club from an untracked league is refused outright —

```
SKIP FC Kairat vs Arsenal: no history for 'FC Kairat' — its league is
     not tracked, so there is nothing to predict from
```

— which is the same contract the name resolver applies to spellings.

### The fitted ELO instance must be reused, never rebuilt

`CrossCompetitionEloBuilder.build()` returns **domestic rows only**. European
results update the ratings without ever appearing in the output, so
recomputing ELO from that frame silently discards all of them and returns
exactly the uncalibrated numbers. The calibrated ratings live on the injected
`FootballELO` instance and nowhere else.

Measured, full corpus:

| team | rebuilt from output (wrong) | fitted instance (right) |
|---|---|---|
| Celtic | 1924.8 | 1855.3 |
| Sp Lisbon | 1906.3 | 1862.1 |
| Man City | 1902.3 | 1936.3 |
| Liverpool | 1781.7 | 1852.1 |

The wrong column is bit-identical to the pre-calibration table in §1 — Celtic
above Man City, exactly the artefact this whole track exists to remove.
`tests/models/test_elo_pool_scope.py::TestFittedRatingsMustBeReused` pins it.

### What is still weak

ELO calibrates well. **Dixon-Coles does not calibrate as strongly**, and the
reason is arithmetic: 2,636 European matches against 63,724 domestic ones, so
a club's attack parameter is still dominated by how it scores against its own
league. Sporting's attack sits at +0.775 against Manchester City's +0.543,
which reflects Portuguese opposition more than European strength.

At the original `dixon_coles_weight: 0.6` that bias carried into the blend:

```
Sp Lisbon vs Man City   H 0.481  D 0.225  A 0.294  -> Home Win
```

Sporting is rated *below* Manchester City on calibrated ELO (1862 vs 1936), so
that outcome came from the Dixon-Coles side of the blend, not the ELO side.

**0.6 was never validated when it was chosen** — it came from analogy with the
domestic blend, not measurement. §14 measured it and the answer was more
emphatic than "lean towards ELO": Dixon-Coles earns no weight at all on
held-out European ties. The weight is now `0.0`, so the symptom above is gone
and the same fixture reads:

```
Sp Lisbon vs Man City   H 0.383  D 0.220  A 0.397  -> Away Win
```

The margin is thin, and honestly so: ELO has Sporting at 1860 against City's
1932, and home advantage covers most of a 72-point gap.

What remains weak is the underlying fact rather than the blend: Dixon-Coles's
cross-league parameters are still dominated by domestic opposition, which is
why it is worth re-measuring as the corpus grows rather than treating `0.0` as
permanent.

---

## 14. Measuring the blend parameters

`scripts/tune_european_weight.py` does for the European blend what
`scripts/tune_blend_weight.py` does for the domestic one. Report-only: it
prints tables and never edits configuration.

### How it is measured

**Rolling-origin folds, one per season.** Only 1,314 of the 2,636 corpus
matches have both teams in a tracked league, leaving roughly 190 evaluable
matches in a recent season — far too few to separate a weight of 0.5 from 0.6.
Five folds pooled give 1,301. Each fold's Dixon-Coles is refitted on matches
strictly before that fold's cut, which is also closer to how the system runs:
it retrains periodically and serves the fixtures that follow.

**ELO needs no per-fold work.** `compute_elo_features` records each match's
rating *before* applying it, so one walk over the whole corpus yields a
leak-free expectation for every European match in every fold at once. This is
why `CrossCompetitionEloBuilder.build_all()` exists — `build()` returns
domestic rows only, and the European rows are exactly what has to be scored.

**The refusal gates are the predictor's own.** The backtester constructs a real
`EuropeanMatchPredictor` per fold and calls `can_predict`, rather than
reimplementing the checks. Tuning on matches the system would refuse would
optimise for fixtures it never serves. Match counts come from the training
side of the cut, so a holdout season cannot vouch for its own teams.

**Both parameters are swept together.** `elo_draw_rate` belongs to the ELO leg
alone, so a badly chosen one makes ELO score worse than it is and drags the
weight toward Dixon-Coles. Varying it is free: the backtest stores raw expected
scores, so a different draw rate is a recomputation rather than a re-walk.

### The result

1,301 held-out matches across five seasons (2021-22 to 2025-26), 637 refused:

```
             model    log_loss     brier   accuracy
          elo only      1.0006    0.5993      0.508
  dixon-coles only      1.0296    0.6185      0.490
         base rate      1.0522    0.6349      0.473
    current config      1.0111    0.6062      0.503
```

Log-loss at `elo_draw_rate 0.22`, by weight:

```
  weight    log_loss
    0.00      0.9994  <- best
    0.20      1.0009
    0.40      1.0047
    0.60      1.0105   (what was configured)
    0.80      1.0187
    1.00      1.0296
```

**The profile is monotonic.** Every unit of weight given to Dixon-Coles makes
the prediction worse, across the whole grid and at every draw rate. This is not
a shallow optimum whose location is arguable — there is no interior minimum at
all. The best cell is `dixon_coles_weight 0.00`, `elo_draw_rate 0.22`, at
log-loss 0.9994 against 1.0111 for the configured 0.6/0.25: an improvement of
+0.0118 with a 95% bootstrap interval of [+0.0042, +0.0199] over 1,301 matches.
The interval excludes zero.

That is the measurement §13 predicted qualitatively — the Sporting-vs-City
example is the visible symptom — and it is stronger than expected. Dixon-Coles
earns no weight on European ties.

### What the result does *not* say

**Not that Dixon-Coles is broken.** It carries the domestic blend at
`blend_weight 0.4` and produces the scoreline, O/U and BTTS markets. The claim
is narrower: its cross-league attack and defence parameters, identified by
2,636 European matches against 63,724 domestic ones, are not informative enough
about a cross-league 1X2 to improve on ELO.

**Not that the European predictions are good.** The gap between the best cell
and a constant base rate is 0.053 nats, and accuracy moves from 0.473 to 0.508.
That is a real edge but a small one, and it is the honest ceiling of what these
two models know about a Champions League tie.

### What was adopted

Both coordinates of the best cell:

```yaml
european:
  prediction:
    dixon_coles_weight: 0.0    # was 0.6
    elo_draw_rate: 0.22        # was 0.25
```

`elo_draw_rate` moved with the weight because at weight 0 it is the model's
only remaining free parameter, and 0.25 was as hand-picked as the 0.6 was.

**The label had to move too.** At weight 0 these predictions are ELO and
nothing else, so a fixed `dixon-coles+elo` would have been false — and the
`model` field exists precisely to stop a prediction overstating what produced
it. `EuropeanMatchPredictor.model_name` is now derived from the weight:

| `dixon_coles_weight` | `model` |
|---|---|
| `0.0` | `elo` |
| between | `dixon-coles+elo` |
| `1.0` | `dixon-coles` |

Deriving rather than renaming keeps it true in both directions. The corpus
grows every season, so re-running the sweep and finding a non-zero weight is a
real possibility, and the label must not need remembering when that happens.
The leg carrying zero weight is not computed at all, so a Dixon-Coles failure
can no longer perturb a prediction it contributes nothing to.

**Dixon-Coles is still required, and still fitted.** `refusal_reason` asks it
whether it `knows` each club, and that gate stays: it is a question about
whether a club has any history, which is different from which model turns that
history into a number — and nothing else can answer it, since a club absent
from `match_counts` entirely passes the count check. So European predictions
still depend on a fitted Poisson model even though its distribution is
discarded. That coupling is deliberate for now; decoupling it means reworking
the gate and `run_inference`'s `poisson_model is None` early return.

### Running it

```bash
uv run python scripts/tune_european_weight.py
uv run python scripts/tune_european_weight.py --max-folds 2   # quicker
```

Roughly two minutes: five Dixon-Coles fits at ~18s each, plus the data load.
Everything it uses is configuration — `european.prediction.tuning` in
`config/config.yaml` holds the grids, the fold count, the season boundary and
the bootstrap seed.
