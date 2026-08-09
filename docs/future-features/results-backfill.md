# Results backfill

Settle finished fixtures against the predictions that were stored for them, so
the product can show whether the model was right — and so the dashboard's
`MODEL ACCURACY` block stops being a placeholder.

This is the cheap alternative to live scores. Live in-play data needs a paid
provider and a continuously-running poller; settled results need neither,
because the pipeline already downloads them.

---

## 1. Goals

- Attach the actual outcome to every stored prediction once the match is over.
- Maintain a rolling accuracy series the frontend can read in one query.
- Report a calibration metric, not only a hit rate — the blend weights are
  tuned on log-loss, so accuracy alone would be the wrong scoreboard.
- Add **no new paid dependency** and no always-on process.

## 2. Non-goals

- In-play scores, minute markers, live status. Explicitly out; see §11.
- Re-scoring or retraining off the back of settlement. The backfill records
  what happened; deciding what to do about it stays with `retrain.yml`.
- Backfilling history older than the retention window in §6.

---

## 3. Where results come from

Two sources, tried in order, using the same first-non-empty chain the European
fixtures already use ([`src/scrapers/european/chained.py`](../../src/scrapers/european/chained.py)).

| Order | Source | Cost | Settles | Why this position |
|---|---|---|---|---|
| 1 | **football-data.co.uk** current-season CSV | free, already fetched by `data_loader` | ~1–2 days after the match | Authoritative, stable schema, carries `FTHG`/`FTAG`/`FTR` directly. Slow, but §7 makes slowness harmless. |
| 2 | **FlashScore** `scrape_results` | free | same night | Already implemented — `FlashScoreFixture` carries `status`, `home_score`, `away_score`, and `run-inference.yml` already installs Playwright. Brittle (DOM scrape), so it is the fallback, not the primary. |

Ordering is a correctness property here in the same sense as the fixture chain:
football-data first means a match that is merely *late* to appear is left
unsettled rather than settled from a scrape that may have parsed the wrong row.

**European competitions (CL/EL/ECL) have no football-data feed.** They settle
from source 2 only, or stay unsettled. That is the existing asymmetry, not a
new one.

---

## 4. Schema

### 4.1 `predictions.payload` — additive

Each entry in `payload.matches[]` gains one optional object. Nothing is
removed or renamed, so the existing read path and the `MatchPrediction`
TypeScript type stay valid; the field is simply absent until settlement.

```jsonc
{
  "home_team": "Liverpool",
  "away_team": "Bournemouth",
  // ... existing fields unchanged ...
  "actual": {
    "home_goals": 3,
    "away_goals": 0,
    "result": "H",              // H | D | A, football-data's FTR convention
    "correct": true,            // argmax(probabilities) == result
    "settled_at": "2026-08-10T03:04:11Z",
    "source": "football-data"   // or "flashscore"
  }
}
```

`correct` is precomputed rather than derived in the frontend so that the
definition of "right" lives in exactly one place, next to the probabilities it
is derived from.

### 4.2 New table `prediction_accuracy`

The sparkline must not require scanning N days of JSONB payloads. One
denormalised row per league per day:

```sql
create table prediction_accuracy (
  day           date        not null,          -- ISO, NOT the dd/mm/yyyy key
  league_code   text        not null,          -- '_ALL_' for the app-wide roll-up
  settled       int         not null,          -- matches with an outcome
  correct       int         not null,
  brier_sum     double precision not null,     -- Σ multiclass Brier
  logloss_sum   double precision not null,     -- Σ −ln p(actual)
  updated_at    timestamptz not null default now(),
  primary key (day, league_code)
);
create index on prediction_accuracy (league_code, day desc);
```

> **`day` is a real `date`, deliberately.** The `predictions` table keys on
> `match_date` as a `dd/mm/yyyy` *string*; ordering those lexically sorts by
> day-of-month and is wrong. The backfill converts once, on write.

Sums rather than averages, so any window is a single `sum()/sum()` and
re-running a day is an overwrite rather than a merge.

---

## 5. The job

`scripts/backfill_results.py`, mirroring `run_inference.py`'s structure.

```
for each day D in the settlement window (§6):
    rows ← supabase: predictions where match_date = fmt(D)
    for each row:
        fixtures ← payload.matches[] lacking `actual`
        if none: continue
        results ← ResultProvider.chain().results_for(row.league_code, D)
        for each fixture matched by (home_team, away_team):
            attach `actual`
        upsert predictions row (payload only)
        upsert prediction_accuracy (D, league_code)
    upsert prediction_accuracy (D, '_ALL_')
```

Team-name matching reuses the existing canonical resolver. A fixture whose
name does not resolve is **left unsettled and reported**, never guessed — same
rule the alias review queue enforces. An unresolvable name is a signal that
belongs in that queue, not a coin flip in an accuracy stat.

### Workflow

`.github/workflows/backfill-results.yml`, `cron: "0 3 * * *"`.

03:00 UTC sits after European late kickoffs have finished and before
`run-inference.yml` at 06:30, so a day's dashboard is already settled by the
time the next day's predictions land.

---

## 6. Settlement window and idempotency

The job processes the **last 4 days**, not just yesterday.

football-data publishes in batches, typically twice a week. A single-day
window would permanently lose any match whose CSV update arrived late. A
4-day window means every match gets four chances to settle, and the job
self-heals with no manual intervention.

This is safe because the job is idempotent:

- matches already carrying `actual` are skipped, so a settled result is never
  rewritten by a later, different source;
- `prediction_accuracy` is recomputed from the payload and upserted on
  `(day, league_code)`, so a re-run converges rather than accumulates.

Days older than the window are never revisited. A match still unsettled after
4 days stays unsettled and is excluded from every metric — it is not counted
as a miss.

---

## 7. Why lag is acceptable

The consumer is a *rolling* accuracy figure over ~20 days. A one-to-two-day
settlement lag shifts the window's edge; it does not distort the number. This
is precisely why the slow-but-authoritative source can go first.

---

## 8. API

```
GET /api/accuracy?window=20&league_code=_ALL_
```

```jsonc
{
  "window_days": 20,
  "settled": 412,
  "correct": 219,
  "accuracy": 0.5316,
  "brier": 0.5980,          // mean, lower is better
  "log_loss": 0.9910,       // mean — comparable to the tuning metric
  "series": [               // oldest → newest, one entry per day with data
    { "day": "2026-07-21", "settled": 22, "correct": 13 }
  ]
}
```

Served by a new `src/backend/api/accuracy.py` reading `prediction_accuracy`
only — never the payloads — behind the same TTL cache
`PredictionService` uses. One indexed range scan per cache miss.

`series` omits days with no fixtures rather than emitting zeros, so an
international break reads as a gap and not as a run of failures.

---

## 9. Frontend

Two changes, both filling holes the design already has:

1. **`DashboardRail`** — the substituted counters block becomes the design's
   real `MODEL ACCURACY`: headline percentage plus the 20-bar sparkline, one
   bar per day, height from that day's accuracy. Log-loss goes underneath as a
   secondary figure.
2. **`MatchRow`** — a settled row shows the final score and a hit/miss marker
   in place of its confidence badge. Unsettled rows are unchanged.

`MatchPrediction` gains `actual: ActualResult | null`. Because the field is
optional and absent until settlement, nothing needs a migration.

---

## 10. Failure modes

| Failure | Behaviour |
|---|---|
| Both sources return nothing for a day | Day left unsettled, retried for 4 days, then dropped. No metric written. |
| Team name does not resolve | Fixture left unsettled and logged; surfaces in the alias queue. |
| FlashScore DOM changes | Chain falls through to nothing; football-data still settles domestic leagues on its own schedule. |
| Job fails entirely | Next night's run covers the same window. No state to repair. |
| Supabase write fails mid-day | Payload and roll-up are upserted per league; a partial day is corrected on the next run. |

---

## 11. What this deliberately does not become

The `actual` field is a *post-match* record. It is not a stepping stone to
live scores: nothing here polls, and the 03:00 cadence is chosen so that it
cannot be repurposed as one without a redesign. Live remains blocked on a paid
provider and a persistent poller.

---

## 12. Cost and effort

- **Recurring cost: none.** Both sources are already fetched by the pipeline;
  no new provider, no quota, no always-on process.
- **Added CI time:** one short job per day, well inside the free allowance.
- **Effort:** roughly one day — script and provider chain (~half), migration
  and endpoint (~quarter), frontend and tests (~quarter).

---

## 13. Open questions

1. **Does a void/abandoned fixture count?** Proposed: excluded entirely, same
   treatment as unsettled. Needs confirming against how football-data reports
   them.
2. **Per-league accuracy in the UI?** The table supports it; the design shows
   only one figure. Proposed: ship `_ALL_` first.
3. **Should European (`model: "elo"`) fixtures be pooled into `_ALL_`?**
   They are predicted by a different model, so pooling makes the headline a
   blend of two systems. Proposed: keep them in `_ALL_` for the product-level
   number, and rely on per-league rows for model-level analysis.
