"""
Local review tool for queued team names.

Stands in for the deployed admin panel while that runs older code whose team
picker cannot handle European scopes. Everything is local: it reads and writes
the same Supabase ``team_aliases`` table the deployed app uses, so approvals
made here are the real thing and show up there without a redeploy.

No login. It binds to localhost only and uses the service key already in
``.env``, so it is a developer tool, not something to expose.

Usage:
    uv run python scripts/review_team_names.py
    uv run python scripts/review_team_names.py --port 8765
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from config.config_loader import Config, load_config  # noqa: E402
from src.backend.core.supabase_client import get_supabase_client  # noqa: E402
from src.backend.services.team_alias_service import TeamAliasService  # noqa: E402
from src.teams.registry import load_team_registry  # noqa: E402
from src.teams.resolver import (  # noqa: E402
    StaticTeamAliasRepository,
    TeamNameQuery,
    TeamNameResolver,
)

APPROVED_BY = "local-review"


class ApproveRequest(BaseModel):
    league_code: str
    raw_name: str
    canonical_name: str


def build_names_resolver(config: Config) -> TeamNameResolver:
    """Resolver used only to propose candidates, exactly as the API does."""
    return TeamNameResolver(
        load_team_registry(config.teams.historical_registry_path),
        StaticTeamAliasRepository(config.teams.aliases.seed_path),
        config.teams.aliases,
    )


def candidate_leagues(scope: str, config: Config) -> tuple[str, ...]:
    """Leagues a queued name may be matched against.

    A domestic scope is itself a league. A European scope carries the country
    instead (``EU-POR``), and the candidates are that country's leagues —
    which is what keeps "AC Sparta Praha" from being offered a Dutch club.
    """
    european = config.european
    prefix = f"{european.alias_scope}-"
    if not scope.startswith(prefix):
        return (scope,)
    country = scope[len(prefix) :]
    return tuple(european.country_leagues.get(country, []))


def country_of(scope: str, config: Config) -> Optional[str]:
    prefix = f"{config.european.alias_scope}-"
    return scope[len(prefix) :] if scope.startswith(prefix) else None


def pending_entries(
    service: Any, resolver: TeamNameResolver, config: Config
) -> list[dict[str, Any]]:
    """Every queued name with its suggestions and full candidate list."""
    registry = load_team_registry(config.teams.historical_registry_path)
    entries: list[dict[str, Any]] = []
    for row in service.list_pending():
        scope = str(row.get("league_code", ""))
        raw_name = str(row.get("raw_name", ""))
        leagues = candidate_leagues(scope, config)
        options: list[str] = []
        for league in leagues:
            options.extend(registry.get(league, []))
        suggestions = resolver.resolve(
            TeamNameQuery(
                league_code=scope,
                raw_name=raw_name,
                candidate_league_codes=leagues,
            )
        ).suggestions
        entries.append(
            {
                "league_code": scope,
                "raw_name": raw_name,
                "country": country_of(scope, config),
                "suggestions": suggestions,
                "options": sorted(set(options)),
            }
        )
    return sorted(entries, key=lambda e: (e["country"] or "", e["raw_name"]))


def create_app(config: Config) -> FastAPI:
    app = FastAPI(title="Team name review (local)")
    resolver = build_names_resolver(config)
    service = TeamAliasService(get_supabase_client())

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.get("/api/pending")
    def pending() -> dict[str, Any]:
        entries = pending_entries(service, resolver, config)
        approved = len(service.list_approved())
        return {"pending": entries, "approved": approved}

    @app.post("/api/approve")
    def approve(body: ApproveRequest) -> dict[str, bool]:
        options = []
        registry = load_team_registry(config.teams.historical_registry_path)
        for league in candidate_leagues(body.league_code, config):
            options.extend(registry.get(league, []))
        if body.canonical_name not in options:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"'{body.canonical_name}' is not a team in "
                    f"'{body.league_code}'."
                ),
            )
        service.approve(
            league_code=body.league_code,
            raw_name=body.raw_name,
            canonical_name=body.canonical_name,
            approved_by=APPROVED_BY,
        )
        return {"success": True}

    return app


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Team name review</title>
<style>
  :root { color-scheme: dark light; }
  body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0;
         background:#0e1116; color:#e6edf3; }
  header { padding: 18px 24px; border-bottom:1px solid #232b36;
           position:sticky; top:0; background:#0e1116; z-index:2; }
  h1 { font-size: 15px; margin:0 0 4px; font-weight:600; }
  .sub { font-size:12px; color:#8b98a5; }
  .wrap { padding: 16px 24px 60px; max-width: 900px; }
  .row { display:flex; gap:10px; align-items:center; padding:10px 12px;
         border:1px solid #232b36; border-radius:10px; margin-bottom:8px;
         background:#151a21; }
  .row.done { opacity:.45; }
  .name { flex:1; min-width:0; font-size:14px;
          overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .badge { font-size:10px; padding:2px 6px; border-radius:5px;
           background:#1b3a5c; color:#7cc4ff; border:1px solid #24507c; }
  select { background:#0e1116; color:#e6edf3; border:1px solid #2c3644;
           border-radius:8px; padding:6px 8px; font-size:12px; width:250px; }
  button { background:#1f6feb; color:#fff; border:0; border-radius:8px;
           padding:7px 14px; font-size:12px; cursor:pointer; }
  button:disabled { background:#30363d; color:#8b98a5; cursor:not-allowed; }
  .err { color:#ff7b72; font-size:12px; padding:6px 0; }
  .empty { color:#8b98a5; font-size:13px; padding:24px 0; }
</style></head>
<body>
<header>
  <h1>Team name review <span class="sub" id="count"></span></h1>
  <div class="sub">The top suggestion is pre-selected — nothing is saved until
  you press Approve. Suggestions are scoped to the club's own country.</div>
</header>
<div class="wrap"><div id="err" class="err"></div><div id="list"></div></div>
<script>
let state = [];
async function load() {
  const res = await fetch('/api/pending');
  if (!res.ok) { document.getElementById('err').textContent =
      'Could not read the queue: ' + res.status; return; }
  const data = await res.json();
  state = data.pending;
  document.getElementById('count').textContent =
    state.length + ' to review \\u00b7 ' + data.approved + ' approved';
  render();
}
function render() {
  const list = document.getElementById('list');
  if (!state.length) { list.innerHTML =
      '<div class="empty">Nothing left to review.</div>'; return; }
  list.innerHTML = '';
  state.forEach((e, i) => {
    const row = document.createElement('div');
    row.className = 'row';
    const opts = [];
    if (e.suggestions.length) {
      opts.push('<optgroup label="Suggestions">' + e.suggestions.map(
        n => `<option value="${n}">${n}</option>`).join('') + '</optgroup>');
    }
    opts.push('<optgroup label="All teams">' + e.options.map(
      n => `<option value="${n}">${n}</option>`).join('') + '</optgroup>');
    row.innerHTML =
      `<span class="name">${e.raw_name}</span>` +
      (e.country ? `<span class="badge">${e.country}</span>` : '') +
      `<select id="sel-${i}"><option value="">Choose team…</option>` +
      opts.join('') + `</select>` +
      `<button id="btn-${i}">Approve</button>`;
    list.appendChild(row);
    const sel = row.querySelector('select');
    if (e.suggestions.length) sel.value = e.suggestions[0];
    row.querySelector('button').onclick = () => approve(e, sel, row);
  });
}
async function approve(entry, sel, row) {
  if (!sel.value) return;
  const btn = row.querySelector('button');
  btn.disabled = true; btn.textContent = '…';
  const res = await fetch('/api/approve', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ league_code: entry.league_code,
      raw_name: entry.raw_name, canonical_name: sel.value })
  });
  if (res.ok) {
    row.classList.add('done'); btn.textContent = 'Approved'; sel.disabled = true;
    const c = document.getElementById('count');
    c.textContent = c.textContent.replace(/^(\\d+)/, (m) => String(+m - 1));
  } else {
    const body = await res.json().catch(() => ({}));
    document.getElementById('err').textContent = body.detail || 'Approval failed';
    btn.disabled = false; btn.textContent = 'Approve';
  }
}
load();
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Local team-name review tool")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"\nReview tool: http://127.0.0.1:{args.port}\n")
    uvicorn.run(
        create_app(config), host="127.0.0.1", port=args.port, log_level="warning"
    )


if __name__ == "__main__":
    main()
