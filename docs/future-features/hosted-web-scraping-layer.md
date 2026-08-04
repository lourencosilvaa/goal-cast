# Future feature: Hosted web-scraping layer (Firecrawl / Tavily)

**Status:** Deferred — reverted from branch `refactor/odds-and-training-improvements`
on 2026-08-04 to avoid adding complexity before it's needed.

## Motivation

Two gaps in the current pipeline this layer was meant to close:

1. **Portuguese-market odds.** The CSV pipeline carries UK-market Bet365 prices,
   which diverge from what Portuguese books (Betano, Betclic) actually offer.
2. **Live match context.** Recent form (W/D/L), head-to-head history and league
   standings that FlashScore renders on its match pages are not currently fed
   into the prediction/model stage.

The idea was to render those pages once through a hosted scraping provider
(Firecrawl, with Tavily as an alternative) and extract structured data via a
JSON schema, keeping it fully optional and off by default.

## Design (as reverted)

### Abstraction

A `BaseWebClient` ABC with two capabilities, so the provider is swappable by
configuration without touching the fetchers that consume it:

- `scrape(url) -> str` — render a page and return markdown/text.
- `extract(url, schema, prompt=None) -> dict` — return structured JSON matching a
  supplied schema.
- Plus `source_name` and `is_available` properties.

`FirecrawlClient` was the concrete implementation, calling the Firecrawl
`/scrape` endpoint directly via `requests` (no SDK) with `Authorization: Bearer`.

### Consumers

- `PortugueseOddsFetcher` — renders each configured betting-site league landing
  page and extracts every match's 1X2 decimal odds in one call, then fuzzy-matches
  (normalized substring) to a specific fixture by team name. Emits `ScrapedOdds`.
- `MatchContextFetcher` — extracts `home_form` / `away_form` (W/D/L lists),
  `head_to_head`, and `home_standing` / `away_standing` from a FlashScore match
  page. Emits a `MatchContext` dataclass.

Both were gated behind `enabled` + `client.is_available` and swallowed extraction
errors (anti-bot blocks, timeouts, malformed responses) by returning empty
results — never breaking the default pipeline.

### Proposed file layout

```
src/scrapers/web/base_web_client.py      # BaseWebClient ABC
src/scrapers/web/firecrawl_client.py     # FirecrawlClient(BaseWebClient)
src/scrapers/odds/pt_odds_fetcher.py     # PortugueseOddsFetcher
src/scrapers/context/match_context_fetcher.py  # MatchContext + MatchContextFetcher
```
(with a mirrored `tests/scrapers/{web,odds,context}/` tree)

### Configuration (Pydantic + YAML)

A `WebScraperConfig` nested under `scrapers.web`, disabled by default and
auto-enabled only when an API key is injected from the environment:

```yaml
scrapers:
  web:
    provider: "firecrawl"        # firecrawl | tavily
    enabled: false               # flips on automatically when an API key is present
    base_url: "https://api.firecrawl.dev/v1"
    api_key: ""                  # set via FIRECRAWL_API_KEY env var
    tavily_api_key: ""           # set via TAVILY_API_KEY env var
    request_timeout: 30
    betting_sites:
      betano:
        name: "Betano"
        league_urls:
          P1: "https://www.betano.pt/sport/futebol/portugal/liga-portugal/17106/"
      betclic:
        name: "Betclic"
        league_urls:
          P1: "https://www.betclic.pt/futebol-s1/portugal-liga-portugal-c33"
    flashscore_context:
      enabled: true
      base_url: "https://www.flashscore.com"
```

Env injection in `load_config`: read `FIRECRAWL_API_KEY` / `TAVILY_API_KEY`,
write them into `scrapers.web`, and set `enabled: true` when the active
provider's key is present — so no secret is ever committed to `config.yaml`.

`.env` keys:

```
FIRECRAWL_API_KEY=
TAVILY_API_KEY=
```

## Why deferred

- Adds a paid third-party dependency and a live-scraping code path before there
  is a proven need for Portuguese-market odds / live context in the model.
- Anti-bot fragility on betting sites means ongoing maintenance.
- The default CSV / Odds-API pipeline already works end-to-end.

## To resurrect

Reintroduce the four modules + mirrored tests, the `WebScraperConfig` Pydantic
model and `scrapers.web` YAML block, the `load_config` env injection, and the
`.env` keys. Then wire `PortugueseOddsFetcher` / `MatchContextFetcher` into the
odds aggregation and prediction stages (they were built standalone and never
connected to the rest of the app).
