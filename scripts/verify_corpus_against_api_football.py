"""
Cross-check the openfootball European corpus against API-Football.

The corpus is parsed from plain-text files by a bespoke parser, and three of
its conventions silently corrupt data if misread — most sharply, a scoreline
puts the *decisive* number first, so `3-4 pen. 1-1 a.e.t. (1-1, 0-0)` records
a penalty shootout whose 90-minute score is 1-1. Reading the leading pair
would file the 2012 final as Bayern 3-4 Chelsea. The parser handles it and
regression tests pin it, but tests only prove the parser agrees with itself.

This compares it against an independent source.

Scope is set by the free plan, not by choice: API-Football answers only for
seasons **2022 to 2024**. Ask for anything newer and it returns HTTP 200 with
an empty result set and an explanation buried in `body["errors"]` — which is
why this script checks that field rather than the status code, and why
API-Football is not used as a fixture provider at all.

This is a validation tool. Nothing at runtime depends on it.

Usage:
    uv run python scripts/verify_corpus_against_api_football.py
    uv run python scripts/verify_corpus_against_api_football.py --season 2023
    uv run python scripts/verify_corpus_against_api_football.py --competition CL

Environment variables (read from .env):
    API_FOOTBALL_API_KEY — free-tier key from api-football.com
"""

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from config.config_loader import load_config  # noqa: E402
from src.corpus.supplementary import StaticFileCorpusSource  # noqa: E402

BASE_URL = "https://v3.football.api-sports.io"
AUTH_HEADER = "x-apisports-key"
API_KEY_ENV = "API_FOOTBALL_API_KEY"

#: Our competition code → API-Football league id. Verified live.
LEAGUE_IDS: dict[str, int] = {"CL": 2, "EL": 3, "UECL": 848}

#: Seasons the free plan actually serves. Outside this the API returns 200
#: with `errors.plan` set, which is indistinguishable from "no matches" unless
#: the body is inspected.
FREE_SEASONS: tuple[int, ...] = (2022, 2023, 2024)

#: Statuses whose score is final and therefore comparable.
FINISHED = {"FT", "AET", "PEN"}

#: Statuses where the API's score includes goals scored after 90 minutes.
#: The corpus deliberately records the 90-minute score to match
#: football-data's FTHG/FTAG semantics, so a difference here is the two
#: sources answering different questions, not one of them being wrong.
EXTRA_TIME = {"AET", "PEN"}


@dataclass
class Discrepancy:
    competition: str
    season: int
    home: str
    away: str
    corpus_score: str
    api_score: str
    extra_time: bool


def fetch_fixtures(api_key: str, league_id: int, season: int) -> list[dict[str, Any]]:
    """One page of finished fixtures, or a raised error explaining why not.

    API-Football signals failure with HTTP 200 and a populated ``errors``
    field. Trusting the status code here would report "0 discrepancies" for a
    request that was rejected outright — a false clean bill of health, which
    is worse than no check at all.
    """
    import requests

    response = requests.get(
        f"{BASE_URL}/fixtures",
        headers={AUTH_HEADER: api_key},
        params={"league": league_id, "season": season},
        timeout=30,
    )
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")

    body = response.json()
    errors = body.get("errors")
    # An empty list means "no errors"; a dict means there were some.
    if isinstance(errors, dict) and errors:
        raise RuntimeError("; ".join(f"{k}: {v}" for k, v in errors.items()))

    remaining = response.headers.get("x-ratelimit-requests-remaining")
    if remaining is not None:
        print(f"    ({remaining} of today's 100 requests left)")
    return body.get("response", [])


@dataclass(frozen=True)
class ApiMatch:
    home: str
    away: str
    score: tuple[int, int]
    status: str

    @property
    def went_to_extra_time(self) -> bool:
        return self.status in EXTRA_TIME


def _api_index(fixtures: list[dict[str, Any]]) -> dict[Any, list[ApiMatch]]:
    """Finished fixtures grouped by kickoff date.

    Grouping by date first is what makes loose name matching safe. A single
    competition plays a handful of matches on any given night, so within one
    date a partial name match is almost certainly the right game — whereas
    across a whole season it could pair the wrong two clubs and invent a
    discrepancy that does not exist.
    """
    from datetime import datetime

    index: dict[Any, list[ApiMatch]] = defaultdict(list)
    for item in fixtures:
        info = item.get("fixture", {})
        if info.get("status", {}).get("short") not in FINISHED:
            continue
        teams = item.get("teams", {})
        home = _normalise(teams.get("home", {}).get("name"))
        away = _normalise(teams.get("away", {}).get("name"))
        goals = item.get("goals", {})
        if not home or not away:
            continue
        if goals.get("home") is None or goals.get("away") is None:
            continue
        raw_date = info.get("date")
        if not isinstance(raw_date, str):
            continue
        try:
            kickoff = datetime.fromisoformat(raw_date).date()
        except ValueError:
            continue
        index[kickoff].append(
            ApiMatch(
                home,
                away,
                (int(goals["home"]), int(goals["away"])),
                str(info.get("status", {}).get("short") or ""),
            )
        )
    return index


def _normalise(name: Any) -> str:
    """Loose key for matching two sources that spell clubs differently."""
    if not isinstance(name, str):
        return ""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _same_club(one: str, other: str) -> bool:
    """Whether two normalised spellings plausibly name the same club.

    Containment rather than equality, because the two sources differ by
    affixes far more than by substance: "FC Bayern München" against "Bayern
    München", "Manchester City FC" against "Manchester City". A minimum length
    stops short fragments matching half the draw.
    """
    if not one or not other:
        return False
    if one == other:
        return True
    shorter, longer = sorted((one, other), key=len)
    return len(shorter) >= 5 and shorter in longer


def _find(api_matches: list[ApiMatch], home: str, away: str) -> ApiMatch | None:
    """The single API fixture matching this pairing, or None if ambiguous.

    Requiring uniqueness is the safeguard: two candidates mean the loose name
    match was not decisive, and reporting a disagreement on a guess would be
    worse than reporting nothing.
    """
    hits = [
        match
        for match in api_matches
        if _same_club(match.home, home) and _same_club(match.away, away)
    ]
    return hits[0] if len(hits) == 1 else None


def compare(corpus, api_index: dict, competition: str, season: int) -> tuple[list, int]:
    """Discrepancies and the number of rows actually compared."""
    import pandas as pd
    from datetime import timedelta

    discrepancies: list[Discrepancy] = []
    compared = 0
    for _, row in corpus.iterrows():
        home = _normalise(row["HomeTeam"])
        away = _normalise(row["AwayTeam"])
        corpus_date = pd.to_datetime(row["Date"]).date()

        # Kickoffs near midnight land on different calendar days once time
        # zones are applied, so neighbouring dates are searched too.
        found = None
        for offset in (0, -1, 1):
            candidates = api_index.get(corpus_date + timedelta(days=offset), [])
            found = _find(candidates, home, away)
            if found is not None:
                break
        if found is None:
            continue

        compared += 1
        corpus_score = (int(row["FTHG"]), int(row["FTAG"]))
        if corpus_score != found.score:
            discrepancies.append(
                Discrepancy(
                    competition=competition,
                    season=season,
                    home=str(row["HomeTeam"]),
                    away=str(row["AwayTeam"]),
                    corpus_score=f"{corpus_score[0]}-{corpus_score[1]}",
                    api_score=f"{found.score[0]}-{found.score[1]}",
                    extra_time=found.went_to_extra_time,
                )
            )
    return discrepancies, compared


def _season_of(value: Any) -> int | None:
    """openfootball season "2223" → the API's starting year 2022."""
    text = str(value)
    if len(text) != 4 or not text.isdigit():
        return None
    return 2000 + int(text[:2])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the openfootball corpus with API-Football"
    )
    parser.add_argument(
        "--season",
        type=int,
        action="append",
        help=f"Season start year. Free plan covers {FREE_SEASONS}. Repeatable.",
    )
    parser.add_argument(
        "--competition",
        action="append",
        choices=sorted(LEAGUE_IDS),
        help="Competition code. Repeatable. Default: all.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    api_key = os.environ.get(API_KEY_ENV, "")
    if not api_key:
        print(f"ERROR: {API_KEY_ENV} is not set (checked .env)")
        sys.exit(1)

    seasons = args.season or list(FREE_SEASONS)
    unsupported = [s for s in seasons if s not in FREE_SEASONS]
    if unsupported:
        print(
            f"ERROR: the free plan serves only {FREE_SEASONS}; "
            f"asked for {unsupported}. It answers HTTP 200 with an empty "
            f"result for anything else, which would look like a clean pass."
        )
        sys.exit(1)

    config = load_config()
    corpus = StaticFileCorpusSource(config.european.cache_path).load()
    if corpus.empty:
        print("No corpus found. Run: uv run python scripts/build_european_corpus.py")
        sys.exit(1)

    competitions = args.competition or sorted(LEAGUE_IDS)
    corpus["_season"] = corpus["Season"].map(_season_of)

    all_discrepancies: list[Discrepancy] = []
    totals: dict[str, int] = defaultdict(int)

    for competition in competitions:
        for season in seasons:
            subset = corpus[
                (corpus["Div"] == competition) & (corpus["_season"] == season)
            ]
            if subset.empty:
                continue
            print(
                f"\n{competition} {season}-{str(season + 1)[2:]}: {len(subset)} corpus matches"
            )
            try:
                fixtures = fetch_fixtures(api_key, LEAGUE_IDS[competition], season)
            except Exception as exc:
                print(f"    SKIPPED — {exc}")
                continue

            index = _api_index(fixtures)
            discrepancies, compared = compare(subset, index, competition, season)
            all_discrepancies.extend(discrepancies)
            totals["compared"] += compared
            totals["unmatched"] += len(subset) - compared
            print(
                f"    {compared} matched by name, "
                f"{len(subset) - compared} not found, "
                f"{len(discrepancies)} disagree"
            )

    print(f"\n{'=' * 60}")
    print(
        f"Compared {totals['compared']} matches "
        f"({totals['unmatched']} could not be paired by name)"
    )
    # Extra-time ties are expected to differ: the API reports the score after
    # 120 minutes, the corpus the score at 90, matching football-data's
    # FTHG/FTAG semantics. Separating them is the difference between a report
    # a human must adjudicate and one that answers the question by itself.
    expected = [d for d in all_discrepancies if d.extra_time]
    unexplained = [d for d in all_discrepancies if not d.extra_time]

    if expected:
        print(
            f"\n{len(expected)} differ only because the tie went past 90 minutes "
            f"(corpus keeps the 90-minute score, by design — see "
            f"docs/european-competitions.md §3):\n"
        )
        for item in expected:
            print(
                f"  {item.competition} {item.season}  {item.home} vs {item.away}: "
                f"90' {item.corpus_score}, after extra time {item.api_score}"
            )

    if not unexplained:
        print(
            f"\n✓ No unexplained disagreements. The parser agrees with an "
            f"independent source on every one of the "
            f"{totals['compared'] - len(expected)} matches decided in normal time."
        )
        return

    print(f"\n✗ {len(unexplained)} UNEXPLAINED disagreements:\n")
    for item in unexplained:
        print(
            f"  {item.competition} {item.season}  {item.home} vs {item.away}: "
            f"corpus {item.corpus_score}, API-Football {item.api_score}"
        )
    sys.exit(1)


if __name__ == "__main__":
    main()
