"""
Measure whether cross-league calibration actually worked.

Report-only: it never mutates config or artefacts.

Two checks, because "the code ran" and "the ratings mean something" are
different claims:

**Stratification.** Without European results every league's ELO drifts around
the same starting value — five of them sit at a mean of exactly 1500.0, because
a closed pool is zero-sum and the arithmetic pins it. After linking, league
means should spread and order sensibly: the Premier League above the
Championship, the top five above the Super League Greece. If they do not
stratify, the linking failed however cleanly the run completed.

**Cross-league discrimination.** The stratification could in principle be
noise. So the same European matches are scored under both rating sets and
compared by log loss and Brier score. A calibrated model should predict actual
cross-league results better than an uncalibrated one.

Usage:
    uv run python scripts/evaluate_calibration.py
    uv run python scripts/evaluate_calibration.py --with-qualifiers
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from config.config_loader import Config, load_config  # noqa: E402
from src.corpus.canonical_names import build_translator  # noqa: E402
from src.corpus.supplementary import StaticFileCorpusSource  # noqa: E402
from src.models.cross_competition import (  # noqa: E402
    CrossCompetitionCorpus,
    CrossCompetitionEloBuilder,
)
from src.models.data_cleaner import DataCleaner  # noqa: E402
from src.models.data_loader import FootballDataLoader  # noqa: E402
from src.models.elo import FootballELO  # noqa: E402

#: Outcome order used by every probability triple here.
OUTCOMES = ("H", "D", "A")

#: Guards log(0) when a model is certain and wrong.
EPSILON = 1e-15


@dataclass(frozen=True)
class Scores:
    """How well a rating set predicted a set of real results."""

    log_loss: float
    brier: float
    matches: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cross-league calibration")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--with-qualifiers",
        action="store_true",
        help="Use the corpus that includes qualifying rounds",
    )
    return parser.parse_args()


def load_european(config: Config, with_qualifiers: bool) -> pd.DataFrame:
    path = (
        config.european.qualifiers_cache_path
        if with_qualifiers
        else config.european.cache_path
    )
    corpus = StaticFileCorpusSource(path).load()
    if corpus.empty:
        return corpus
    return build_translator(config).translate(corpus).frame


def load_domestic(config: Config) -> pd.DataFrame:
    loader = FootballDataLoader(config.data)
    raw = loader.load_all()
    if raw.empty:
        return raw
    return DataCleaner().clean(raw)


#: The loader labels rows with the league *name*; there is no ``Div`` column in
#: the domestic corpus, because ``data.columns_to_keep`` does not request one.
LEAGUE_COLUMN = "League"


def league_of(domestic: pd.DataFrame) -> dict[str, str]:
    """The league each team last played in.

    Teams move between divisions, so the most recent one is the honest label —
    a side relegated three seasons ago should count towards its current league,
    not the one it left.
    """
    stacked = domestic.melt(
        id_vars=[LEAGUE_COLUMN, "Date"],
        value_vars=["HomeTeam", "AwayTeam"],
        value_name="Team",
    )
    grouped = stacked.sort_values("Date").groupby("Team")[LEAGUE_COLUMN]
    return grouped.last().to_dict()


def fit_ratings(domestic: pd.DataFrame, european: pd.DataFrame, config: Config):
    """ELO fitted with and without the European corpus."""
    elo = FootballELO(config.features.elo)
    CrossCompetitionEloBuilder(elo).build(
        CrossCompetitionCorpus(domestic=domestic, supplementary=european)
    )
    return elo


def stratification(elo, teams_league: dict[str, str]) -> pd.DataFrame:
    rows = [
        {"league": teams_league[team], "rating": rating}
        for team, rating in elo.ratings.items()
        if team in teams_league
    ]
    frame = pd.DataFrame(rows)
    summary = frame.groupby("league")["rating"].agg(["mean", "max", "count"])
    summary["name"] = list(summary.index)
    return summary.sort_values("mean", ascending=False)


def _probabilities(elo, home: str, away: str, config: Config) -> np.ndarray:
    """ELO's 1X2 triple for a fixture.

    ELO yields a win expectancy, not three outcomes, so a draw share is carved
    out around parity. The constant is crude but identical for both rating
    sets, so it cannot favour either — this compares ratings, not draw models.
    """
    draw_share = 0.26
    r_home = elo.get_rating(home) + config.features.elo.home_advantage
    r_away = elo.get_rating(away)
    expected = elo.expected_score(r_home, r_away)
    home_p = expected * (1 - draw_share)
    away_p = (1 - expected) * (1 - draw_share)
    return np.array([home_p, draw_share, away_p])


def score(elo, matches: pd.DataFrame, config: Config) -> Scores:
    """Log loss and Brier score over real European results."""
    losses: list[float] = []
    briers: list[float] = []
    for _, row in matches.iterrows():
        probs = _probabilities(elo, row["HomeTeam"], row["AwayTeam"], config)
        probs = probs / probs.sum()
        actual = np.array([1.0 if row["FTR"] == o else 0.0 for o in OUTCOMES])
        losses.append(-float(np.sum(actual * np.log(np.clip(probs, EPSILON, 1)))))
        briers.append(float(np.sum((probs - actual) ** 2)))
    if not losses:
        return Scores(float("nan"), float("nan"), 0)
    return Scores(float(np.mean(losses)), float(np.mean(briers)), len(losses))


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)

    european = load_european(config, args.with_qualifiers)
    if european.empty:
        print(
            "ERROR: no European corpus. " "Run scripts/build_european_corpus.py first."
        )
        sys.exit(1)

    domestic = load_domestic(config)
    if domestic.empty:
        print("ERROR: no domestic data loaded.")
        sys.exit(1)

    print(f"\ndomestic matches: {len(domestic)} | European: {len(european)}")

    teams_league = league_of(domestic)
    linked = fit_ratings(domestic, european, config)
    isolated = fit_ratings(domestic, pd.DataFrame(), config)

    print("\n=== League ELO means ===")
    before = stratification(isolated, teams_league)
    after = stratification(linked, teams_league)
    print(f"{'league':26s} {'before':>8s} {'after':>8s} {'shift':>8s}")
    for code, row in after.iterrows():
        was = before.loc[code, "mean"] if code in before.index else float("nan")
        shift = row["mean"] - was
        print(f"{row['name']:26s} {was:8.0f} {row['mean']:8.0f} {shift:+8.0f}")

    spread_before = before["mean"].max() - before["mean"].min()
    spread_after = after["mean"].max() - after["mean"].min()
    print(f"\nspread of league means: {spread_before:.0f} -> {spread_after:.0f}")
    initial = config.features.elo.initial_rating
    pinned = int((before["mean"].round(1) == initial).sum())
    still = int((after["mean"].round(1) == initial).sum())
    print(f"leagues pinned at the initial rating: {pinned} -> {still}")

    print("\n=== Predicting real European results ===")
    linked_scores = score(linked, european, config)
    isolated_scores = score(isolated, european, config)
    print(f"{'':12s} {'log loss':>10s} {'Brier':>10s}")
    print(
        f"{'uncalibrated':12s} {isolated_scores.log_loss:10.4f} "
        f"{isolated_scores.brier:10.4f}"
    )
    print(
        f"{'calibrated':12s} {linked_scores.log_loss:10.4f} "
        f"{linked_scores.brier:10.4f}"
    )
    delta = isolated_scores.log_loss - linked_scores.log_loss
    print(f"\nlog-loss improvement: {delta:+.4f} over {linked_scores.matches} matches")
    print(
        "Lower is better. A positive improvement means the calibrated ratings "
        "predict actual cross-league results better."
    )
    print(
        "\nNote: these ratings are fitted on the same European matches they are "
        "scored against, so this measures whether the linkage carries signal, "
        "not out-of-sample skill."
    )


if __name__ == "__main__":
    main()
