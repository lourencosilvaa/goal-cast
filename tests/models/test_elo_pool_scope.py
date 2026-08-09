"""ELO must be walked over the same pool at training and at inference.

Training builds ratings over every tracked league at once. An inference path
that loads one league at a time used to be equivalent — domestic leagues are
disjoint pools, so a per-league walk produced bit-identical ratings to a
combined one.

The European corpus ends that equivalence, and that is not a side effect, it is
the mechanism: linking the pools is the whole point of the calibration. But it
means a per-league walk can no longer reproduce the combined result. A team's
opponent in a European tie has a rating built from *its* domestic history, and
a walk that omits that league starts it at the default instead.

``tests/models/test_train_serve_parity.py`` cannot catch this: it feeds the
same domestic frame to both sides and so only ever proved that identical
inputs give identical outputs. These tests vary the input the way the real
inference path does.
"""

import pandas as pd
import pytest

from config.config_loader import EloConfig
from src.models.cross_competition import (
    CrossCompetitionCorpus,
    CrossCompetitionEloBuilder,
)
from src.models.elo import FootballELO


def _elo() -> FootballELO:
    return FootballELO(EloConfig(k_factor=32, home_advantage=65, initial_rating=1500.0))


def _domestic() -> pd.DataFrame:
    """Two leagues whose teams never meet domestically."""
    return pd.DataFrame(
        {
            "Div": ["E0", "E0", "P1", "P1", "E0", "P1"],
            "League": ["E0", "E0", "P1", "P1", "E0", "P1"],
            "Date": pd.to_datetime(
                [
                    "2024-08-01",
                    "2024-08-08",
                    "2024-08-02",
                    "2024-08-09",
                    "2024-10-01",
                    "2024-10-02",
                ]
            ),
            "HomeTeam": [
                "Arsenal",
                "Arsenal",
                "Benfica",
                "Benfica",
                "Arsenal",
                "Benfica",
            ],
            "AwayTeam": [
                "Everton",
                "Everton",
                "Porto",
                "Porto",
                "Everton",
                "Porto",
            ],
            "FTHG": [3, 2, 5, 4, 1, 1],
            "FTAG": [0, 0, 0, 0, 0, 0],
            "FTR": ["H"] * 6,
            "HS": [18, 17, 20, 19, 12, 13],
        }
    )


def _european() -> pd.DataFrame:
    """One tie linking the two pools, in September — between the domestic rounds."""
    return pd.DataFrame(
        {
            "Div": ["CL"],
            "League": ["CL"],
            "Date": pd.to_datetime(["2024-09-17"]),
            "HomeTeam": ["Benfica"],
            "AwayTeam": ["Arsenal"],
            "FTHG": [3],
            "FTAG": [0],
            "FTR": ["H"],
        }
    )


def _build(domestic: pd.DataFrame, european: pd.DataFrame) -> pd.DataFrame:
    return CrossCompetitionEloBuilder(_elo()).build(
        CrossCompetitionCorpus(domestic=domestic, supplementary=european)
    )


def _last_elo(built: pd.DataFrame, team: str) -> float:
    rows = built[(built["HomeTeam"] == team) | (built["AwayTeam"] == team)]
    row = rows.sort_values("Date").iloc[-1]
    return float(row["elo_home"] if row["HomeTeam"] == team else row["elo_away"])


class TestPoolScope:
    def test_one_league_alone_cannot_reproduce_the_combined_rating(self):
        """The defect, stated directly. This is what inference was doing."""
        combined = _build(_domestic(), _european())
        single = _build(_domestic()[lambda d: d["League"] == "E0"], _european())
        assert _last_elo(combined, "Arsenal") != _last_elo(single, "Arsenal")

    def test_without_the_corpus_the_two_agree_exactly(self):
        """Why this was safe before the calibration, and stopped being safe."""
        empty = pd.DataFrame()
        combined = _build(_domestic(), empty)
        single = _build(_domestic()[lambda d: d["League"] == "E0"], empty)
        assert _last_elo(combined, "Arsenal") == _last_elo(single, "Arsenal")

    def test_filtering_after_the_walk_preserves_the_combined_rating(self):
        """The fix: walk every league, then narrow. Order matters, not scope."""
        combined = _build(_domestic(), _european())
        narrowed = combined[combined["League"] == "E0"]
        assert _last_elo(narrowed, "Arsenal") == _last_elo(combined, "Arsenal")

    def test_narrowing_keeps_only_the_requested_league(self):
        combined = _build(_domestic(), _european())
        narrowed = combined[combined["League"] == "E0"]
        assert set(narrowed["League"]) == {"E0"}

    def test_the_opponents_league_is_what_carries_the_information(self):
        """Benfica's rating comes from its P1 form, and that changes Arsenal's.

        Benfica wins the tie 3-0. With P1 in the walk, Benfica arrives having
        beaten Porto 5-0 and 4-0, so it is strongly rated and Arsenal's defeat
        costs it *less*. Omit P1 and Benfica enters at the default 1500, making
        the same defeat look like an upset and taking more off Arsenal.

        That gap is the whole defect: the rating a model was trained on
        depends on a league the inference path never loaded.
        """
        combined = _build(_domestic(), _european())
        without_portugal = _build(
            _domestic()[lambda d: d["League"] == "E0"], _european()
        )
        assert _last_elo(combined, "Arsenal") > _last_elo(without_portugal, "Arsenal")


class TestFittedRatingsMustBeReused:
    """The calibrated ratings live on the injected ELO, not in the output frame.

    ``CrossCompetitionEloBuilder`` returns domestic rows only — European rows
    update the ratings but never come back out. So recomputing ELO from the
    returned frame silently discards every European result and yields exactly
    the uncalibrated ratings the whole corpus exists to correct.

    That is not hypothetical: it produced Celtic 1924.8 and Sp Lisbon 1906.3,
    above Man City, in a full-season run — numbers identical to the
    pre-calibration table in the documentation.
    """

    def test_the_builder_exposes_its_fitted_ratings(self):
        from src.models.elo import FootballELO

        elo = _elo()
        CrossCompetitionEloBuilder(elo).build(
            CrossCompetitionCorpus(domestic=_domestic(), supplementary=_european())
        )
        assert isinstance(elo, FootballELO)
        assert elo.get_rating("Arsenal") != 1500.0

    def test_recomputing_from_the_output_frame_loses_the_calibration(self):
        """The defect, pinned. Reuse the instance; never rebuild from output."""
        from src.models.elo import FootballELO

        fitted = _elo()
        built = CrossCompetitionEloBuilder(fitted).build(
            CrossCompetitionCorpus(domestic=_domestic(), supplementary=_european())
        )
        rebuilt = FootballELO(
            EloConfig(k_factor=32, home_advantage=65, initial_rating=1500.0)
        )
        rebuilt.compute_elo_features(built)
        assert fitted.get_rating("Arsenal") != rebuilt.get_rating("Arsenal")

    def test_rebuilding_matches_the_no_corpus_case_exactly(self):
        """Proving what the lost calibration costs: it reverts entirely."""
        from src.models.elo import FootballELO

        fitted = _elo()
        built = CrossCompetitionEloBuilder(fitted).build(
            CrossCompetitionCorpus(domestic=_domestic(), supplementary=_european())
        )
        rebuilt = FootballELO(
            EloConfig(k_factor=32, home_advantage=65, initial_rating=1500.0)
        )
        rebuilt.compute_elo_features(built)

        uncalibrated = _elo()
        CrossCompetitionEloBuilder(uncalibrated).build(
            CrossCompetitionCorpus(domestic=_domestic(), supplementary=pd.DataFrame())
        )
        assert rebuilt.get_rating("Arsenal") == pytest.approx(
            uncalibrated.get_rating("Arsenal")
        )
