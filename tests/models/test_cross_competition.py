"""Tests for cross-league ELO calibration.

This is the point of the whole European track. Each domestic league is a
near-closed pool: teams only ever play in-league opponents, so ELO — which is
zero-sum *within* a pool — leaves every league floating around the same
starting rating regardless of its real strength. Measured on the real corpus,
five leagues sit at a mean of exactly 1500.0.

Only real cross-league matches link the pools. These tests assert that
supplementary European results do that linking, and — just as important — that
they never leak into the ensemble's feature matrix, whose rows must carry the
shot/foul/corner columns the European feed does not have.
"""

import pandas as pd

from config.config_loader import EloConfig
from src.models.cross_competition import (
    CrossCompetitionCorpus,
    CrossCompetitionEloBuilder,
)
from src.models.elo import FootballELO


def _elo_config() -> EloConfig:
    return EloConfig(k_factor=32, home_advantage=65, initial_rating=1500.0)


def _domestic() -> pd.DataFrame:
    """Two leagues that never meet, each with its own dominant side."""
    return pd.DataFrame(
        {
            "Div": ["E0", "E0", "P1", "P1", "E0", "P1"],
            "Date": pd.to_datetime(
                [
                    "2024-08-01",
                    "2024-08-08",
                    "2024-08-02",
                    "2024-08-09",
                    "2024-11-01",
                    "2024-11-02",
                ]
            ),
            "HomeTeam": ["Arsenal", "Arsenal", "Benfica", "Benfica", "Arsenal", "Benfica"],
            "AwayTeam": ["Everton", "Everton", "Boavista", "Boavista", "Everton", "Boavista"],
            "FTHG": [3, 2, 3, 2, 1, 1],
            "FTAG": [0, 1, 0, 1, 0, 0],
            "FTR": ["H", "H", "H", "H", "H", "H"],
            "HS": [18, 15, 17, 14, 12, 13],
        }
    )


def _supplementary() -> pd.DataFrame:
    """One Champions League tie, played between the domestic rounds."""
    return pd.DataFrame(
        {
            "Div": ["CL"],
            "Date": pd.to_datetime(["2024-09-17"]),
            "HomeTeam": ["Benfica"],
            "AwayTeam": ["Arsenal"],
            "FTHG": [5],
            "FTAG": [0],
            "FTR": ["H"],
        }
    )


def _build(supplementary: pd.DataFrame) -> pd.DataFrame:
    builder = CrossCompetitionEloBuilder(FootballELO(_elo_config()))
    return builder.build(
        CrossCompetitionCorpus(domestic=_domestic(), supplementary=supplementary)
    )


class TestSupplementaryRowsStayOutOfTheMatrix:
    """European rows calibrate the ratings; they never become training rows."""

    def test_output_holds_only_domestic_rows(self):
        assert len(_build(_supplementary())) == len(_domestic())

    def test_no_european_competition_appears_in_the_output(self):
        assert "CL" not in set(_build(_supplementary())["Div"])

    def test_domestic_columns_survive(self):
        assert "HS" in _build(_supplementary()).columns

    def test_output_is_chronological(self):
        """Downstream cross-validation splits on time."""
        assert _build(_supplementary())["Date"].is_monotonic_increasing

    def test_both_paths_agree_on_row_order(self):
        with_european = _build(_supplementary())
        without = _build(pd.DataFrame())
        assert list(with_european["HomeTeam"]) == list(without["HomeTeam"])
        assert list(with_european["Date"]) == list(without["Date"])

    def test_no_helper_columns_leak_out(self):
        assert not [c for c in _build(_supplementary()).columns if c.startswith("_")]


class TestEloFeatures:
    def test_elo_columns_are_attached(self):
        result = _build(_supplementary())
        for column in (
            "elo_home",
            "elo_away",
            "elo_diff",
            "elo_expected_home",
            "elo_expected_away",
        ):
            assert column in result.columns

    def test_without_supplementary_matches_plain_elo(self):
        """The no-European path must behave exactly as it always has."""
        plain = FootballELO(_elo_config()).compute_elo_features(_domestic())
        result = _build(pd.DataFrame())
        assert list(result["elo_home"].round(6)) == list(
            plain.sort_values("Date")["elo_home"].round(6)
        )


class TestPoolsBecomeLinked:
    """The crux: a cross-league result must move ratings across pools."""

    def _november_arsenal_rating(self, supplementary: pd.DataFrame) -> float:
        result = _build(supplementary)
        november = result[result["Date"] == pd.Timestamp("2024-11-01")]
        return float(november.iloc[0]["elo_home"])

    def test_a_heavy_european_defeat_lowers_the_domestic_rating(self):
        without = self._november_arsenal_rating(pd.DataFrame())
        with_european = self._november_arsenal_rating(_supplementary())
        assert with_european < without

    def test_the_european_winner_gains(self):
        linked = CrossCompetitionEloBuilder(FootballELO(_elo_config()))
        linked.build(
            CrossCompetitionCorpus(_domestic(), _supplementary())
        )
        isolated = CrossCompetitionEloBuilder(FootballELO(_elo_config()))
        isolated.build(CrossCompetitionCorpus(_domestic(), pd.DataFrame()))
        assert linked.elo.get_rating("Benfica") > isolated.elo.get_rating("Benfica")

    def test_a_club_seen_only_in_europe_still_gets_a_rating(self):
        """Shakhtar has no domestic history but earns one from European play."""
        supplementary = pd.DataFrame(
            {
                "Div": ["CL"],
                "Date": pd.to_datetime(["2024-09-17"]),
                "HomeTeam": ["Arsenal"],
                "AwayTeam": ["FK Shakhtar Donetsk"],
                "FTHG": [1],
                "FTAG": [2],
                "FTR": ["A"],
            }
        )
        builder = CrossCompetitionEloBuilder(FootballELO(_elo_config()))
        builder.build(CrossCompetitionCorpus(_domestic(), supplementary))
        assert (
            builder.elo.get_rating("FK Shakhtar Donetsk")
            > _elo_config().initial_rating
        )


class TestGoalsOnlyCombination:
    """What the Dixon-Coles fit is handed."""

    def _corpus(self) -> CrossCompetitionCorpus:
        return CrossCompetitionCorpus(_domestic(), _supplementary())

    def test_combines_both_corpora(self):
        assert len(self._corpus().combined_goals()) == len(_domestic()) + 1

    def test_is_chronological(self):
        assert self._corpus().combined_goals()["Date"].is_monotonic_increasing

    def test_carries_the_columns_dixon_coles_needs(self):
        combined = self._corpus().combined_goals()
        for column in ("HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"):
            assert column in combined.columns

    def test_without_supplementary_returns_the_domestic_goals(self):
        corpus = CrossCompetitionCorpus(_domestic(), pd.DataFrame())
        assert len(corpus.combined_goals()) == len(_domestic())

    def test_two_empty_corpora_yield_an_empty_frame(self):
        corpus = CrossCompetitionCorpus(pd.DataFrame(), pd.DataFrame())
        combined = corpus.combined_goals()
        assert combined.empty
        assert list(combined.columns) == list(CrossCompetitionCorpus.GOALS_COLUMNS)

    def test_only_supplementary(self):
        corpus = CrossCompetitionCorpus(pd.DataFrame(), _supplementary())
        assert len(corpus.combined_goals()) == 1

    def test_defaults_to_no_supplementary(self):
        corpus = CrossCompetitionCorpus(domestic=_domestic())
        assert corpus.has_supplementary is False
        assert len(corpus.combined_goals()) == len(_domestic())


class TestEdgeCases:
    def test_empty_domestic_yields_empty_output(self):
        builder = CrossCompetitionEloBuilder(FootballELO(_elo_config()))
        result = builder.build(
            CrossCompetitionCorpus(pd.DataFrame(), _supplementary())
        )
        assert result.empty

    def test_supplementary_missing_goals_is_ignored(self):
        result = _build(_supplementary().assign(FTHG=[None]))
        assert len(result) == len(_domestic())

    def test_supplementary_predating_all_domestic_rows_still_applies(self):
        supplementary = _supplementary().assign(Date=pd.to_datetime(["2024-01-01"]))
        result = _build(supplementary)
        first = result[result["Date"] == pd.Timestamp("2024-08-01")].iloc[0]
        assert float(first["elo_home"]) != _elo_config().initial_rating


class TestBuildAllExposesBothSides:
    """The European rows exist, calibrated, and are reachable on request.

    ``build()`` returns domestic rows only, and that contract must not change —
    the training matrix depends on it. But the blend-weight backtest needs the
    *European* rows carrying their pre-match ELO columns, which is exactly what
    ``build()`` throws away. ``build_all()`` hands back both sides so neither
    caller has to recompute anything.

    Recomputing is the trap this avoids: ELO walked over the returned domestic
    frame alone reverts to the uncalibrated ratings, which is a live defect
    pinned in ``test_elo_pool_scope.TestFittedRatingsMustBeReused``.
    """

    def _features(self):
        builder = CrossCompetitionEloBuilder(FootballELO(_elo_config()))
        return builder.build_all(
            CrossCompetitionCorpus(
                domestic=_domestic(), supplementary=_supplementary()
            )
        )

    def test_domestic_side_is_identical_to_build(self):
        """The refactor must not move a single number in the training matrix."""
        pd.testing.assert_frame_equal(
            self._features().domestic, _build(_supplementary())
        )

    def test_supplementary_side_holds_the_european_rows(self):
        assert len(self._features().supplementary) == len(_supplementary())

    def test_supplementary_rows_carry_elo_columns(self):
        supplementary = self._features().supplementary
        assert "elo_expected_home" in supplementary.columns
        assert "elo_home" in supplementary.columns

    def test_the_european_row_sees_ratings_built_from_domestic_form(self):
        """Pre-match, not post-match: the tie is scored on prior evidence only.

        Benfica and Arsenal have each won twice at home by the time they meet
        in September, so neither carries the default rating into the tie.
        """
        row = self._features().supplementary.iloc[0]
        assert float(row["elo_home"]) != _elo_config().initial_rating
        assert float(row["elo_away"]) != _elo_config().initial_rating

    def test_no_helper_columns_leak_into_the_supplementary_side(self):
        columns = self._features().supplementary.columns
        assert CrossCompetitionEloBuilder.SOURCE_COLUMN not in columns
        assert CrossCompetitionEloBuilder.ROW_ID_COLUMN not in columns

    def test_without_supplementary_that_side_is_empty(self):
        builder = CrossCompetitionEloBuilder(FootballELO(_elo_config()))
        features = builder.build_all(
            CrossCompetitionCorpus(domestic=_domestic(), supplementary=pd.DataFrame())
        )
        assert features.supplementary.empty
        pd.testing.assert_frame_equal(features.domestic, _build(pd.DataFrame()))

    def test_empty_domestic_yields_two_empty_sides(self):
        builder = CrossCompetitionEloBuilder(FootballELO(_elo_config()))
        features = builder.build_all(
            CrossCompetitionCorpus(pd.DataFrame(), _supplementary())
        )
        assert features.domestic.empty
        assert features.supplementary.empty
