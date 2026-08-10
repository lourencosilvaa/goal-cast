"""Building the one feature row a trained model needs for an unplayed match.

This is the module that exists because there were two of these. The scheduled
job (``scripts/run_inference.py``) computed it correctly; the HuggingFace Space
had its own version that only understood ``home_``/``away_`` prefixes and took
everything else — every ``diff_*``, every ``elo_*``, every ``h2h_*`` — from the
home team's *previous* match. For Estrela vs Sp Lisbon on 2026-08-08 that put a
rating gap belonging to a different opponent in front of the model and turned a
74% away win into a 44% home win.

So the tests below are not abstract. Each one pins a family of features that
the two implementations disagreed about, and the pair-dependent ones are the
point: a feature that describes *this* match-up must be computed from both
teams, never copied from one.
"""

import pandas as pd
import pytest

from src.models.fixture_features import FixtureFeatureBuilder, FixtureTeams

_HOME = "Estrela"
_AWAY = "Sp Lisbon"
_TEAMS = FixtureTeams(home=_HOME, away=_AWAY)


def _row(**overrides) -> dict:
    """One played match, in the shape the feature engineer leaves behind."""
    row = {
        "HomeTeam": "Estrela",
        "AwayTeam": "Casa Pia",
        "Date": pd.Timestamp("2026-08-01"),
        "home_avg_GF": 1.0,
        "away_avg_GF": 0.8,
        "home_avg_GA": 1.5,
        "away_avg_GA": 1.2,
        "home_Form": 4.0,
        "away_Form": 6.0,
        "home_xG_rolling": 1.1,
        "away_xG_rolling": 0.9,
        "home_rest_days": 7.0,
        "away_rest_days": 4.0,
        "home_draw_pct": 0.30,
        "away_draw_pct": 0.20,
        "diff_avg_GF": 0.2,
        "elo_home": 1450.0,
        "elo_away": 1400.0,
        "elo_diff": 50.0,
        "elo_expected_home": 0.57,
        "elo_expected_away": 0.43,
        "h2h_home_wins": 0.0,
        "h2h_draws": 0.0,
    }
    row.update(overrides)
    return row


def _data(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


#: A corpus where each side last played someone else, at home.
def _two_teams() -> pd.DataFrame:
    return _data(
        _row(
            HomeTeam=_HOME,
            AwayTeam="Casa Pia",
            elo_home=1450.0,
            elo_away=1400.0,
            elo_diff=50.0,
            elo_expected_home=0.57,
            elo_expected_away=0.43,
            home_avg_GF=1.0,
            away_avg_GF=0.8,
            home_Form=4.0,
            home_xG_rolling=1.1,
            home_rest_days=7.0,
            home_draw_pct=0.30,
        ),
        _row(
            HomeTeam=_AWAY,
            AwayTeam="Moreirense",
            Date=pd.Timestamp("2026-08-02"),
            elo_home=1800.0,
            elo_away=1500.0,
            elo_diff=300.0,
            elo_expected_home=0.85,
            elo_expected_away=0.15,
            home_avg_GF=2.4,
            away_avg_GF=1.0,
            home_Form=13.0,
            home_xG_rolling=2.2,
            home_rest_days=3.0,
            home_draw_pct=0.10,
            home_avg_GA=0.6,
        ),
    )


def _build(names, data=None) -> dict:
    return FixtureFeatureBuilder(data if data is not None else _two_teams()).build(
        _TEAMS, names
    )


class TestPerSideFeatures:
    def test_a_home_feature_comes_from_the_home_team(self):
        assert _build(["home_avg_GF"])["home_avg_GF"] == 1.0

    def test_an_away_feature_comes_from_the_away_team(self):
        """Sporting's own attack, not whatever Estrela last conceded to."""
        assert _build(["away_avg_GF"])["away_avg_GF"] == 2.4

    def test_the_teams_are_carried_on_the_row(self):
        row = _build(["home_avg_GF"])
        assert (row["HomeTeam"], row["AwayTeam"]) == (_HOME, _AWAY)


class TestOrientationFlip:
    """A team's last match may have been away. Its home statistics then live in
    that row's ``away_*`` columns, and reading ``home_*`` would describe its
    opponent."""

    def test_a_home_team_whose_last_match_was_away_is_read_from_the_away_columns(
        self,
    ):
        data = _data(
            _row(HomeTeam="Braga", AwayTeam=_HOME, home_avg_GF=3.0, away_avg_GF=0.4),
            _row(HomeTeam=_AWAY, AwayTeam="Moreirense", home_avg_GF=2.4),
        )
        assert _build(["home_avg_GF"], data)["home_avg_GF"] == 0.4

    def test_an_away_team_whose_last_match_was_at_home_is_read_from_the_home_columns(
        self,
    ):
        assert _build(["away_avg_GF"])["away_avg_GF"] == 2.4


class TestPairDependentFeatures:
    """The bug. Every feature here describes the match-up, so copying it from
    one team's previous game attributes another fixture's numbers to this one."""

    def test_a_diff_feature_is_recomputed_for_this_pair(self):
        # Estrela 1.0 − Sporting 2.4, not the 0.2 sitting in Estrela's last row.
        assert _build(["diff_avg_GF"])["diff_avg_GF"] == pytest.approx(-1.4)

    def test_the_away_elo_is_the_away_team_s_own_rating(self):
        # 1800 is Sporting's; 1400 was Casa Pia's, and is what the Space used.
        assert _build(["elo_away"])["elo_away"] == 1800.0

    def test_the_home_elo_is_the_home_team_s_own_rating(self):
        assert _build(["elo_home"])["elo_home"] == 1450.0

    def test_the_elo_gap_is_recomputed_for_this_pair(self):
        assert _build(["elo_diff"])["elo_diff"] == pytest.approx(1450.0 - 1800.0)

    def test_the_elo_gap_favours_the_stronger_side(self):
        """The sign is the whole prediction: a wrong one flips the outcome."""
        assert _build(["elo_diff"])["elo_diff"] < 0

    def test_each_expected_score_comes_from_its_own_side(self):
        """Each side's own expected score from its own last match, read in its
        own orientation — Sporting played at home, so its 0.85 is mirrored in
        rather than Casa Pia's 0.43 being copied across."""
        row = _build(["elo_expected_home", "elo_expected_away"])
        assert (row["elo_expected_home"], row["elo_expected_away"]) == (0.57, 0.85)

    def test_the_expected_goals_gap_is_recomputed(self):
        assert _build(["xG_diff"])["xG_diff"] == pytest.approx(1.1 - 2.2)

    def test_the_rest_advantage_is_recomputed(self):
        assert _build(["rest_advantage"])["rest_advantage"] == pytest.approx(7.0 - 3.0)

    def test_the_average_draw_rate_uses_both_sides(self):
        assert _build(["avg_draw_pct"])["avg_draw_pct"] == pytest.approx(0.20)

    def test_the_form_gap_is_a_magnitude(self):
        """It measures mismatch, so it must not change sign with the fixture."""
        assert _build(["form_gap"])["form_gap"] == pytest.approx(9.0)

    def test_attack_similarity_falls_as_the_two_attacks_diverge(self):
        assert _build(["attack_similarity"])["attack_similarity"] == pytest.approx(
            1 / (1 + abs(1.0 - 2.4))
        )

    def test_defense_similarity_uses_both_defences(self):
        assert _build(["defense_similarity"])["defense_similarity"] == pytest.approx(
            1 / (1 + abs(1.5 - 0.6))
        )

    def test_combined_defensive_sums_both_sides(self):
        assert _build(["combined_defensive"])["combined_defensive"] == pytest.approx(
            1 / (1 + 1.5) + 1 / (1 + 0.6)
        )


class TestHeadToHead:
    def test_h2h_features_come_from_a_previous_meeting(self):
        data = _data(
            _row(HomeTeam=_HOME, AwayTeam=_AWAY, h2h_home_wins=1.0, h2h_draws=0.0),
            _row(HomeTeam=_AWAY, AwayTeam="Moreirense"),
        )
        assert _build(["h2h_home_wins"], data)["h2h_home_wins"] == 1.0

    def test_h2h_home_wins_is_inverted_when_the_meeting_was_the_reverse_fixture(self):
        """Stored per that match's home side. Read unflipped, one club's record
        becomes the other's."""
        data = _data(
            _row(HomeTeam=_AWAY, AwayTeam=_HOME, h2h_home_wins=0.75, h2h_draws=0.0),
            _row(HomeTeam=_HOME, AwayTeam="Casa Pia"),
        )
        assert _build(["h2h_home_wins"], data)["h2h_home_wins"] == pytest.approx(0.25)

    def test_an_inversion_never_goes_negative(self):
        data = _data(
            _row(HomeTeam=_AWAY, AwayTeam=_HOME, h2h_home_wins=0.8, h2h_draws=0.5),
            _row(HomeTeam=_HOME, AwayTeam="Casa Pia"),
        )
        assert _build(["h2h_home_wins"], data)["h2h_home_wins"] == 0

    def test_teams_that_have_never_met_score_zero(self):
        assert _build(["h2h_home_wins"])["h2h_home_wins"] == 0

    def test_other_h2h_features_are_read_as_they_stand(self):
        data = _data(
            _row(HomeTeam=_HOME, AwayTeam=_AWAY, h2h_draws=0.4),
            _row(HomeTeam=_AWAY, AwayTeam="Moreirense"),
        )
        assert _build(["h2h_draws"], data)["h2h_draws"] == pytest.approx(0.4)


class TestUnknownTeams:
    def test_an_unseen_home_team_falls_back_to_league_averages(self):
        builder = FixtureFeatureBuilder(_two_teams())
        row = builder.build(FixtureTeams(home="Nobody FC", away=_AWAY), ["home_avg_GF"])
        assert row["home_avg_GF"] == pytest.approx(_two_teams()["home_avg_GF"].mean())

    def test_an_unseen_away_team_falls_back_to_league_averages(self):
        builder = FixtureFeatureBuilder(_two_teams())
        row = builder.build(FixtureTeams(home=_HOME, away="Nobody FC"), ["away_avg_GF"])
        assert row["away_avg_GF"] == pytest.approx(_two_teams()["away_avg_GF"].mean())

    def test_an_empty_corpus_yields_only_the_team_names(self):
        row = FixtureFeatureBuilder(pd.DataFrame()).build(_TEAMS, ["home_avg_GF"])
        assert row == {"HomeTeam": _HOME, "AwayTeam": _AWAY}


class TestFallbacks:
    def test_a_midweek_flag_is_zero_without_a_kick_off_date(self):
        """An unplayed fixture has no date here; guessing one would be worse
        than a stated default."""
        assert _build(["is_midweek"])["is_midweek"] == 0

    def test_an_unrecognised_feature_comes_from_the_home_row(self):
        data = _data(
            _row(HomeTeam=_HOME, AwayTeam="Casa Pia", odds_prob_H=0.42),
            _row(HomeTeam=_AWAY, AwayTeam="Moreirense", odds_prob_H=0.9),
        )
        assert _build(["odds_prob_H"], data)["odds_prob_H"] == 0.42

    def test_a_missing_column_becomes_zero(self):
        assert _build(["not_a_column"])["not_a_column"] == 0

    def test_a_non_numeric_column_becomes_zero(self):
        """The corpus carries text columns — Referee, Div, Time. A model would
        never list one as a feature, but a caller passing every column must get
        a usable row rather than a ValueError deep in the loop."""
        data = _data(
            _row(HomeTeam=_HOME, AwayTeam="Casa Pia", Referee="Sp Braga"),
            _row(HomeTeam=_AWAY, AwayTeam="Moreirense", Referee="A Nobody"),
        )
        assert _build(["Referee"], data)["Referee"] == 0

    def test_a_nan_becomes_zero(self):
        data = _data(
            _row(HomeTeam=_HOME, AwayTeam="Casa Pia", home_avg_GF=float("nan")),
            _row(HomeTeam=_AWAY, AwayTeam="Moreirense"),
        )
        assert _build(["home_avg_GF"], data)["home_avg_GF"] == 0

    def test_every_requested_feature_is_present(self):
        names = ["home_avg_GF", "away_avg_GF", "diff_avg_GF", "elo_diff", "is_midweek"]
        assert set(_build(names)) == set(names) | {"HomeTeam", "AwayTeam"}
