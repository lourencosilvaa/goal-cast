"""Deciding that a fixture spans two leagues, and counting the evidence for it.

Both halves are shared between the scheduled job and the HuggingFace Space, so
both live here rather than being written twice — the same reasoning that
produced :mod:`src.models.fixture_features`, and for the same reason: the last
pair of copies disagreed for months without anything noticing.

The routing question is deliberately not "is this a UEFA competition?". A
Liga Portugal side against a Premier League side is a cross-league fixture
whether or not it carries a European badge, and it is the *pairing* that makes
the ensemble invalid — it was trained only on domestic rows and has never seen
one.
"""

import pandas as pd

from src.models.cross_league import is_cross_league, match_counts


class TestRouting:
    def test_two_different_leagues_are_cross_league(self):
        assert is_cross_league("P1", "E0") is True

    def test_the_same_league_is_not(self):
        assert is_cross_league("P1", "P1") is False

    def test_an_unstated_away_league_means_the_same_league(self):
        """The domestic caller sends one league code and nothing else; that
        must keep meaning "both sides play in it"."""
        assert is_cross_league("P1", "") is False

    def test_an_unstated_home_league_is_not_cross_league(self):
        """Nothing to compare against — refusing to guess beats routing a
        request to a model on the strength of a missing field."""
        assert is_cross_league("", "E0") is False

    def test_neither_league_stated_is_not_cross_league(self):
        assert is_cross_league("", "") is False

    def test_league_codes_are_compared_exactly(self):
        """They are identifiers, not free text."""
        assert is_cross_league("P1", "p1") is True

    def test_surrounding_whitespace_does_not_invent_a_second_league(self):
        assert is_cross_league("P1", " P1 ") is False


class TestMatchCounts:
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "HomeTeam": ["Porto", "Benfica", "Porto"],
                "AwayTeam": ["Benfica", "Porto", "Braga"],
            }
        )

    def test_a_team_is_counted_home_and_away(self):
        assert match_counts(self._frame())["Porto"] == 3

    def test_every_team_appears(self):
        assert set(match_counts(self._frame())) == {"Porto", "Benfica", "Braga"}

    def test_a_team_seen_once_is_counted_once(self):
        assert match_counts(self._frame())["Braga"] == 1

    def test_an_empty_corpus_counts_nothing(self):
        assert match_counts(pd.DataFrame()) == {}

    def test_a_frame_without_the_team_columns_counts_nothing(self):
        """A malformed snapshot must not take a prediction down; an empty map
        simply means the evidence gate cannot refuse anyone."""
        assert match_counts(pd.DataFrame({"Date": [1, 2]})) == {}
