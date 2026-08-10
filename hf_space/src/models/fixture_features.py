"""The one feature row a trained model needs for a match nobody has played.

Training rows are built by :class:`~src.models.feature_engineer.FeatureEngineer`
from a match that *happened*. A fixture has no result, no statistics and often
no date, so its row has to be assembled from each side's most recent match —
and that assembly is where the subtlety is.

**Two kinds of feature, and only one may be copied.** A per-side feature
(``home_avg_GF``, ``elo_home``) describes one club and can be read off that
club's last match. A pair-dependent feature (``diff_*``, ``elo_diff``,
``xG_diff``, ``form_gap``, every ``h2h_*``, the similarity scores) describes
*this* match-up and must be recomputed from both sides. Copying one of those
from a team's previous game silently attributes a different fixture's numbers
to this one.

That is not hypothetical. Until this module existed there were two
implementations: the scheduled job's, which recomputed them, and the
HuggingFace Space's, which copied everything that did not start with ``home_``
or ``away_`` from the home team's last row. For Estrela vs Sp Lisbon on
2026-08-08 the Space fed the model an ELO gap belonging to Estrela's previous
opponent and predicted a 44% home win where the correct row gave a 74% away
win. One module, used by both, is the fix — the duplication was the bug.

**Orientation.** A club's last match may have been away, in which case its home
statistics live in that row's ``away_*`` columns. Reading the column name
literally would describe its opponent, so every lookup carries which side the
club was on.

Kept dependency-light on purpose: pandas and nothing else. It is mirrored into
``hf_space/`` (see ``tests/test_hf_space_contract.py``), and anything imported
here has to exist in that image too.
"""

from dataclasses import dataclass
from typing import Any, ClassVar, Sequence

import pandas as pd


@dataclass(frozen=True)
class FixtureTeams:
    """The two clubs, in the spelling the corpus uses."""

    home: str
    away: str


@dataclass(frozen=True)
class _SideRow:
    """A club's most recent match, and which side of it the club was on.

    ``was_expected`` answers "was this club playing in the position we are
    building features for?" — home for the home side, away for the away side.
    When it is false, every column has to be read from its mirror.
    """

    row: Any
    was_expected: bool

    def get(self, expected: str, opposite: str) -> float:
        value = self.row.get(expected if self.was_expected else opposite, 0)
        if value is None or pd.isna(value):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


class FixtureFeatureBuilder:
    """Assembles the model's feature row for an unplayed fixture.

    Constructed once per corpus and reused across fixtures: the league averages
    are computed here rather than per call, and they are the fallback for a
    club the corpus has never seen.
    """

    HOME_PREFIX: ClassVar[str] = "home_"
    AWAY_PREFIX: ClassVar[str] = "away_"
    DIFF_PREFIX: ClassVar[str] = "diff_"
    H2H_PREFIX: ClassVar[str] = "h2h_"
    #: Wins are stored against whichever club was at home in that meeting.
    H2H_HOME_WINS: ClassVar[str] = "h2h_home_wins"
    H2H_DRAWS: ClassVar[str] = "h2h_draws"

    def __init__(self, featured_data: pd.DataFrame) -> None:
        self._data = featured_data
        self._league_average = (
            featured_data.mean(numeric_only=True)
            if not featured_data.empty
            else pd.Series(dtype=float)
        )

    def build(
        self, teams: FixtureTeams, feature_names: Sequence[str]
    ) -> dict[str, Any]:
        """Values for ``feature_names``, plus the two team names.

        An empty corpus yields the names alone. The caller decides what that
        means — the scheduled job falls back to pricing the match from odds
        rather than predicting from nothing.
        """
        row: dict[str, Any] = {}
        if not self._data.empty:
            home = self._side(teams.home, expected_home=True)
            away = self._side(teams.away, expected_home=False)
            for name in feature_names:
                row[name] = self._value(name, home, away, teams)
        row["HomeTeam"] = teams.home
        row["AwayTeam"] = teams.away
        return row

    # ── locating each side ───────────────────────────────────────────────

    def _side(self, team: str, expected_home: bool) -> _SideRow:
        rows = self._data[
            (self._data["HomeTeam"] == team) | (self._data["AwayTeam"] == team)
        ]
        if rows.empty:
            # Never seen. League averages are already orientation-neutral, so
            # they are read as if the club were in its expected position.
            return _SideRow(self._league_average, was_expected=True)
        last = rows.iloc[-1]
        column = "HomeTeam" if expected_home else "AwayTeam"
        return _SideRow(last, was_expected=last.get(column, "") == team)

    # ── one feature ──────────────────────────────────────────────────────

    def _value(
        self, name: str, home: _SideRow, away: _SideRow, teams: FixtureTeams
    ) -> float:
        derived = self._DERIVED.get(name)
        if derived is not None:
            value: float = derived(self, home, away)
            return value
        if name.startswith(self.H2H_PREFIX):
            return self._head_to_head(name, teams)
        if name.startswith(self.HOME_PREFIX):
            stat = name[len(self.HOME_PREFIX) :]
            return home.get(f"home_{stat}", f"away_{stat}")
        if name.startswith(self.AWAY_PREFIX):
            stat = name[len(self.AWAY_PREFIX) :]
            return away.get(f"away_{stat}", f"home_{stat}")
        if name.startswith(self.DIFF_PREFIX):
            stat = name[len(self.DIFF_PREFIX) :]
            return home.get(f"home_{stat}", f"away_{stat}") - away.get(
                f"away_{stat}", f"home_{stat}"
            )
        # Anything else — odds probabilities and the like — is a property of
        # the match rather than of a side, and the home team's last row is the
        # only sample available. Coerced defensively: the corpus also carries
        # text columns (Referee, Div, Time), and a caller that passes the whole
        # column list must get a row back rather than a ValueError.
        return self._number(home.row.get(name, 0))

    # ── the pair-dependent families ──────────────────────────────────────

    @staticmethod
    def _pair(home: _SideRow, away: _SideRow, stat: str) -> tuple[float, float]:
        """The same statistic for each side, each read in its own orientation."""
        return (
            home.get(f"home_{stat}", f"away_{stat}"),
            away.get(f"away_{stat}", f"home_{stat}"),
        )

    def _elo_home(self, home: _SideRow, _away: _SideRow) -> float:
        return home.get("elo_home", "elo_away")

    def _elo_away(self, _home: _SideRow, away: _SideRow) -> float:
        return away.get("elo_away", "elo_home")

    def _elo_diff(self, home: _SideRow, away: _SideRow) -> float:
        return self._elo_home(home, away) - self._elo_away(home, away)

    def _elo_expected_home(self, home: _SideRow, _away: _SideRow) -> float:
        return home.get("elo_expected_home", "elo_expected_away")

    def _elo_expected_away(self, _home: _SideRow, away: _SideRow) -> float:
        return away.get("elo_expected_away", "elo_expected_home")

    def _xg_diff(self, home: _SideRow, away: _SideRow) -> float:
        h, a = self._pair(home, away, "xG_rolling")
        return h - a

    def _rest_advantage(self, home: _SideRow, away: _SideRow) -> float:
        h, a = self._pair(home, away, "rest_days")
        return h - a

    def _avg_draw_pct(self, home: _SideRow, away: _SideRow) -> float:
        h, a = self._pair(home, away, "draw_pct")
        return (h + a) / 2

    def _form_gap(self, home: _SideRow, away: _SideRow) -> float:
        """A magnitude: it measures mismatch, not who is better."""
        h, a = self._pair(home, away, "Form")
        return abs(h - a)

    def _attack_similarity(self, home: _SideRow, away: _SideRow) -> float:
        h, a = self._pair(home, away, "avg_GF")
        return 1 / (1 + abs(h - a))

    def _defense_similarity(self, home: _SideRow, away: _SideRow) -> float:
        h, a = self._pair(home, away, "avg_GA")
        return 1 / (1 + abs(h - a))

    def _combined_defensive(self, home: _SideRow, away: _SideRow) -> float:
        h, a = self._pair(home, away, "avg_GA")
        return 1 / (1 + h) + 1 / (1 + a)

    def _is_midweek(self, _home: _SideRow, _away: _SideRow) -> float:
        """Zero: a fixture row carries no kick-off date at this point, and a
        guessed weekday is a fabricated feature.

        A plain method rather than a ``staticmethod`` because the table below
        stores the function and calls it with ``self`` — a staticmethod object
        would refuse the extra argument.
        """
        return 0.0

    #: Feature name → how to derive it. A dispatch table rather than a chain of
    #: ``elif``s so adding a derived feature is one entry, and so the set of
    #: pair-dependent features is enumerable — which is what the two
    #: implementations disagreed about.
    _DERIVED: ClassVar[dict[str, Any]] = {
        "elo_home": _elo_home,
        "elo_away": _elo_away,
        "elo_diff": _elo_diff,
        "elo_expected_home": _elo_expected_home,
        "elo_expected_away": _elo_expected_away,
        "xG_diff": _xg_diff,
        "rest_advantage": _rest_advantage,
        "avg_draw_pct": _avg_draw_pct,
        "form_gap": _form_gap,
        "attack_similarity": _attack_similarity,
        "defense_similarity": _defense_similarity,
        "combined_defensive": _combined_defensive,
        "is_midweek": _is_midweek,
    }

    # ── head to head ─────────────────────────────────────────────────────

    def _head_to_head(self, name: str, teams: FixtureTeams) -> float:
        """The last meeting's stored counts, flipped if it was the reverse tie.

        ``h2h_home_wins`` is recorded against whoever was at home *in that
        match*. Read unflipped, one club's record becomes the other's.
        """
        meetings = self._data[
            (
                (self._data["HomeTeam"] == teams.home)
                & (self._data["AwayTeam"] == teams.away)
            )
            | (
                (self._data["HomeTeam"] == teams.away)
                & (self._data["AwayTeam"] == teams.home)
            )
        ]
        if meetings.empty:
            return 0.0
        last = meetings.iloc[-1]
        if name == self.H2H_HOME_WINS and last.get("HomeTeam") != teams.home:
            wins = self._number(last.get(self.H2H_HOME_WINS))
            draws = self._number(last.get(self.H2H_DRAWS))
            # The remaining share, floored: rounding in the stored counts must
            # not produce a negative win rate.
            return max(0.0, 1 - wins - draws)
        return self._number(last.get(name))

    @staticmethod
    def _number(value: Any) -> float:
        if value is None or pd.isna(value):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
