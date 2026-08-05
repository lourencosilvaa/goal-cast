"""Error paths and boundary conditions in canonical-name resolution.

These cover the branches that only fire when something upstream is broken —
a dead alias source, an all-zero scoring history, an unplayable fixture — where
the required behaviour is to keep answering rather than to raise.
"""

import pandas as pd

from config.config_loader import InsightsConfig, TeamAliasConfig
from src.analysis.team_insights import (
    EmpiricalPoissonModel,
    FixtureQuery,
    TeamInsightsCalculator,
    _poisson_prob,
)
from src.teams.resolver import (
    ChainedTeamAliasRepository,
    NullUnresolvedNameSink,
    Resolution,
    TeamAlias,
    TeamAliasRepository,
)

CONFIG = TeamAliasConfig(
    seed_path="config/team_aliases.yaml", suggestion_count=5, suggestion_cutoff=0.4
)


class WorkingRepository(TeamAliasRepository):
    def get_aliases(self) -> list[TeamAlias]:
        return [TeamAlias("P1", "Sporting CP", "Sp Lisbon")]


class BrokenRepository(TeamAliasRepository):
    def get_aliases(self) -> list[TeamAlias]:
        raise RuntimeError("alias source down")


class TestChainedTeamAliasRepository:

    def test_collects_from_every_source(self):
        chained = ChainedTeamAliasRepository([WorkingRepository(), WorkingRepository()])
        assert len(chained.get_aliases()) == 2

    def test_a_broken_source_is_skipped(self):
        """One dead source must not cost the aliases the others provide."""
        chained = ChainedTeamAliasRepository([BrokenRepository(), WorkingRepository()])
        assert chained.get_aliases() == [TeamAlias("P1", "Sporting CP", "Sp Lisbon")]

    def test_all_sources_broken_yields_nothing(self):
        chained = ChainedTeamAliasRepository([BrokenRepository(), BrokenRepository()])
        assert chained.get_aliases() == []

    def test_no_sources_yields_nothing(self):
        assert ChainedTeamAliasRepository([]).get_aliases() == []


class TestNullUnresolvedNameSink:

    def test_discards_silently(self):
        sink = NullUnresolvedNameSink()
        assert sink.record(Resolution(raw_name="x", league_code="P1")) is None


class TestPoissonHelper:

    def test_zero_rate_puts_all_mass_on_zero_goals(self):
        assert _poisson_prob(0.0, 0) == 1.0
        assert _poisson_prob(0.0, 3) == 0.0

    def test_negative_rate_is_treated_as_zero(self):
        assert _poisson_prob(-1.0, 0) == 1.0

    def test_positive_rate_is_a_probability(self):
        assert 0.0 < _poisson_prob(1.5, 1) < 1.0


def _frame(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


class TestEmpiricalModelEdges:

    def test_goalless_history_still_yields_positive_expectations(self):
        """Clamping keeps a scoreless run from collapsing the whole market."""
        model = EmpiricalPoissonModel(
            matches=_frame(
                [
                    ("2024-01-01", "Sporting", "Porto", 0, 0),
                    ("2024-01-08", "Porto", "Sporting", 0, 0),
                ]
            ),
            league_average_goals=0.0,
            max_goals=6,
        )
        prediction = model.predict("Sporting", "Porto")
        assert prediction.lambda_home >= EmpiricalPoissonModel.MIN_EXPECTED_GOALS
        assert prediction.lambda_away >= EmpiricalPoissonModel.MIN_EXPECTED_GOALS

    def test_zero_league_average_does_not_divide_by_zero(self):
        model = EmpiricalPoissonModel(
            matches=_frame([("2024-01-01", "Sporting", "Porto", 2, 1)]),
            league_average_goals=0.0,
            max_goals=6,
        )
        assert model.predict("Sporting", "Porto").lambda_home > 0

    def test_team_absent_from_one_venue_falls_back_to_its_overall_rate(self):
        """Sporting have only played away; a home rate must still exist."""
        model = EmpiricalPoissonModel(
            matches=_frame([("2024-01-01", "Porto", "Sporting", 1, 3)]),
            league_average_goals=1.35,
            max_goals=6,
        )
        assert model.predict("Sporting", "Porto").lambda_home > 0

    def test_unknown_team_falls_back_to_the_league_average(self):
        model = EmpiricalPoissonModel(
            matches=_frame([("2024-01-01", "Porto", "Benfica", 1, 1)]),
            league_average_goals=1.35,
            max_goals=6,
        )
        assert model.predict("Nobody", "Porto").lambda_home > 0

    def test_empty_frame_knows_nothing(self):
        model = EmpiricalPoissonModel(
            matches=pd.DataFrame(), league_average_goals=1.35, max_goals=6
        )
        assert model.knows("Sporting") is False


class TestLeagueAverageFallback:

    def test_missing_scores_fall_back_to_the_configured_average(self):
        """A frame of unplayed fixtures has no mean to normalise by."""
        frame = _frame([("2024-01-01", "Sporting", "Porto", None, None)])
        calculator = TeamInsightsCalculator(
            matches=frame,
            config=InsightsConfig(
                recent_matches=5,
                h2h_matches=5,
                form_sequence_length=5,
                max_scorelines=3,
            ),
        )
        average = calculator._league_average_goals(frame)
        assert average == TeamInsightsCalculator.FALLBACK_LEAGUE_AVERAGE_GOALS

    def test_unplayed_fixtures_still_produce_markets(self):
        frame = _frame([("2024-01-01", "Sporting", "Porto", None, None)])
        calculator = TeamInsightsCalculator(
            matches=frame,
            config=InsightsConfig(
                recent_matches=5,
                h2h_matches=5,
                form_sequence_length=5,
                max_scorelines=3,
            ),
        )
        insights = calculator.match_insights(
            FixtureQuery(league_code="P1", home_team="Sporting", away_team="Porto")
        )
        assert insights.goal_markets is not None
