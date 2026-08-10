"""football-data.org as a results source, history and live.

The payloads here are recorded from the real API (``samples/``), not invented:
the free tier returns ``LIVE`` where the documentation says ``IN_PLAY``, and it
sends no ``minute`` at all. Both were discovered by calling it, and a
hand-written fixture would have hidden them.
"""

import json
from datetime import date, datetime
from pathlib import Path

from config.config_loader import ResultsProviderConfig
from src.scrapers.results.football_data import (
    FootballDataHistoryProvider,
    FootballDataLiveProvider,
)
from src.scrapers.results.models import HistoryQuery, MatchStatus

SAMPLES = Path(__file__).parent / "samples"


def _live_payload() -> dict:
    return json.loads((SAMPLES / "football_data_live.json").read_text())


def _config(**overrides) -> ResultsProviderConfig:
    defaults = dict(
        enabled=True,
        base_url="https://api.football-data.org/v4",
        api_key_env="FOOTBALL_DATA_API_KEY",
        timeout=15,
        competitions={"P1": "PPL", "N1": "DED", "E0": "PL"},
    )
    defaults.update(overrides)
    return ResultsProviderConfig(**defaults)


class _Response:
    def __init__(self, body, status_code: int = 200, headers: dict | None = None):
        self._body = body
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _Transport:
    """Records every call so quota-shaped assertions are possible."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None, headers=None):
        self.calls.append((url, dict(params or {})))
        response = self._responses.pop(0) if self._responses else _Response({})
        if isinstance(response, Exception):
            raise response
        return response


class TestLiveOneRequestPerDay:
    def test_a_single_request_covers_every_requested_league(self):
        """Ten requests a minute is the whole budget: one call must answer for
        all leagues, not one call per league."""
        transport = _Transport(_Response(_live_payload()))
        provider = FootballDataLiveProvider(
            _config(), transport, "key", today=lambda: date(2026, 8, 9)
        )
        provider.fetch_live(["P1", "N1"])
        assert len(transport.calls) == 1

    def test_the_request_asks_for_todays_matches(self):
        transport = _Transport(_Response(_live_payload()))
        provider = FootballDataLiveProvider(
            _config(), transport, "key", today=lambda: date(2026, 8, 9)
        )
        provider.fetch_live(["P1"])
        url, params = transport.calls[0]
        assert url.endswith("/matches")
        assert params["date"] == "2026-08-09"

    def test_the_key_travels_in_the_auth_header(self):
        class _Capture(_Transport):
            def get(self, url, params=None, headers=None):
                self.headers = dict(headers or {})
                return super().get(url, params, headers)

        transport = _Capture(_Response(_live_payload()))
        FootballDataLiveProvider(
            _config(), transport, "secret", today=lambda: date(2026, 8, 9)
        ).fetch_live(["P1"])
        assert transport.headers["X-Auth-Token"] == "secret"


class TestLiveParsing:
    def _matches(self, leagues=("P1", "N1")):
        transport = _Transport(_Response(_live_payload()))
        provider = FootballDataLiveProvider(
            _config(), transport, "key", today=lambda: date(2026, 8, 9)
        )
        return provider.fetch_live(list(leagues))

    def test_matches_come_back_for_the_requested_leagues(self):
        assert {m.league for m in self._matches()} == {"P1", "N1"}

    def test_a_league_that_was_not_asked_for_is_dropped(self):
        assert {m.league for m in self._matches(["P1"])} == {"P1"}

    def test_a_live_match_is_mapped_to_the_live_status(self):
        live = [m for m in self._matches() if m.status is MatchStatus.LIVE]
        assert [m.home_team for m in live] == ["Porto"]

    def test_a_live_match_carries_its_current_score(self):
        live = next(m for m in self._matches() if m.status is MatchStatus.LIVE)
        assert (live.home_goals, live.away_goals) == (2, 0)

    def test_the_free_tier_sends_no_minute_and_none_is_invented(self):
        """Recorded from the live API: there is no ``minute`` field. Deriving
        one from kickoff would be a guess presented as a fact."""
        live = next(m for m in self._matches() if m.status is MatchStatus.LIVE)
        assert live.minute == ""

    def test_a_timed_match_is_scheduled_and_has_no_goals(self):
        scheduled = [m for m in self._matches() if m.status is MatchStatus.SCHEDULED]
        assert scheduled and all(m.home_goals is None for m in scheduled)

    def test_a_finished_match_keeps_its_full_time_score(self):
        finished = [m for m in self._matches() if m.status is MatchStatus.FINISHED]
        assert finished and all(m.home_goals is not None for m in finished)

    def test_the_short_name_is_preferred_over_the_full_name(self):
        """"Porto" is the canonical spelling; "FC Porto" would need an alias."""
        live = next(m for m in self._matches() if m.status is MatchStatus.LIVE)
        assert live.home_team == "Porto"

    def test_kickoff_is_parsed_as_a_naive_utc_datetime(self):
        live = next(m for m in self._matches() if m.status is MatchStatus.LIVE)
        assert live.kickoff == datetime(2026, 8, 9, 17, 0)

    def test_every_match_records_the_provider_that_supplied_it(self):
        assert {m.source for m in self._matches()} == {"football-data.org"}


class TestLiveFailureModes:
    def _provider(self, *responses):
        transport = _Transport(*responses)
        return (
            FootballDataLiveProvider(
                _config(), transport, "key", today=lambda: date(2026, 8, 9)
            ),
            transport,
        )

    def test_a_transport_error_yields_nothing_rather_than_raising(self):
        provider, _ = self._provider(RuntimeError("connection reset"))
        assert provider.fetch_live(["P1"]) == []

    def test_a_server_error_yields_nothing(self):
        provider, _ = self._provider(_Response({}, status_code=500))
        assert provider.fetch_live(["P1"]) == []

    def test_an_unreadable_body_yields_nothing(self):
        provider, _ = self._provider(_Response(ValueError("not json")))
        assert provider.fetch_live(["P1"]) == []

    def test_a_league_with_no_mapping_is_never_requested(self):
        provider, transport = self._provider()
        assert provider.fetch_live(["SC0"]) == []
        assert transport.calls == []

    def test_a_provider_without_a_key_is_disabled(self):
        provider = FootballDataLiveProvider(
            _config(), _Transport(), "", today=lambda: date(2026, 8, 9)
        )
        assert provider.enabled is False

    def test_a_disabled_provider_makes_no_request(self):
        transport = _Transport(_Response(_live_payload()))
        provider = FootballDataLiveProvider(
            _config(enabled=False), transport, "key", today=lambda: date(2026, 8, 9)
        )
        assert provider.fetch_live(["P1"]) == []
        assert transport.calls == []

    def test_quota_headers_are_recorded(self):
        provider, _ = self._provider(
            _Response(_live_payload(), headers={"x-requests-available-minute": "7"})
        )
        provider.fetch_live(["P1"])
        assert provider.quota.remaining == 7


class TestHistory:
    def _payload(self):
        return {
            "matches": [
                {
                    "id": 1,
                    "utcDate": "2025-08-08T19:15:00Z",
                    "status": "FINISHED",
                    "homeTeam": {"name": "Casa Pia AC", "shortName": "Casa Pia"},
                    "awayTeam": {"name": "Sporting CP", "shortName": "Sporting CP"},
                    "score": {"fullTime": {"home": 0, "away": 2}},
                },
                {
                    "id": 2,
                    "utcDate": "2025-08-10T19:15:00Z",
                    "status": "FINISHED",
                    "homeTeam": {"name": "FC Porto", "shortName": "Porto"},
                    "awayTeam": {"name": "SL Benfica", "shortName": "Benfica"},
                    "score": {"fullTime": {"home": 1, "away": 1}},
                },
            ]
        }

    def _provider(self, *responses):
        transport = _Transport(*responses)
        return FootballDataHistoryProvider(_config(), transport, "key"), transport

    def test_the_season_code_is_translated_to_a_start_year(self):
        provider, transport = self._provider(_Response(self._payload()))
        provider.fetch_history(HistoryQuery(league="P1", season="2526"))
        _, params = transport.calls[0]
        assert params["season"] == 2025

    def test_only_finished_matches_are_requested(self):
        provider, transport = self._provider(_Response(self._payload()))
        provider.fetch_history(HistoryQuery(league="P1", season="2526"))
        _, params = transport.calls[0]
        assert params["status"] == "FINISHED"

    def test_the_competition_code_is_taken_from_configuration(self):
        provider, transport = self._provider(_Response(self._payload()))
        provider.fetch_history(HistoryQuery(league="P1", season="2526"))
        assert transport.calls[0][0].endswith("/competitions/PPL/matches")

    def test_matches_are_returned_finished_and_scored(self):
        provider, _ = self._provider(_Response(self._payload()))
        matches = provider.fetch_history(HistoryQuery(league="P1", season="2526"))
        assert [(m.home_goals, m.away_goals) for m in matches] == [(0, 2), (1, 1)]

    def test_matches_are_ordered_by_kickoff(self):
        payload = self._payload()
        payload["matches"].reverse()
        provider, _ = self._provider(_Response(payload))
        matches = provider.fetch_history(HistoryQuery(league="P1", season="2526"))
        assert [m.kickoff for m in matches] == sorted(m.kickoff for m in matches)

    def test_an_unmapped_league_is_never_requested(self):
        provider, transport = self._provider()
        assert provider.fetch_history(HistoryQuery(league="G1", season="2526")) == []
        assert transport.calls == []

    def test_a_malformed_season_is_refused_without_a_request(self):
        provider, transport = self._provider()
        assert provider.fetch_history(HistoryQuery(league="P1", season="next")) == []
        assert transport.calls == []


class TestUncoveredCompetitions:
    """403/404 mean "your plan does not include this" — a fact about the plan,
    not a transient failure. Repeating it burns the 10/minute allowance to
    receive the same answer."""

    def test_a_restricted_competition_returns_nothing(self):
        transport = _Transport(_Response({}, status_code=403))
        provider = FootballDataHistoryProvider(_config(), transport, "key")
        assert provider.fetch_history(HistoryQuery(league="P1", season="2526")) == []

    def test_a_restricted_competition_is_not_requested_again(self):
        transport = _Transport(
            _Response({}, status_code=403), _Response({"matches": []})
        )
        provider = FootballDataHistoryProvider(_config(), transport, "key")
        provider.fetch_history(HistoryQuery(league="P1", season="2526"))
        provider.fetch_history(HistoryQuery(league="P1", season="2425"))
        assert len(transport.calls) == 1

    def test_an_unknown_competition_is_recorded_the_same_way(self):
        transport = _Transport(_Response({}, status_code=404))
        provider = FootballDataHistoryProvider(_config(), transport, "key")
        provider.fetch_history(HistoryQuery(league="P1", season="2526"))
        assert "P1" in provider.uncovered

    def test_an_empty_answer_does_not_disable_the_competition(self):
        """Out of season is temporary; a plan restriction is not."""
        transport = _Transport(_Response({"matches": []}), _Response({"matches": []}))
        provider = FootballDataHistoryProvider(_config(), transport, "key")
        provider.fetch_history(HistoryQuery(league="P1", season="2627"))
        provider.fetch_history(HistoryQuery(league="P1", season="2526"))
        assert len(transport.calls) == 2
