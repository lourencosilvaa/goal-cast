"""The results service's own HTTP surface.

Everything the main app will ever see of this service is here, so these tests
pin the wire contract: the shape of a live payload, the shape of a history
payload, and — as much as either — what happens when it cannot answer. A 200
with an empty list would tell the app "nothing is being played" when the truth
is "every provider is down".
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from src.contracts.results import HistoryResponse, LiveResultsResponse
from src.results_service.app import ServiceRuntime, create_app
from src.results_service.service import (
    HistoryResult,
    UnknownLeagueError,
    UnknownSeasonError,
)
from src.scrapers.results.models import (
    EventType,
    LiveSnapshot,
    LiveUpdate,
    MatchEvent,
    MatchResult,
    MatchStatus,
)

_KEY = "s3cret"
_NOW = datetime(2026, 8, 9, 17, 30)


def _match(status=MatchStatus.LIVE, home_goals=2, away_goals=0) -> MatchResult:
    return MatchResult(
        league="P1",
        kickoff=datetime(2026, 8, 9, 17, 0),
        home_team="Porto",
        away_team="Alverca",
        status=status,
        home_goals=home_goals,
        away_goals=away_goals,
        minute="67'",
        source="football-data.org",
        source_id="567265",
    )


class _Service:
    """A results service whose answers the test dictates."""

    def __init__(self, live=None, history=None, error=None):
        self._live = live
        self._history = history
        self._error = error
        self.live_calls: list[list[str]] = []
        self.history_calls: list[tuple[str, str]] = []

    def live(self, leagues):
        self.live_calls.append(list(leagues))
        if self._error:
            raise self._error
        if self._live is not None:
            return self._live
        return LiveUpdate(
            snapshot=LiveSnapshot(_NOW, (_match(),), "football-data.org"),
            events=(MatchEvent(EventType.GOAL, _match(), _NOW),),
            stale=False,
        )

    def history(self, query):
        self.history_calls.append((query.league, query.season))
        if self._error:
            raise self._error
        if self._history is not None:
            return self._history
        return HistoryResult(
            league=query.league,
            season=query.season,
            source="local-corpus",
            matches=(_match(status=MatchStatus.FINISHED),),
        )


def _client(service=None) -> TestClient:
    return TestClient(
        create_app(ServiceRuntime(service=service or _Service(), api_key=_KEY))
    )


def _auth() -> dict[str, str]:
    return {"X-API-Key": _KEY}


class TestAuthentication:
    def test_live_without_a_key_is_unauthorised(self):
        assert _client().get("/live").status_code == 401

    def test_history_without_a_key_is_unauthorised(self):
        response = _client().get("/history", params={"league": "P1", "season": "2526"})
        assert response.status_code == 401

    def test_a_wrong_key_is_unauthorised(self):
        response = _client().get("/live", headers={"X-API-Key": "nope"})
        assert response.status_code == 401

    def test_an_unauthorised_call_never_reaches_the_service(self):
        """Otherwise an open scraper sits behind a closed door."""
        service = _Service()
        TestClient(create_app(ServiceRuntime(service, _KEY))).get("/live")
        assert service.live_calls == []

    def test_health_needs_no_key(self):
        """Render's health check cannot send one."""
        assert _client().get("/health").status_code == 200


class TestLiveEndpoint:
    def test_a_valid_call_succeeds(self):
        assert _client().get("/live", headers=_auth()).status_code == 200

    def test_the_payload_validates_against_the_shared_contract(self):
        payload = _client().get("/live", headers=_auth()).json()
        assert LiveResultsResponse.model_validate(payload).source == (
            "football-data.org"
        )

    def test_matches_carry_scores_status_and_minute(self):
        payload = _client().get("/live", headers=_auth()).json()
        match = payload["matches"][0]
        assert (match["home_goals"], match["status"], match["minute"]) == (
            2,
            "live",
            "67'",
        )

    def test_a_match_in_play_carries_an_elapsed_minute(self):
        """The in-play prediction divides by it, and the football-data.org
        free tier sends no clock — so the wire must always carry one."""
        payload = _client().get("/live", headers=_auth()).json()
        assert payload["matches"][0]["elapsed_minutes"] > 0

    def test_a_provider_clock_is_not_flagged_as_estimated(self):
        payload = _client().get("/live", headers=_auth()).json()
        assert payload["matches"][0]["minute_estimated"] is False

    def test_a_missing_clock_is_flagged_as_estimated(self):
        """So the UI can show "~57'" instead of passing arithmetic off as the
        official minute."""
        clockless = _match()
        service = _Service(
            live=LiveUpdate(
                snapshot=LiveSnapshot(
                    _NOW,
                    (
                        MatchResult(
                            league=clockless.league,
                            kickoff=clockless.kickoff,
                            home_team=clockless.home_team,
                            away_team=clockless.away_team,
                            status=MatchStatus.LIVE,
                            home_goals=2,
                            away_goals=0,
                            minute="",
                            source="football-data.org",
                        ),
                    ),
                    "football-data.org",
                ),
                events=(),
                stale=False,
            )
        )
        match = _client(service).get("/live", headers=_auth()).json()["matches"][0]
        assert match["minute_estimated"] is True
        assert match["elapsed_minutes"] > 0

    def test_a_finished_match_is_ninety_minutes_and_not_an_estimate(self):
        service = _Service(
            live=LiveUpdate(
                snapshot=LiveSnapshot(
                    _NOW, (_match(status=MatchStatus.FINISHED),), "football-data.org"
                ),
                events=(),
                stale=False,
            )
        )
        match = _client(service).get("/live", headers=_auth()).json()["matches"][0]
        assert (match["elapsed_minutes"], match["minute_estimated"]) == (90, False)

    def test_events_are_included_for_the_ui_to_highlight(self):
        payload = _client().get("/live", headers=_auth()).json()
        assert payload["events"][0]["type"] == "goal"

    def test_staleness_is_reported_rather_than_hidden(self):
        service = _Service(
            live=LiveUpdate(
                snapshot=LiveSnapshot(_NOW, (_match(),), "football-data.org"),
                events=(),
                stale=True,
            )
        )
        payload = _client(service).get("/live", headers=_auth()).json()
        assert payload["stale"] is True

    def test_the_leagues_parameter_is_split_on_commas(self):
        service = _Service()
        _client(service).get("/live", params={"leagues": "E0,P1"}, headers=_auth())
        assert service.live_calls == [["E0", "P1"]]

    def test_blank_entries_in_the_parameter_are_ignored(self):
        service = _Service()
        _client(service).get("/live", params={"leagues": "E0,,P1,"}, headers=_auth())
        assert service.live_calls == [["E0", "P1"]]

    def test_surrounding_whitespace_is_trimmed(self):
        service = _Service()
        _client(service).get("/live", params={"leagues": "E0, P1"}, headers=_auth())
        assert service.live_calls == [["E0", "P1"]]

    def test_omitting_the_parameter_asks_for_everything(self):
        service = _Service()
        _client(service).get("/live", headers=_auth())
        assert service.live_calls == [[]]

    def test_an_unknown_league_is_a_422_not_an_empty_list(self):
        service = _Service(error=UnknownLeagueError("unknown league 'ZZ'"))
        response = _client(service).get(
            "/live", params={"leagues": "ZZ"}, headers=_auth()
        )
        assert response.status_code == 422
        assert "ZZ" in response.json()["detail"]


class TestHistoryEndpoint:
    def _get(self, client, **params):
        query = {"league": "P1", "season": "2526"}
        query.update(params)
        return client.get("/history", params=query, headers=_auth())

    def test_a_valid_call_succeeds(self):
        assert self._get(_client()).status_code == 200

    def test_the_payload_validates_against_the_shared_contract(self):
        payload = self._get(_client()).json()
        assert HistoryResponse.model_validate(payload).league == "P1"

    def test_the_answering_source_is_reported(self):
        assert self._get(_client()).json()["source"] == "local-corpus"

    def test_matches_are_finished(self):
        assert self._get(_client()).json()["matches"][0]["status"] == "finished"

    def test_the_league_and_season_reach_the_service(self):
        service = _Service()
        self._get(_client(service), league="E0", season="2425")
        assert service.history_calls == [("E0", "2425")]

    def test_an_unknown_league_is_a_422(self):
        service = _Service(error=UnknownLeagueError("unknown league 'ZZ'"))
        assert self._get(_client(service), league="ZZ").status_code == 422

    def test_an_unknown_season_is_a_422(self):
        service = _Service(error=UnknownSeasonError("unknown season '1999'"))
        assert self._get(_client(service), season="1999").status_code == 422

    def test_the_league_parameter_is_required(self):
        response = _client().get(
            "/history", params={"season": "2526"}, headers=_auth()
        )
        assert response.status_code == 422

    def test_the_season_parameter_is_required(self):
        response = _client().get("/history", params={"league": "P1"}, headers=_auth())
        assert response.status_code == 422


class TestUpstreamFailures:
    def test_a_broken_service_is_a_502_not_a_200_with_nothing(self):
        service = _Service(error=RuntimeError("provider exploded"))
        response = _client(service).get("/live", headers=_auth())
        assert response.status_code == 502

    def test_an_http_error_raised_deeper_down_keeps_its_own_status(self):
        """Wrapping it in a 502 would relabel a deliberate status as an
        upstream fault."""
        from fastapi import HTTPException

        service = _Service(error=HTTPException(status_code=429, detail="slow down"))
        assert _client(service).get("/live", headers=_auth()).status_code == 429

    def test_the_failure_says_which_side_broke(self):
        service = _Service(error=RuntimeError("provider exploded"))
        detail = _client(service).get("/live", headers=_auth()).json()["detail"]
        assert "results" in detail.lower()


class TestHealth:
    def test_health_reports_ok(self):
        assert _client().get("/health").json()["status"] == "ok"

    def test_health_reports_the_build_it_is_running(self, monkeypatch):
        monkeypatch.setenv("GIT_SHA", "abc1234")
        assert _client().get("/health").json()["commit"] == "abc1234"

    def test_health_reports_unknown_when_the_build_is_not_stamped(
        self, monkeypatch
    ):
        monkeypatch.delenv("GIT_SHA", raising=False)
        assert _client().get("/health").json()["commit"] == "unknown"

    def test_health_does_not_touch_the_providers(self):
        """A health check that polls upstream would spend quota every 30
        seconds and fail the service when a third party is down."""
        service = _Service(error=RuntimeError("provider exploded"))
        assert _client(service).get("/health").status_code == 200


class TestRuntimeWiring:
    def test_the_app_refuses_to_be_built_without_a_key(self):
        from src.results_service.auth import MissingServiceKeyError

        with pytest.raises(MissingServiceKeyError):
            create_app(ServiceRuntime(service=_Service(), api_key=""))
