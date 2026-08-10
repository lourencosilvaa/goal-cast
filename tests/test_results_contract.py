"""The contract between two separately deployed services.

The service and the gateway are built, shipped and restarted independently, so
nothing but this holds their agreement in place. The failure it exists to
catch is quiet: one side starts sending a field the other drops, and it shows
up as missing data on a screen rather than as an error anywhere.

The test is the same shape as ``tests/test_hf_space_contract.py`` — take what
one side really produces, and put it through what the other side really parses.
No hand-written JSON in the middle, because a hand-written payload only proves
that the fixture and the parser agree.
"""

from datetime import datetime

from fastapi.testclient import TestClient

from config.config_loader import ResultsGatewayConfig
from src.backend.services.results_gateway import HttpResultsGateway
from src.contracts import results as contract
from src.results_service.app import create_app
from src.results_service.factory import ServiceRuntime
from src.results_service.service import HistoryResult
from src.scrapers.results.models import (
    EventType,
    LiveSnapshot,
    LiveUpdate,
    MatchEvent,
    MatchResult,
    MatchStatus,
)

_KEY = "contract-key"
_NOW = datetime(2026, 8, 9, 17, 30)


def _match(status: MatchStatus = MatchStatus.LIVE) -> MatchResult:
    return MatchResult(
        league="P1",
        kickoff=datetime(2026, 8, 9, 17, 0),
        home_team="Porto",
        away_team="Alverca",
        status=status,
        home_goals=2,
        away_goals=0,
        minute="67'",
        source="football-data.org",
        source_id="567265",
    )


class _Service:
    """Answers with the service's own internal objects, not with JSON."""

    def live(self, leagues):
        return LiveUpdate(
            snapshot=LiveSnapshot(_NOW, (_match(),), "football-data.org"),
            events=(MatchEvent(EventType.GOAL, _match(), _NOW),),
            stale=True,
        )

    def history(self, query):
        return HistoryResult(
            league=query.league,
            season=query.season,
            source="local-corpus",
            matches=(_match(status=MatchStatus.FINISHED),),
        )


class _ClientTransport:
    """Routes the gateway's HTTP calls into the real service application."""

    def __init__(self, client: TestClient) -> None:
        self._client = client

    def get(self, url, params=None, headers=None):
        # The gateway builds an absolute URL; TestClient wants the path.
        path = url.split("://", 1)[-1].split("/", 1)[-1]
        return self._client.get(f"/{path}", params=params, headers=headers)


def _gateway() -> HttpResultsGateway:
    app = create_app(ServiceRuntime(service=_Service(), api_key=_KEY))
    return HttpResultsGateway(
        ResultsGatewayConfig(
            base_url_env="RESULTS_SERVICE_URL",
            api_key_env="RESULTS_SERVICE_API_KEY",
            timeout=10,
        ),
        transport=_ClientTransport(TestClient(app)),
        env={
            "RESULTS_SERVICE_URL": "http://results.internal:8000",
            "RESULTS_SERVICE_API_KEY": _KEY,
        },
    )


class TestLiveEndToEnd:
    def test_the_gateway_parses_what_the_service_produces(self):
        assert _gateway().live(["P1"]).source == "football-data.org"

    def test_a_match_survives_the_round_trip_intact(self):
        match = _gateway().live(["P1"]).matches[0]
        original = _match()
        assert (
            match.league,
            match.kickoff,
            match.home_team,
            match.away_team,
            match.status,
            match.home_goals,
            match.away_goals,
            match.minute,
            match.source_id,
        ) == (
            original.league,
            original.kickoff,
            original.home_team,
            original.away_team,
            original.status.value,
            original.home_goals,
            original.away_goals,
            original.minute,
            original.source_id,
        )

    def test_staleness_survives_the_round_trip(self):
        """The one flag whose loss would be invisible and misleading."""
        assert _gateway().live(["P1"]).stale is True

    def test_events_survive_the_round_trip(self):
        event = _gateway().live(["P1"]).events[0]
        assert (event.type, event.detected_at) == ("goal", _NOW)

    def test_the_service_key_is_what_gets_the_gateway_in(self):
        """If it were not required, this test would pass with the wrong key."""
        from src.backend.services.results_gateway import ResultsServiceUnavailable

        gateway = _gateway()
        gateway._env = {
            "RESULTS_SERVICE_URL": "http://results.internal:8000",
            "RESULTS_SERVICE_API_KEY": "wrong",
        }
        try:
            gateway.live(["P1"])
        except ResultsServiceUnavailable as exc:
            assert "401" in str(exc)
        else:  # pragma: no cover - the assertion above is the outcome
            raise AssertionError("a wrong service key was accepted")


class TestHistoryEndToEnd:
    def test_the_gateway_parses_what_the_service_produces(self):
        response = _gateway().history("P1", "2526")
        assert (response.league, response.season) == ("P1", "2526")

    def test_the_source_survives_the_round_trip(self):
        assert _gateway().history("P1", "2526").source == "local-corpus"

    def test_a_finished_match_keeps_its_score(self):
        match = _gateway().history("P1", "2526").matches[0]
        assert (match.status, match.home_goals, match.away_goals) == (
            "finished",
            2,
            0,
        )


class TestBothSidesUseTheSameModule:
    def test_the_service_responses_come_from_the_shared_contract(self):
        from src.results_service.api import results as service_api

        assert service_api.LiveResultsResponse is contract.LiveResultsResponse

    def test_the_gateway_parses_with_the_shared_contract(self):
        from src.backend.services import results_gateway

        assert results_gateway.LiveResultsResponse is contract.LiveResultsResponse

    def test_the_contract_module_imports_nothing_from_either_service(self):
        """It is copied into both images; a dependency on one would drag that
        service's code into the other's container.

        Parsed rather than grepped: the module's own docstring names both
        services, and it should — explaining why they are separate is the
        point of the file."""
        import ast
        from pathlib import Path

        tree = ast.parse(Path(contract.__file__).read_text())
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)

        assert not [name for name in imported if name.startswith("src.")]
