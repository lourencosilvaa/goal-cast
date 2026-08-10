"""The main app's client for the results service.

An interface plus one HTTP implementation. The interface is what makes the
microservice decision reversible: an in-process gateway calling the provider
chain directly would be a second implementation, not a rewrite of the router.

Everything here is about the failure side, because that is where a gateway
earns its keep. A results service that is asleep, slow, misconfigured or
answering nonsense must produce a *stated* failure — never an empty list that
the UI would render as "no matches today".
"""

import pytest

from config.config_loader import ResultsGatewayConfig
from src.backend.services.results_gateway import (
    HttpResultsGateway,
    MissingGatewayConfigError,
    ResultsGateway,
    ResultsServiceUnavailable,
)

_CONFIG = ResultsGatewayConfig(
    enabled=True,
    base_url_env="RESULTS_SERVICE_URL",
    api_key_env="RESULTS_SERVICE_API_KEY",
    timeout=10,
)
_ENV = {
    "RESULTS_SERVICE_URL": "http://results.internal:8000",
    "RESULTS_SERVICE_API_KEY": "svc-key",
}

_LIVE_BODY = {
    "fetched_at": "2026-08-09T17:30:00",
    "source": "football-data.org",
    "stale": False,
    "matches": [
        {
            "league": "P1",
            "kickoff": "2026-08-09T17:00:00",
            "home_team": "Porto",
            "away_team": "Alverca",
            "status": "live",
            "home_goals": 2,
            "away_goals": 0,
            "minute": "67'",
            "source": "football-data.org",
            "source_id": "567265",
        }
    ],
    "events": [],
}

_HISTORY_BODY = {
    "league": "P1",
    "season": "2526",
    "source": "local-corpus",
    "matches": [],
}


class _Response:
    def __init__(self, body, status_code: int = 200):
        self._body = body
        self.status_code = status_code
        self.headers: dict[str, str] = {}

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _Transport:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls: list[tuple[str, dict, dict]] = []

    def get(self, url, params=None, headers=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        response = self._responses.pop(0) if self._responses else _Response({})
        if isinstance(response, Exception):
            raise response
        return response


def _gateway(*responses, env=None) -> HttpResultsGateway:
    return HttpResultsGateway(
        _CONFIG, _Transport(*responses), env=env if env is not None else _ENV
    )


class TestInterface:
    def test_the_http_gateway_implements_the_interface(self):
        """Nothing above the gateway may depend on it being HTTP."""
        assert issubclass(HttpResultsGateway, ResultsGateway)

    def test_the_interface_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            ResultsGateway()  # type: ignore[abstract]


class TestConfiguration:
    def test_the_base_url_comes_from_the_named_variable(self):
        gateway = _gateway(_Response(_LIVE_BODY))
        gateway.live(["P1"])
        assert gateway._transport.calls[0][0].startswith(
            "http://results.internal:8000"
        )

    def test_a_trailing_slash_on_the_base_url_does_not_double_up(self):
        gateway = HttpResultsGateway(
            _CONFIG,
            _Transport(_Response(_LIVE_BODY)),
            env={**_ENV, "RESULTS_SERVICE_URL": "http://results.internal:8000/"},
        )
        gateway.live(["P1"])
        assert "//live" not in gateway._transport.calls[0][0]

    def test_the_service_key_travels_in_the_header(self):
        gateway = _gateway(_Response(_LIVE_BODY))
        gateway.live(["P1"])
        assert gateway._transport.calls[0][2]["X-API-Key"] == "svc-key"

    def test_an_unset_url_is_a_configuration_error_not_a_request(self):
        gateway = _gateway(env={"RESULTS_SERVICE_API_KEY": "svc-key"})
        with pytest.raises(MissingGatewayConfigError, match="RESULTS_SERVICE_URL"):
            gateway.live(["P1"])

    def test_an_unset_key_is_a_configuration_error(self):
        gateway = _gateway(env={"RESULTS_SERVICE_URL": "http://x"})
        with pytest.raises(MissingGatewayConfigError, match="RESULTS_SERVICE_API_KEY"):
            gateway.live(["P1"])

    def test_a_misconfigured_gateway_makes_no_request(self):
        gateway = _gateway(env={})
        with pytest.raises(MissingGatewayConfigError):
            gateway.live(["P1"])
        assert gateway._transport.calls == []


class TestLive:
    def test_a_successful_call_returns_the_parsed_contract(self):
        response = _gateway(_Response(_LIVE_BODY)).live(["P1"])
        assert response.matches[0].home_goals == 2

    def test_the_requested_leagues_are_sent_as_one_parameter(self):
        gateway = _gateway(_Response(_LIVE_BODY))
        gateway.live(["E0", "P1"])
        assert gateway._transport.calls[0][1]["leagues"] == "E0,P1"

    def test_asking_for_nothing_sends_no_league_filter(self):
        gateway = _gateway(_Response(_LIVE_BODY))
        gateway.live([])
        assert gateway._transport.calls[0][1].get("leagues", "") == ""

    def test_staleness_is_carried_through_rather_than_swallowed(self):
        body = {**_LIVE_BODY, "stale": True}
        assert _gateway(_Response(body)).live(["P1"]).stale is True


class TestHistory:
    def test_a_successful_call_returns_the_parsed_contract(self):
        response = _gateway(_Response(_HISTORY_BODY)).history("P1", "2526")
        assert (response.league, response.season) == ("P1", "2526")

    def test_the_league_and_season_are_sent_as_parameters(self):
        gateway = _gateway(_Response(_HISTORY_BODY))
        gateway.history("P1", "2526")
        assert gateway._transport.calls[0][1] == {"league": "P1", "season": "2526"}

    def test_the_history_path_is_used(self):
        gateway = _gateway(_Response(_HISTORY_BODY))
        gateway.history("P1", "2526")
        assert gateway._transport.calls[0][0].endswith("/history")


class TestFailures:
    def test_an_unreachable_service_is_reported_explicitly(self):
        with pytest.raises(ResultsServiceUnavailable, match="unreachable"):
            _gateway(ConnectionError("connection refused")).live(["P1"])

    def test_a_timeout_is_reported_explicitly(self):
        with pytest.raises(ResultsServiceUnavailable):
            _gateway(TimeoutError("timed out")).live(["P1"])

    def test_a_server_error_is_reported_with_its_status(self):
        with pytest.raises(ResultsServiceUnavailable, match="502"):
            _gateway(_Response({}, status_code=502)).live(["P1"])

    def test_a_rejected_key_is_reported_as_such(self):
        """401 from the service is our configuration's fault, not the user's,
        and must not look like a transient outage."""
        with pytest.raises(ResultsServiceUnavailable, match="401"):
            _gateway(_Response({}, status_code=401)).live(["P1"])

    def test_an_unreadable_body_is_reported(self):
        with pytest.raises(ResultsServiceUnavailable):
            _gateway(_Response(ValueError("not json"))).live(["P1"])

    def test_a_body_that_does_not_match_the_contract_is_reported(self):
        """Drift between the two services must surface here, not as a
        half-populated screen."""
        with pytest.raises(ResultsServiceUnavailable):
            _gateway(_Response({"unexpected": True})).live(["P1"])

    def test_a_failure_never_degrades_into_an_empty_list(self):
        with pytest.raises(ResultsServiceUnavailable):
            _gateway(_Response({}, status_code=500)).history("P1", "2526")

    def test_a_422_from_the_service_is_passed_through_as_a_bad_request(self):
        """The caller asked for a league the service does not offer — that is
        a different problem from the service being down."""
        from src.backend.services.results_gateway import ResultsRequestRejected

        with pytest.raises(ResultsRequestRejected):
            _gateway(_Response({"detail": "unknown league 'ZZ'"}, 422)).live(["ZZ"])
