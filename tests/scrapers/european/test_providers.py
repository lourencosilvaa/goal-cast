"""Tests for the European fixture providers.

Every assertion here runs against payloads captured from the real APIs with
real keys (see ``samples/``), not against invented shapes. That matters: the
free tiers behave in ways no amount of reading the docs would reveal, and two
of them are documented below because the code has to survive them.

No test touches the network. The transport is injected, so a provider is a
pure request-shape plus response-parse pair.
"""

import json
from datetime import date
from pathlib import Path

import pytest

from config.config_loader import ProviderConfig
from src.scrapers.european.football_data import FootballDataProvider
from src.scrapers.european.odds_api import OddsApiProvider
from src.scrapers.european.providers import EuropeanFixture, FixtureWindow

_SAMPLES = Path(__file__).parent / "samples"


def _sample(name: str) -> dict:
    return json.loads((_SAMPLES / name).read_text(encoding="utf-8"))


class _Response:
    """Stands in for a `requests.Response`."""

    def __init__(self, payload, status_code: int = 200, headers=None) -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = json.dumps(payload) if payload is not None else ""

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


class _Transport:
    """Records what a provider asked for and returns canned responses."""

    def __init__(self, *responses) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url, params=None, headers=None):
        self.calls.append({"url": url, "params": params or {}, "headers": headers or {}})
        if not self._responses:
            return _Response([], 200)
        return self._responses.pop(0)


class _FailingTransport:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls: list[dict] = []

    def get(self, url, params=None, headers=None):
        self.calls.append({"url": url})
        raise self._exc


_WINDOW = FixtureWindow(start=date(2026, 8, 1), end=date(2026, 8, 31))


# ─────────────────────────── The Odds API ────────────────────────────


def _odds_config() -> ProviderConfig:
    return ProviderConfig(
        enabled=True,
        base_url="https://api.the-odds-api.com/v4",
        api_key_env="THE_ODDS_API_KEY",
        competitions={
            "CL": "soccer_uefa_champs_league",
            "CLQ": "soccer_uefa_champs_league_qualification",
        },
    )


class TestOddsApiCallShape:
    """The call must match the official samples-python repo exactly."""

    def test_uses_the_events_endpoint(self):
        transport = _Transport(_Response(_sample("odds_api_events.json")))
        OddsApiProvider(_odds_config(), transport, api_key="k").fetch(_WINDOW, ["CLQ"])
        url = transport.calls[0]["url"]
        assert url.endswith(
            "/sports/soccer_uefa_champs_league_qualification/events"
        ), "the /events endpoint is the free one; /odds costs credits"

    def test_the_api_key_goes_in_the_query_string_as_api_key(self):
        """samples-python uses `api_key`, not `apiKey`. Both work; follow the sample."""
        transport = _Transport(_Response(_sample("odds_api_events.json")))
        OddsApiProvider(_odds_config(), transport, api_key="secret").fetch(
            _WINDOW, ["CLQ"]
        )
        assert transport.calls[0]["params"]["api_key"] == "secret"

    def test_the_key_is_never_sent_as_a_header(self):
        transport = _Transport(_Response(_sample("odds_api_events.json")))
        OddsApiProvider(_odds_config(), transport, api_key="secret").fetch(
            _WINDOW, ["CLQ"]
        )
        assert "secret" not in str(transport.calls[0]["headers"])

    def test_an_unmapped_competition_makes_no_call(self):
        transport = _Transport()
        OddsApiProvider(_odds_config(), transport, api_key="k").fetch(_WINDOW, ["XX"])
        assert transport.calls == []


class TestOddsApiParsing:
    def test_parses_the_real_payload(self):
        transport = _Transport(_Response(_sample("odds_api_events.json")))
        fixtures = OddsApiProvider(_odds_config(), transport, api_key="k").fetch(
            _WINDOW, ["CLQ"]
        )
        assert fixtures
        assert isinstance(fixtures[0], EuropeanFixture)

    def test_reads_team_names_verbatim(self):
        """Names are carried as the provider spells them; resolution happens later."""
        transport = _Transport(_Response(_sample("odds_api_events.json")))
        fixtures = OddsApiProvider(_odds_config(), transport, api_key="k").fetch(
            _WINDOW, ["CLQ"]
        )
        assert fixtures[0].home_team == "FC Kairat"
        assert fixtures[0].away_team == "PFC Levski Sofia"

    def test_reads_the_kickoff_date(self):
        transport = _Transport(_Response(_sample("odds_api_events.json")))
        fixtures = OddsApiProvider(_odds_config(), transport, api_key="k").fetch(
            _WINDOW, ["CLQ"]
        )
        assert fixtures[0].kickoff.date() == date(2026, 8, 11)

    def test_tags_the_competition_we_asked_for(self):
        transport = _Transport(_Response(_sample("odds_api_events.json")))
        fixtures = OddsApiProvider(_odds_config(), transport, api_key="k").fetch(
            _WINDOW, ["CLQ"]
        )
        assert fixtures[0].competition == "CLQ"

    def test_keeps_the_provider_event_id(self):
        """Needed to request odds for that event without a second search."""
        transport = _Transport(_Response(_sample("odds_api_events.json")))
        fixtures = OddsApiProvider(_odds_config(), transport, api_key="k").fetch(
            _WINDOW, ["CLQ"]
        )
        assert fixtures[0].source_id == "9495fba16433b26525c818b794813b74"

    def test_fixtures_outside_the_window_are_dropped(self):
        transport = _Transport(_Response(_sample("odds_api_events.json")))
        window = FixtureWindow(start=date(2026, 9, 1), end=date(2026, 9, 30))
        fixtures = OddsApiProvider(_odds_config(), transport, api_key="k").fetch(
            window, ["CLQ"]
        )
        assert fixtures == []


class TestOddsApiQuota:
    def test_reports_remaining_quota_from_the_headers(self):
        transport = _Transport(
            _Response(
                _sample("odds_api_events.json"),
                headers={"x-requests-remaining": "499", "x-requests-used": "1"},
            )
        )
        provider = OddsApiProvider(_odds_config(), transport, api_key="k")
        provider.fetch(_WINDOW, ["CLQ"])
        assert provider.quota.remaining == 499
        assert provider.quota.used == 1

    def test_absent_quota_headers_are_tolerated(self):
        transport = _Transport(_Response(_sample("odds_api_events.json")))
        provider = OddsApiProvider(_odds_config(), transport, api_key="k")
        provider.fetch(_WINDOW, ["CLQ"])
        assert provider.quota.remaining is None


class TestOddsApiFailures:
    def test_a_non_200_yields_no_fixtures(self):
        transport = _Transport(_Response({"message": "bad key"}, status_code=401))
        assert (
            OddsApiProvider(_odds_config(), transport, api_key="k").fetch(
                _WINDOW, ["CLQ"]
            )
            == []
        )

    def test_a_transport_error_yields_no_fixtures(self):
        transport = _FailingTransport(RuntimeError("connection reset"))
        assert (
            OddsApiProvider(_odds_config(), transport, api_key="k").fetch(
                _WINDOW, ["CLQ"]
            )
            == []
        )

    def test_a_competition_out_of_season_returns_empty_not_an_error(self):
        """CL/EL/UECL all return [] in August — that is normal, not a failure."""
        transport = _Transport(_Response([]))
        assert (
            OddsApiProvider(_odds_config(), transport, api_key="k").fetch(
                _WINDOW, ["CL"]
            )
            == []
        )


# ────────────────────────── football-data.org ─────────────────────────


def _fd_config() -> ProviderConfig:
    return ProviderConfig(
        enabled=True,
        base_url="https://api.football-data.org/v4",
        api_key_env="FOOTBALL_DATA_API_KEY",
        competitions={"CL": "CL", "EL": "EL", "UECL": "ECL"},
    )


class TestFootballDataCallShape:
    def test_authenticates_with_the_x_auth_token_header(self):
        transport = _Transport(_Response(_sample("football_data_matches.json")))
        FootballDataProvider(_fd_config(), transport, api_key="secret").fetch(
            _WINDOW, ["CL"]
        )
        assert transport.calls[0]["headers"]["X-Auth-Token"] == "secret"

    def test_requests_the_competition_matches_endpoint(self):
        transport = _Transport(_Response(_sample("football_data_matches.json")))
        FootballDataProvider(_fd_config(), transport, api_key="k").fetch(
            _WINDOW, ["CL"]
        )
        assert transport.calls[0]["url"].endswith("/competitions/CL/matches")

    def test_passes_the_window_as_date_filters(self):
        transport = _Transport(_Response(_sample("football_data_matches.json")))
        FootballDataProvider(_fd_config(), transport, api_key="k").fetch(
            _WINDOW, ["CL"]
        )
        params = transport.calls[0]["params"]
        assert params["dateFrom"] == "2026-08-01"
        assert params["dateTo"] == "2026-08-31"


class TestFootballDataParsing:
    def test_parses_the_real_payload(self):
        transport = _Transport(_Response(_sample("football_data_matches.json")))
        window = FixtureWindow(start=date(2026, 5, 1), end=date(2026, 5, 31))
        fixtures = FootballDataProvider(_fd_config(), transport, api_key="k").fetch(
            window, ["CL"]
        )
        assert fixtures

    def test_prefers_short_name_because_it_matches_canonical_spellings(self):
        """'Arsenal FC' is the name; 'Arsenal' is the shortName and our key."""
        transport = _Transport(_Response(_sample("football_data_matches.json")))
        window = FixtureWindow(start=date(2026, 5, 1), end=date(2026, 5, 31))
        fixtures = FootballDataProvider(_fd_config(), transport, api_key="k").fetch(
            window, ["CL"]
        )
        assert fixtures[0].home_team == "Arsenal"


class TestFootballDataCoverageLimits:
    """The free tier covers CL only, and says so with two different codes."""

    def test_a_restricted_competition_is_not_an_error(self):
        """EL returns 403 on the free plan — expected, not a failure to report."""
        transport = _Transport(_Response({"message": "restricted"}, status_code=403))
        provider = FootballDataProvider(_fd_config(), transport, api_key="k")
        assert provider.fetch(_WINDOW, ["EL"]) == []
        assert "EL" in provider.uncovered

    def test_an_unknown_competition_is_not_an_error(self):
        """UECL returns 404 — the code does not exist for this account."""
        transport = _Transport(_Response({"message": "not found"}, status_code=404))
        provider = FootballDataProvider(_fd_config(), transport, api_key="k")
        assert provider.fetch(_WINDOW, ["UECL"]) == []
        assert "UECL" in provider.uncovered

    def test_an_empty_season_is_not_recorded_as_uncovered(self):
        """CL returns 200 with zero matches until 2026-27 loads. Dormant, not dead."""
        transport = _Transport(_Response({"resultSet": {"count": 0}, "matches": []}))
        provider = FootballDataProvider(_fd_config(), transport, api_key="k")
        assert provider.fetch(_WINDOW, ["CL"]) == []
        assert provider.uncovered == []


class TestFootballDataQuota:
    def test_reports_the_per_minute_allowance(self):
        transport = _Transport(
            _Response(
                _sample("football_data_matches.json"),
                headers={"x-requests-available-minute": "9"},
            )
        )
        provider = FootballDataProvider(_fd_config(), transport, api_key="k")
        provider.fetch(_WINDOW, ["CL"])
        assert provider.quota.remaining == 9


class TestDisabledProvider:
    @pytest.mark.parametrize(
        "provider_class,config_factory",
        [(OddsApiProvider, _odds_config), (FootballDataProvider, _fd_config)],
    )
    def test_a_disabled_provider_makes_no_call(self, provider_class, config_factory):
        config = config_factory()
        config.enabled = False
        transport = _Transport()
        assert provider_class(config, transport, api_key="k").fetch(_WINDOW, ["CL"]) == []
        assert transport.calls == []
