"""Boundary and malformed-input coverage for the fixture providers.

A provider sits between an API nobody here controls and a prediction shown to
a user. The failure that matters is not a crash — it is a provider that
returns something plausible from something broken, because the chain then
accepts it as a complete answer and stops asking anyone else.
"""

import json
from datetime import date, datetime

import pytest

from config.config_loader import ProviderConfig
from src.scrapers.european.football_data import FootballDataProvider
from src.scrapers.european.odds_api import OddsApiProvider
from src.scrapers.european.providers import (
    EuropeanFixture,
    FixtureWindow,
    QuotaReport,
)


class _Response:
    def __init__(self, payload, status_code=200, headers=None, raises=False) -> None:
        self._payload = payload
        self._raises = raises
        self.status_code = status_code
        self.headers = headers or {}
        self.text = "" if raises else json.dumps(payload, default=str)

    def json(self):
        if self._raises:
            raise ValueError("not JSON")
        return self._payload


class _Transport:
    def __init__(self, *responses) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url, params=None, headers=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        return self._responses.pop(0) if self._responses else _Response([])


_WINDOW = FixtureWindow(start=date(2026, 8, 1), end=date(2026, 8, 31))


def _odds(**kw) -> OddsApiProvider:
    config = ProviderConfig(
        base_url="https://x.test/v4",
        api_key_env="K",
        competitions={"CL": "soccer_cl"},
    )
    return OddsApiProvider(config, kw["transport"], api_key=kw.get("key", "k"))


def _fd(**kw) -> FootballDataProvider:
    config = ProviderConfig(
        base_url="https://x.test/v4", api_key_env="K", competitions={"CL": "CL"}
    )
    return FootballDataProvider(config, kw["transport"], api_key=kw.get("key", "k"))


class TestMalformedPayloads:
    def test_unparseable_json_yields_nothing(self):
        assert _odds(transport=_Transport(_Response(None, raises=True))).fetch(
            _WINDOW, ["CL"]
        ) == []

    def test_a_json_object_where_a_list_was_expected_yields_nothing(self):
        assert _odds(transport=_Transport(_Response({"message": "x"}))).fetch(
            _WINDOW, ["CL"]
        ) == []

    def test_non_dict_entries_are_skipped(self):
        transport = _Transport(_Response(["not-a-dict", 42, None]))
        assert _odds(transport=transport).fetch(_WINDOW, ["CL"]) == []

    def test_an_event_missing_team_names_is_skipped(self):
        transport = _Transport(
            _Response([{"commence_time": "2026-08-11T15:00:00Z", "home_team": "A"}])
        )
        assert _odds(transport=transport).fetch(_WINDOW, ["CL"]) == []

    def test_an_event_with_a_blank_team_name_is_skipped(self):
        transport = _Transport(
            _Response(
                [
                    {
                        "commence_time": "2026-08-11T15:00:00Z",
                        "home_team": "   ",
                        "away_team": "B",
                    }
                ]
            )
        )
        assert _odds(transport=transport).fetch(_WINDOW, ["CL"]) == []

    def test_a_valid_event_survives_alongside_broken_ones(self):
        """One bad row must not discard the good ones."""
        transport = _Transport(
            _Response(
                [
                    {"garbage": True},
                    {
                        "commence_time": "2026-08-11T15:00:00Z",
                        "home_team": "A",
                        "away_team": "B",
                    },
                ]
            )
        )
        fixtures = _odds(transport=transport).fetch(_WINDOW, ["CL"])
        assert [f.home_team for f in fixtures] == ["A"]

    def test_football_data_without_a_matches_key_yields_nothing(self):
        assert _fd(transport=_Transport(_Response({"resultSet": {}}))).fetch(
            _WINDOW, ["CL"]
        ) == []


class TestDateHandling:
    @pytest.mark.parametrize("raw", ["", None, "not-a-date", 12345, "2026-13-45"])
    def test_an_unusable_kickoff_is_skipped(self, raw):
        transport = _Transport(
            _Response([{"commence_time": raw, "home_team": "A", "away_team": "B"}])
        )
        assert _odds(transport=transport).fetch(_WINDOW, ["CL"]) == []

    def test_a_utc_offset_is_normalised_away(self):
        """Mixing aware and naive datetimes downstream compares silently wrong."""
        transport = _Transport(
            _Response(
                [
                    {
                        "commence_time": "2026-08-11T15:00:00Z",
                        "home_team": "A",
                        "away_team": "B",
                    }
                ]
            )
        )
        fixtures = _odds(transport=transport).fetch(_WINDOW, ["CL"])
        assert fixtures[0].kickoff.tzinfo is None

    def test_the_window_is_inclusive_at_both_ends(self):
        window = FixtureWindow(start=date(2026, 8, 11), end=date(2026, 8, 11))
        assert window.contains(datetime(2026, 8, 11, 23, 59))
        assert not window.contains(datetime(2026, 8, 12, 0, 0))


class TestQuotaParsing:
    @pytest.mark.parametrize("value", ["not-a-number", None, ""])
    def test_an_unparseable_quota_header_becomes_none(self, value):
        report = QuotaReport.from_headers({"r": value}, "r")
        assert report.remaining is None

    def test_headers_without_a_get_method_are_tolerated(self):
        assert QuotaReport.from_headers(object(), "r").remaining is None

    def test_a_valid_header_is_read(self):
        assert QuotaReport.from_headers({"r": "17"}, "r").remaining == 17


class TestNotCoveredIsRememberedNotRetried:
    def test_a_restricted_competition_is_requested_only_once(self):
        """Retrying a 403 spends the 10/minute allowance to learn nothing."""
        transport = _Transport(
            _Response({}, status_code=403), _Response({}, status_code=403)
        )
        provider = _fd(transport=transport)
        provider.fetch(_WINDOW, ["CL"])
        provider.fetch(_WINDOW, ["CL"])
        assert len(transport.calls) == 1


class TestAuthGuard:
    def test_a_provider_without_a_key_makes_no_call(self):
        transport = _Transport()
        assert _odds(transport=transport, key="").fetch(_WINDOW, ["CL"]) == []
        assert transport.calls == []


class TestFixtureIdentity:
    def test_fixtures_compare_by_value(self):
        """Needed for de-duplication and for readable test failures."""
        kickoff = datetime(2026, 8, 11, 15, 0)
        one = EuropeanFixture("CL", kickoff, "A", "B", "s")
        two = EuropeanFixture("CL", kickoff, "A", "B", "s")
        assert one == two


class TestFootballDataErrorPaths:
    """The same failure modes as the Odds API, exercised on the other parser.

    Kept explicit rather than parametrised across both providers: the two
    parse different payload shapes, and a shared test would only prove the
    shape they have in common.
    """

    def test_a_transport_error_yields_nothing(self):
        class _Boom:
            calls: list = []

            def get(self, url, params=None, headers=None):
                raise RuntimeError("connection reset")

        assert _fd(transport=_Boom()).fetch(_WINDOW, ["CL"]) == []

    def test_an_unexpected_status_is_not_recorded_as_uncovered(self):
        """A 500 is the server failing, not the plan excluding a competition."""
        provider = _fd(transport=_Transport(_Response({}, status_code=500)))
        assert provider.fetch(_WINDOW, ["CL"]) == []
        assert provider.uncovered == []

    def test_unparseable_json_yields_nothing(self):
        transport = _Transport(_Response(None, raises=True))
        assert _fd(transport=transport).fetch(_WINDOW, ["CL"]) == []

    def test_a_non_list_matches_value_yields_nothing(self):
        transport = _Transport(_Response({"matches": "not-a-list"}))
        assert _fd(transport=transport).fetch(_WINDOW, ["CL"]) == []

    def test_a_non_dict_match_is_skipped(self):
        transport = _Transport(_Response({"matches": ["nope", 7, None]}))
        assert _fd(transport=transport).fetch(_WINDOW, ["CL"]) == []

    def test_a_match_without_teams_is_skipped(self):
        transport = _Transport(
            _Response({"matches": [{"utcDate": "2026-08-11T19:00:00Z"}]})
        )
        assert _fd(transport=transport).fetch(_WINDOW, ["CL"]) == []

    def test_a_non_dict_team_yields_no_name(self):
        transport = _Transport(
            _Response(
                {
                    "matches": [
                        {
                            "utcDate": "2026-08-11T19:00:00Z",
                            "homeTeam": "Arsenal",
                            "awayTeam": {"shortName": "Benfica"},
                        }
                    ]
                }
            )
        )
        assert _fd(transport=transport).fetch(_WINDOW, ["CL"]) == []

    def test_name_is_used_when_short_name_is_absent(self):
        transport = _Transport(
            _Response(
                {
                    "matches": [
                        {
                            "utcDate": "2026-08-11T19:00:00Z",
                            "homeTeam": {"name": "Arsenal FC"},
                            "awayTeam": {"name": "SL Benfica"},
                        }
                    ]
                }
            )
        )
        fixtures = _fd(transport=transport).fetch(_WINDOW, ["CL"])
        assert fixtures[0].home_team == "Arsenal FC"

    @pytest.mark.parametrize("raw", ["", None, "not-a-date", 99])
    def test_an_unusable_utc_date_is_skipped(self, raw):
        transport = _Transport(
            _Response(
                {
                    "matches": [
                        {
                            "utcDate": raw,
                            "homeTeam": {"shortName": "Arsenal"},
                            "awayTeam": {"shortName": "Benfica"},
                        }
                    ]
                }
            )
        )
        assert _fd(transport=transport).fetch(_WINDOW, ["CL"]) == []

    def test_a_match_outside_the_window_is_dropped(self):
        transport = _Transport(
            _Response(
                {
                    "matches": [
                        {
                            "utcDate": "2027-01-01T19:00:00Z",
                            "homeTeam": {"shortName": "Arsenal"},
                            "awayTeam": {"shortName": "Benfica"},
                        }
                    ]
                }
            )
        )
        assert _fd(transport=transport).fetch(_WINDOW, ["CL"]) == []
