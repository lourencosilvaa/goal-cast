"""Tests for the HTTP transport.

This is the only place the project makes an outbound fixture request, and it is
our own ``requests`` code rather than any vendor SDK — deliberately, so what
goes on the wire is visible here and cannot change under a package upgrade.

The retry policy is the part worth pinning: retrying a *status* would burn
quota measured in hundreds per month to receive the same answer, while
retrying a connection that never reached a server is free and occasionally
necessary — api.football-data.org returned a spurious SSL EOF during
development and succeeded immediately on retry.
"""

from unittest.mock import patch

import pytest
import requests

from src.scrapers.european.http import HttpTransport


class _Recorder:
    def __init__(self, *outcomes) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict] = []

    def __call__(self, url, params=None, headers=None, timeout=None):
        self.calls.append(
            {"url": url, "params": params, "headers": headers, "timeout": timeout}
        )
        outcome = self._outcomes.pop(0) if self._outcomes else "ok"
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class TestRequestShape:
    def test_passes_url_params_and_headers_through(self):
        recorder = _Recorder()
        with patch("src.scrapers.european.http.requests.get", recorder):
            HttpTransport().get(
                "https://x.test/a", params={"k": "v"}, headers={"H": "1"}
            )
        assert recorder.calls[0]["url"] == "https://x.test/a"
        assert recorder.calls[0]["params"] == {"k": "v"}
        assert recorder.calls[0]["headers"] == {"H": "1"}

    def test_applies_the_configured_timeout(self):
        recorder = _Recorder()
        with patch("src.scrapers.european.http.requests.get", recorder):
            HttpTransport(timeout=42).get("https://x.test/a")
        assert recorder.calls[0]["timeout"] == 42

    def test_omitted_params_become_empty_not_none(self):
        recorder = _Recorder()
        with patch("src.scrapers.european.http.requests.get", recorder):
            HttpTransport().get("https://x.test/a")
        assert recorder.calls[0]["params"] == {}
        assert recorder.calls[0]["headers"] == {}


class TestRetries:
    def test_a_connection_error_is_retried(self):
        recorder = _Recorder(requests.exceptions.ConnectionError("ssl eof"), "ok")
        with patch("src.scrapers.european.http.requests.get", recorder):
            with patch("src.scrapers.european.http.time.sleep"):
                assert HttpTransport(retries=2).get("https://x.test/a") == "ok"
        assert len(recorder.calls) == 2

    def test_a_timeout_is_retried(self):
        recorder = _Recorder(requests.exceptions.Timeout("slow"), "ok")
        with patch("src.scrapers.european.http.requests.get", recorder):
            with patch("src.scrapers.european.http.time.sleep"):
                assert HttpTransport(retries=1).get("https://x.test/a") == "ok"

    def test_retries_are_bounded(self):
        recorder = _Recorder(
            *[requests.exceptions.ConnectionError("down")] * 5
        )
        with patch("src.scrapers.european.http.requests.get", recorder):
            with patch("src.scrapers.european.http.time.sleep"):
                with pytest.raises(requests.exceptions.ConnectionError):
                    HttpTransport(retries=2).get("https://x.test/a")
        assert len(recorder.calls) == 3  # the original plus two retries

    def test_zero_retries_makes_exactly_one_attempt(self):
        recorder = _Recorder(requests.exceptions.ConnectionError("down"))
        with patch("src.scrapers.european.http.requests.get", recorder):
            with pytest.raises(requests.exceptions.ConnectionError):
                HttpTransport(retries=0).get("https://x.test/a")
        assert len(recorder.calls) == 1

    def test_a_non_transport_error_is_not_retried(self):
        """A bad status is an answer. Repeating it only spends quota."""
        recorder = _Recorder(ValueError("nonsense"))
        with patch("src.scrapers.european.http.requests.get", recorder):
            with pytest.raises(ValueError):
                HttpTransport(retries=3).get("https://x.test/a")
        assert len(recorder.calls) == 1

    def test_backoff_grows_between_attempts(self):
        recorder = _Recorder(
            requests.exceptions.ConnectionError("a"),
            requests.exceptions.ConnectionError("b"),
            "ok",
        )
        slept: list[float] = []
        with patch("src.scrapers.european.http.requests.get", recorder):
            with patch(
                "src.scrapers.european.http.time.sleep", lambda s: slept.append(s)
            ):
                HttpTransport(retries=2, backoff_seconds=0.5).get("https://x.test/a")
        assert slept == [0.5, 1.0]
