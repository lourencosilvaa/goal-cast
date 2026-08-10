"""Edge cases found while implementing, not while designing.

Everything here answers "what does a real provider do that a clean payload
does not?" — a match with no competition block, a season file in cp1252, a
disallowed goal, a transport that dies mid-chain. Each one exists because the
code has a branch for it, and a branch nothing exercises is a guess.
"""

from datetime import date, datetime

from config.config_loader import LocalCorpusConfig, ResultsProviderConfig
from src.scrapers.results.flashscore_live import FlashscoreLiveProvider
from src.scrapers.results.football_data import (
    FootballDataHistoryProvider,
    FootballDataLiveProvider,
)
from src.scrapers.results.local_corpus import LocalCorpusHistoryProvider
from src.scrapers.results.models import HistoryQuery, MatchStatus


class _Response:
    def __init__(self, body, status_code: int = 200, text: str = ""):
        self._body = body
        self.status_code = status_code
        self.text = text
        self.headers: dict[str, str] = {}

    def json(self):
        return self._body


class _Transport:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls: list[str] = []

    def get(self, url, params=None, headers=None):
        self.calls.append(url)
        response = self._responses.pop(0) if self._responses else _Response({})
        if isinstance(response, Exception):
            raise response
        return response


def _api_config(**overrides) -> ResultsProviderConfig:
    defaults = dict(
        enabled=True,
        base_url="https://api.test/v4",
        api_key_env="FOOTBALL_DATA_API_KEY",
        competitions={"P1": "PPL"},
    )
    defaults.update(overrides)
    return ResultsProviderConfig(**defaults)


def _live(*responses) -> FootballDataLiveProvider:
    return FootballDataLiveProvider(
        _api_config(), _Transport(*responses), "key", today=lambda: date(2026, 8, 9)
    )


def _wrap(*matches) -> _Response:
    return _Response({"matches": list(matches)})


def _payload(**overrides) -> dict:
    match = {
        "id": 1,
        "utcDate": "2026-08-09T17:00:00Z",
        "status": "LIVE",
        "homeTeam": {"name": "FC Porto", "shortName": "Porto"},
        "awayTeam": {"name": "FC Alverca", "shortName": "Alverca"},
        "score": {"fullTime": {"home": 2, "away": 0}},
        "competition": {"code": "PPL"},
    }
    match.update(overrides)
    return match


class TestMalformedApiPayloads:
    def test_a_match_that_is_not_an_object_is_dropped(self):
        assert _live(_wrap("not a match")).fetch_live(["P1"]) == []

    def test_a_match_with_no_competition_block_is_dropped(self):
        payload = _payload()
        del payload["competition"]
        assert _live(_wrap(payload)).fetch_live(["P1"]) == []

    def test_a_competition_that_is_not_an_object_is_dropped(self):
        assert _live(_wrap(_payload(competition="PPL"))).fetch_live(["P1"]) == []

    def test_an_unrecognised_status_is_dropped_rather_than_guessed(self):
        """A status we have no mapping for could be anything; putting the
        match on the board under an invented state is worse than omitting it."""
        assert _live(_wrap(_payload(status="TEAPOT"))).fetch_live(["P1"]) == []

    def test_a_match_with_no_teams_is_dropped(self):
        assert _live(_wrap(_payload(homeTeam={}))).fetch_live(["P1"]) == []

    def test_a_team_that_is_not_an_object_is_dropped(self):
        assert _live(_wrap(_payload(awayTeam="Alverca"))).fetch_live(["P1"]) == []

    def test_an_unparseable_kickoff_is_dropped(self):
        assert _live(_wrap(_payload(utcDate="yesterday"))).fetch_live(["P1"]) == []

    def test_a_missing_kickoff_is_dropped(self):
        assert _live(_wrap(_payload(utcDate=None))).fetch_live(["P1"]) == []

    def test_a_body_that_is_not_an_object_yields_nothing(self):
        assert _live(_Response(["surprise"])).fetch_live(["P1"]) == []

    def test_a_matches_field_that_is_not_a_list_yields_nothing(self):
        assert _live(_Response({"matches": {}})).fetch_live(["P1"]) == []

    def test_the_full_name_is_used_when_there_is_no_short_name(self):
        payload = _payload(homeTeam={"name": "FC Porto"})
        match = _live(_wrap(payload)).fetch_live(["P1"])[0]
        assert match.home_team == "FC Porto"


class TestScoreEdges:
    def test_a_score_block_of_the_wrong_shape_yields_no_goals(self):
        match = _live(_wrap(_payload(score={"fullTime": []}))).fetch_live(["P1"])[0]
        assert (match.home_goals, match.away_goals) == (None, None)

    def test_a_missing_score_block_yields_no_goals(self):
        payload = _payload()
        del payload["score"]
        match = _live(_wrap(payload)).fetch_live(["P1"])[0]
        assert match.home_goals is None

    def test_a_non_numeric_goal_count_is_not_coerced(self):
        payload = _payload(score={"fullTime": {"home": "2", "away": 0}})
        match = _live(_wrap(payload)).fetch_live(["P1"])[0]
        assert match.home_goals is None

    def test_a_scheduled_match_never_reports_a_score(self):
        """The API sends the nulls, but a stale non-null must not slip past."""
        payload = _payload(status="TIMED", score={"fullTime": {"home": 0, "away": 0}})
        match = _live(_wrap(payload)).fetch_live(["P1"])[0]
        assert (match.home_goals, match.away_goals) == (None, None)

    def test_a_paused_match_keeps_its_half_time_score(self):
        payload = _payload(status="PAUSED")
        match = _live(_wrap(payload)).fetch_live(["P1"])[0]
        assert (match.status, match.home_goals) == (MatchStatus.PAUSED, 2)

    def test_an_awarded_match_counts_as_finished(self):
        """A result awarded off the pitch is still a result."""
        payload = _payload(status="AWARDED")
        match = _live(_wrap(payload)).fetch_live(["P1"])[0]
        assert match.status is MatchStatus.FINISHED

    def test_a_suspended_match_is_a_pause_not_a_cancellation(self):
        payload = _payload(status="SUSPENDED")
        match = _live(_wrap(payload)).fetch_live(["P1"])[0]
        assert match.status is MatchStatus.PAUSED

    def test_a_minute_is_used_when_the_provider_does_send_one(self):
        """Absent on the free tier; paid tiers do send it."""
        match = _live(_wrap(_payload(minute="67"))).fetch_live(["P1"])[0]
        assert match.minute == "67"


class TestHistoryEdges:
    def _history(self, *responses) -> FootballDataHistoryProvider:
        return FootballDataHistoryProvider(_api_config(), _Transport(*responses), "key")

    def test_a_transport_error_yields_nothing(self):
        provider = self._history(RuntimeError("connection reset"))
        assert provider.fetch_history(HistoryQuery("P1", "2526")) == []

    def test_a_disabled_provider_makes_no_request(self):
        provider = FootballDataHistoryProvider(
            _api_config(enabled=False), _Transport(), "key"
        )
        assert provider.fetch_history(HistoryQuery("P1", "2526")) == []

    def test_a_season_of_the_wrong_length_is_refused(self):
        provider = self._history()
        assert provider.fetch_history(HistoryQuery("P1", "25")) == []

    def test_a_five_digit_season_is_refused(self):
        provider = self._history()
        assert provider.fetch_history(HistoryQuery("P1", "20256")) == []

    def test_a_history_entry_that_is_not_an_object_is_dropped(self):
        provider = self._history(_wrap("not a match", _payload(status="FINISHED")))
        assert len(provider.fetch_history(HistoryQuery("P1", "2526"))) == 1

    def test_an_already_uncovered_league_is_not_requested_again(self):
        provider = self._history()
        provider.uncovered.append("P1")
        assert provider.fetch_history(HistoryQuery("P1", "2526")) == []


class TestCorpusEncodings:
    """Some season files carry a stray cp1252 byte inside an odds column. A
    strict UTF-8 read aborts on it and loses the whole season, which is a real
    thing that happened to 1819/SC0 and 1819/I2."""

    def _provider(self, tmp_path) -> LocalCorpusHistoryProvider:
        return LocalCorpusHistoryProvider(
            LocalCorpusConfig(
                enabled=True,
                base_url="https://example.test",
                search_dirs=[str(tmp_path / "corpus")],
                leagues=["P1"],
            ),
            _Transport(),
            cache_dir=str(tmp_path / "cache"),
        )

    def test_a_file_with_a_cp1252_byte_is_still_read(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir(parents=True)
        body = (
            "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG\n"
            "P1,08/08/2025,20:15,Casa Pia,Sp Lisbon’s,0,2\n"
        )
        (corpus / "2526_P1.csv").write_bytes(body.encode("cp1252"))
        matches = self._provider(tmp_path).fetch_history(HistoryQuery("P1", "2526"))
        assert len(matches) == 1

    def test_a_download_that_returns_only_whitespace_is_not_cached(self, tmp_path):
        provider = LocalCorpusHistoryProvider(
            LocalCorpusConfig(
                enabled=True,
                base_url="https://example.test",
                search_dirs=[str(tmp_path / "corpus")],
                leagues=["P1"],
            ),
            _Transport(_Response(None, 200, text="   \n")),
            cache_dir=str(tmp_path / "cache"),
        )
        assert provider.fetch_history(HistoryQuery("P1", "2526")) == []
        assert not (tmp_path / "cache" / "2526_P1.csv").exists()

    def test_an_unwritable_cache_directory_still_answers(self, tmp_path):
        """A read-only disk costs a repeated download, not a failed request."""
        blocked = tmp_path / "blocked"
        blocked.write_text("I am a file, not a directory")
        provider = LocalCorpusHistoryProvider(
            LocalCorpusConfig(
                enabled=True,
                base_url="https://example.test",
                search_dirs=[],
                leagues=["P1"],
            ),
            _Transport(
                _Response(
                    None,
                    200,
                    text=(
                        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG\n"
                        "P1,08/08/2025,20:15,Casa Pia,Sp Lisbon,0,2\n"
                    ),
                )
            ),
            cache_dir=str(blocked / "cache"),
        )
        assert len(provider.fetch_history(HistoryQuery("P1", "2526"))) == 1

    def test_a_headerless_file_yields_nothing(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir(parents=True)
        (corpus / "2526_P1.csv").write_text("\n\n")
        assert self._provider(tmp_path).fetch_history(HistoryQuery("P1", "2526")) == []


class TestFlashscoreEdges:
    class _Client:
        def __init__(self, rows):
            self._rows = rows

        def scrape_live(self, league_code):
            return list(self._rows)

    class _Row:
        def __init__(self, **fields):
            for name, value in fields.items():
                setattr(self, name, value)

    def _provider(self, rows) -> FlashscoreLiveProvider:
        return FlashscoreLiveProvider(
            ResultsProviderConfig(enabled=True),
            self._Client(rows),
            today=lambda: date(2026, 8, 9),
        )

    def test_an_unparseable_kickoff_falls_back_to_today(self):
        row = self._Row(
            home_team="Porto",
            away_team="Alverca",
            home_goals=1,
            away_goals=0,
            minute="12'",
            kickoff="soon",
        )
        match = self._provider([row]).fetch_live(["P1"])[0]
        assert match.kickoff == datetime(2026, 8, 9, 0, 0)

    def test_a_finished_row_does_not_repeat_its_stage_as_a_minute(self):
        row = self._Row(
            home_team="Porto",
            away_team="Alverca",
            home_goals=1,
            away_goals=0,
            minute="Finished",
        )
        assert self._provider([row]).fetch_live(["P1"])[0].minute == ""

    def test_a_postponed_row_is_postponed(self):
        row = self._Row(
            home_team="Porto",
            away_team="Alverca",
            home_goals=None,
            away_goals=None,
            minute="Postponed",
        )
        assert self._provider([row]).fetch_live(["P1"])[0].status is (
            MatchStatus.POSTPONED
        )

    def test_the_transport_this_provider_must_not_use_refuses_to_be_used(self):
        """It reaches the network through the browser client; a silent None
        would surface as an AttributeError somewhere less obvious."""
        import pytest

        from src.scrapers.results.flashscore_live import _NO_TRANSPORT

        with pytest.raises(NotImplementedError):
            _NO_TRANSPORT.get("https://example.test")
