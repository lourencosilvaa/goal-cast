"""History from the football-data.co.uk CSVs, read without pandas.

The results service installs the ``results`` dependency group — fastapi,
requests, playwright — and no scientific stack, so this provider parses with
the standard library. That is the constraint that keeps the service image
small, and it is worth a test of its own.
"""

from datetime import datetime

from config.config_loader import LocalCorpusConfig
from src.scrapers.results.local_corpus import LocalCorpusHistoryProvider
from src.scrapers.results.models import HistoryQuery, MatchStatus

_HEADER = "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR"
_ROWS = [
    "P1,08/08/2025,20:15,Casa Pia,Sp Lisbon,0,2,A",
    "P1,10/08/2025,18:00,Porto,Benfica,1,1,D",
]


def _csv(rows=None) -> str:
    return "\n".join([_HEADER, *(rows if rows is not None else _ROWS)]) + "\n"


class _Response:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code
        self.headers: dict[str, str] = {}


class _Transport:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls: list[str] = []

    def get(self, url, params=None, headers=None):
        self.calls.append(url)
        response = self._responses.pop(0) if self._responses else _Response("", 404)
        if isinstance(response, Exception):
            raise response
        return response


def _config(tmp_path, **overrides) -> LocalCorpusConfig:
    defaults = dict(
        enabled=True,
        base_url="https://example.test/mmz4281",
        search_dirs=[str(tmp_path / "corpus")],
        leagues=["P1", "E0"],
        timeout=30,
    )
    defaults.update(overrides)
    return LocalCorpusConfig(**defaults)


def _provider(tmp_path, transport=None, **overrides) -> LocalCorpusHistoryProvider:
    return LocalCorpusHistoryProvider(
        _config(tmp_path, **overrides),
        transport or _Transport(),
        cache_dir=str(tmp_path / "cache"),
    )


def _seed(tmp_path, name="2526_P1.csv", content=None) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / name).write_text(content if content is not None else _csv())


class TestReadingLocalFiles:
    def test_a_local_season_file_answers_without_any_request(self, tmp_path):
        _seed(tmp_path)
        transport = _Transport()
        matches = _provider(tmp_path, transport).fetch_history(
            HistoryQuery(league="P1", season="2526")
        )
        assert len(matches) == 2
        assert transport.calls == []

    def test_teams_and_goals_are_read_from_the_full_time_columns(self, tmp_path):
        _seed(tmp_path)
        first = _provider(tmp_path).fetch_history(
            HistoryQuery(league="P1", season="2526")
        )[0]
        assert (first.home_team, first.away_team) == ("Casa Pia", "Sp Lisbon")
        assert (first.home_goals, first.away_goals) == (0, 2)

    def test_the_date_and_time_columns_become_the_kickoff(self, tmp_path):
        _seed(tmp_path)
        first = _provider(tmp_path).fetch_history(
            HistoryQuery(league="P1", season="2526")
        )[0]
        assert first.kickoff == datetime(2025, 8, 8, 20, 15)

    def test_every_corpus_row_is_a_finished_match(self, tmp_path):
        _seed(tmp_path)
        matches = _provider(tmp_path).fetch_history(
            HistoryQuery(league="P1", season="2526")
        )
        assert all(m.status is MatchStatus.FINISHED for m in matches)

    def test_matches_come_back_ordered_by_kickoff(self, tmp_path):
        _seed(tmp_path, content=_csv(list(reversed(_ROWS))))
        matches = _provider(tmp_path).fetch_history(
            HistoryQuery(league="P1", season="2526")
        )
        assert [m.kickoff for m in matches] == sorted(m.kickoff for m in matches)

    def test_the_answer_is_attributed_to_the_local_corpus(self, tmp_path):
        _seed(tmp_path)
        matches = _provider(tmp_path).fetch_history(
            HistoryQuery(league="P1", season="2526")
        )
        assert matches[0].source == "local-corpus"


class TestDownloadingWhenAbsent:
    def test_a_missing_season_is_downloaded_from_the_configured_base_url(
        self, tmp_path
    ):
        transport = _Transport(_Response(_csv()))
        matches = _provider(tmp_path, transport).fetch_history(
            HistoryQuery(league="P1", season="2526")
        )
        assert transport.calls == ["https://example.test/mmz4281/2526/P1.csv"]
        assert len(matches) == 2

    def test_a_downloaded_season_is_cached_and_not_fetched_twice(self, tmp_path):
        transport = _Transport(_Response(_csv()))
        provider = _provider(tmp_path, transport)
        provider.fetch_history(HistoryQuery(league="P1", season="2526"))
        provider.fetch_history(HistoryQuery(league="P1", season="2526"))
        assert len(transport.calls) == 1

    def test_a_failed_download_yields_nothing_rather_than_raising(self, tmp_path):
        transport = _Transport(RuntimeError("no network"))
        assert (
            _provider(tmp_path, transport).fetch_history(
                HistoryQuery(league="P1", season="2526")
            )
            == []
        )

    def test_a_404_yields_nothing_and_writes_no_cache_file(self, tmp_path):
        transport = _Transport(_Response("Not found", 404))
        provider = _provider(tmp_path, transport)
        assert provider.fetch_history(HistoryQuery(league="P1", season="2526")) == []
        assert not (tmp_path / "cache" / "2526_P1.csv").exists()


class TestLeaguesWithoutAFeed:
    def test_a_league_outside_the_configured_list_is_never_requested(self, tmp_path):
        """The UEFA competitions have no football-data.co.uk feed; asking for
        one would 404 on every history call."""
        transport = _Transport()
        assert (
            _provider(tmp_path, transport).fetch_history(
                HistoryQuery(league="CL", season="2526")
            )
            == []
        )
        assert transport.calls == []

    def test_a_disabled_provider_reads_nothing(self, tmp_path):
        _seed(tmp_path)
        provider = _provider(tmp_path, enabled=False)
        assert provider.fetch_history(HistoryQuery(league="P1", season="2526")) == []


class TestMalformedRows:
    def test_a_row_with_no_score_is_skipped_not_guessed(self, tmp_path):
        _seed(tmp_path, content=_csv(["P1,08/08/2025,20:15,Casa Pia,Sp Lisbon,,,"]))
        assert (
            _provider(tmp_path).fetch_history(HistoryQuery(league="P1", season="2526"))
            == []
        )

    def test_a_row_with_an_unreadable_date_is_skipped(self, tmp_path):
        _seed(tmp_path, content=_csv(["P1,not-a-date,20:15,Casa Pia,Sp Lisbon,0,2,A"]))
        assert (
            _provider(tmp_path).fetch_history(HistoryQuery(league="P1", season="2526"))
            == []
        )

    def test_a_row_with_no_time_still_yields_a_match_at_midnight(self, tmp_path):
        _seed(tmp_path, content=_csv(["P1,08/08/2025,,Casa Pia,Sp Lisbon,0,2,A"]))
        match = _provider(tmp_path).fetch_history(
            HistoryQuery(league="P1", season="2526")
        )[0]
        assert match.kickoff == datetime(2025, 8, 8, 0, 0)

    def test_a_two_digit_year_is_read_as_this_century(self, tmp_path):
        """Older football-data seasons use DD/MM/YY."""
        _seed(tmp_path, content=_csv(["P1,08/08/25,20:15,Casa Pia,Sp Lisbon,0,2,A"]))
        match = _provider(tmp_path).fetch_history(
            HistoryQuery(league="P1", season="2526")
        )[0]
        assert match.kickoff.year == 2025

    def test_an_empty_file_yields_nothing(self, tmp_path):
        _seed(tmp_path, content="")
        assert (
            _provider(tmp_path).fetch_history(HistoryQuery(league="P1", season="2526"))
            == []
        )

    def test_a_file_missing_the_expected_columns_yields_nothing(self, tmp_path):
        _seed(tmp_path, content="Foo,Bar\n1,2\n")
        assert (
            _provider(tmp_path).fetch_history(HistoryQuery(league="P1", season="2526"))
            == []
        )
