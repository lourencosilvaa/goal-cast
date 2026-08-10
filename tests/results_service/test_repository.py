"""Persisting the last live snapshot to disk.

Its whole reason to exist is the free-tier cold start: an idle service is put
to sleep, so the process that wakes up has an empty cache. If the provider is
also unreachable at that moment — the likeliest time for it, since nobody has
been polling — the board would be blank with nothing to say. A snapshot on
disk turns that into last-known scores flagged ``stale``.

It is a cache, not a record: a corrupt or unreadable file must cost a fetch,
never a failed request.
"""

import json
from datetime import datetime

from src.results_service.repository import JsonResultsRepository
from src.scrapers.results.models import LiveSnapshot, MatchResult, MatchStatus

_KEY = ("E0", "P1")
_WHEN = datetime(2026, 8, 9, 17, 30)


def _snapshot(home_goals=2) -> LiveSnapshot:
    return LiveSnapshot(
        fetched_at=_WHEN,
        matches=(
            MatchResult(
                league="P1",
                kickoff=datetime(2026, 8, 9, 17, 0),
                home_team="Porto",
                away_team="Alverca",
                status=MatchStatus.LIVE,
                home_goals=home_goals,
                away_goals=0,
                minute="67'",
                source="football-data.org",
                source_id="567265",
            ),
        ),
        source="football-data.org",
    )


def _repository(tmp_path) -> JsonResultsRepository:
    return JsonResultsRepository(str(tmp_path / "results"))


class TestRoundTrip:
    def test_a_saved_snapshot_can_be_loaded_back(self, tmp_path):
        repo = _repository(tmp_path)
        repo.save(_KEY, _snapshot())
        assert repo.load(_KEY) is not None

    def test_every_field_of_a_match_survives(self, tmp_path):
        repo = _repository(tmp_path)
        repo.save(_KEY, _snapshot())
        restored = repo.load(_KEY).matches[0]
        assert restored == _snapshot().matches[0]

    def test_the_fetch_time_survives(self, tmp_path):
        repo = _repository(tmp_path)
        repo.save(_KEY, _snapshot())
        assert repo.load(_KEY).fetched_at == _WHEN

    def test_the_source_survives(self, tmp_path):
        repo = _repository(tmp_path)
        repo.save(_KEY, _snapshot())
        assert repo.load(_KEY).source == "football-data.org"

    def test_saving_again_replaces_the_previous_snapshot(self, tmp_path):
        repo = _repository(tmp_path)
        repo.save(_KEY, _snapshot(home_goals=1))
        repo.save(_KEY, _snapshot(home_goals=3))
        assert repo.load(_KEY).matches[0].home_goals == 3

    def test_different_league_sets_do_not_overwrite_each_other(self, tmp_path):
        repo = _repository(tmp_path)
        repo.save(("P1",), _snapshot(home_goals=1))
        repo.save(("E0",), _snapshot(home_goals=2))
        assert repo.load(("P1",)).matches[0].home_goals == 1

    def test_an_empty_snapshot_round_trips(self, tmp_path):
        repo = _repository(tmp_path)
        repo.save(_KEY, LiveSnapshot(_WHEN, (), ""))
        assert repo.load(_KEY).matches == ()


class TestMissingAndCorrupt:
    def test_an_unknown_key_loads_as_nothing(self, tmp_path):
        assert _repository(tmp_path).load(("ZZ",)) is None

    def test_unparseable_json_loads_as_nothing(self, tmp_path):
        repo = _repository(tmp_path)
        repo.save(_KEY, _snapshot())
        repo.path_for(_KEY).write_text("{not json")
        assert repo.load(_KEY) is None

    def test_json_of_the_wrong_shape_loads_as_nothing(self, tmp_path):
        repo = _repository(tmp_path)
        repo.save(_KEY, _snapshot())
        repo.path_for(_KEY).write_text(json.dumps({"unexpected": True}))
        assert repo.load(_KEY) is None

    def test_a_match_missing_a_required_field_is_dropped(self, tmp_path):
        repo = _repository(tmp_path)
        repo.save(_KEY, _snapshot())
        body = json.loads(repo.path_for(_KEY).read_text())
        del body["matches"][0]["home_team"]
        repo.path_for(_KEY).write_text(json.dumps(body))
        assert repo.load(_KEY).matches == ()

    def test_a_file_holding_something_other_than_an_object_loads_as_nothing(
        self, tmp_path
    ):
        repo = _repository(tmp_path)
        repo.save(_KEY, _snapshot())
        repo.path_for(_KEY).write_text(json.dumps(["not", "a", "snapshot"]))
        assert repo.load(_KEY) is None

    def test_an_unparseable_fetch_time_loads_as_nothing(self, tmp_path):
        repo = _repository(tmp_path)
        repo.save(_KEY, _snapshot())
        body = json.loads(repo.path_for(_KEY).read_text())
        body["fetched_at"] = "whenever"
        repo.path_for(_KEY).write_text(json.dumps(body))
        assert repo.load(_KEY) is None

    def test_a_match_entry_that_is_not_an_object_is_dropped(self, tmp_path):
        repo = _repository(tmp_path)
        repo.save(_KEY, _snapshot())
        body = json.loads(repo.path_for(_KEY).read_text())
        body["matches"] = ["Porto 2-0 Alverca"]
        repo.path_for(_KEY).write_text(json.dumps(body))
        assert repo.load(_KEY).matches == ()

    def test_a_match_with_an_unrecognised_status_is_dropped(self, tmp_path):
        """The rest of the board is still worth serving."""
        repo = _repository(tmp_path)
        repo.save(_KEY, _snapshot())
        body = json.loads(repo.path_for(_KEY).read_text())
        body["matches"][0]["status"] = "teapot"
        repo.path_for(_KEY).write_text(json.dumps(body))
        assert repo.load(_KEY).matches == ()

    def test_a_match_with_an_unparseable_kickoff_is_dropped(self, tmp_path):
        repo = _repository(tmp_path)
        repo.save(_KEY, _snapshot())
        body = json.loads(repo.path_for(_KEY).read_text())
        body["matches"][0]["kickoff"] = "kick-off time"
        repo.path_for(_KEY).write_text(json.dumps(body))
        assert repo.load(_KEY).matches == ()

    def test_saving_into_an_unwritable_location_does_not_raise(self, tmp_path):
        blocked = tmp_path / "blocked"
        blocked.write_text("I am a file, not a directory")
        JsonResultsRepository(str(blocked / "results")).save(_KEY, _snapshot())


class TestFileLayout:
    def test_the_directory_is_created_on_demand(self, tmp_path):
        repo = _repository(tmp_path)
        repo.save(_KEY, _snapshot())
        assert (tmp_path / "results").is_dir()

    def test_the_filename_is_derived_from_the_league_set(self, tmp_path):
        assert _repository(tmp_path).path_for(("E0", "P1")).name == "live_E0-P1.json"

    def test_an_empty_league_set_still_has_a_stable_filename(self, tmp_path):
        assert _repository(tmp_path).path_for(()).name == "live_all.json"

    def test_a_league_set_with_a_path_separator_cannot_escape_the_directory(
        self, tmp_path
    ):
        """League codes come from a query string. A code of "../../etc" must
        not be able to name a file outside the cache."""
        path = _repository(tmp_path).path_for(("../../etc/passwd",))
        assert path.parent == (tmp_path / "results")
