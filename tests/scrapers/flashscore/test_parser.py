from src.scrapers.flashscore.parser import FlashScoreParser
from src.scrapers.base_scraper import FlashScoreFixture


SAMPLE_FIXTURE_RESPONSE = (
    "ZEE\xf7abc123\xacAA\xf7Arsenal\xacAB\xf7Chelsea\xac"
    "AD\xf71714000000\xacAE\xf7scheduled\xacAF\xf7\xacAG\xf7\xac~"
    "ZEE\xf7def456\xacAA\xf7Liverpool\xacAB\xf7Man City\xac"
    "AD\xf71714086400\xacAE\xf7scheduled\xacAF\xf7\xacAG\xf7\xac~"
)

SAMPLE_RESULT_RESPONSE = (
    "ZEE\xf7xyz789\xacAA\xf7Tottenham\xacAB\xf7West Ham\xac"
    "AD\xf71713913600\xacAE\xf7finished\xacAF\xf72\xacAG\xf71\xac~"
)


class TestFlashScoreParser:

    def setup_method(self):
        self.parser = FlashScoreParser()

    def test_parse_fixtures_returns_list(self):
        fixtures = self.parser.parse_fixtures(SAMPLE_FIXTURE_RESPONSE, league="Premier League", country="England")
        assert isinstance(fixtures, list)

    def test_parse_fixtures_extracts_correct_count(self):
        fixtures = self.parser.parse_fixtures(SAMPLE_FIXTURE_RESPONSE, league="Premier League", country="England")
        assert len(fixtures) == 2

    def test_parse_fixtures_home_team(self):
        fixtures = self.parser.parse_fixtures(SAMPLE_FIXTURE_RESPONSE, league="Premier League", country="England")
        assert fixtures[0].home_team == "Arsenal"

    def test_parse_fixtures_away_team(self):
        fixtures = self.parser.parse_fixtures(SAMPLE_FIXTURE_RESPONSE, league="Premier League", country="England")
        assert fixtures[0].away_team == "Chelsea"

    def test_parse_fixtures_match_id(self):
        fixtures = self.parser.parse_fixtures(SAMPLE_FIXTURE_RESPONSE, league="Premier League", country="England")
        assert fixtures[0].match_id == "abc123"

    def test_parse_fixtures_league_populated(self):
        fixtures = self.parser.parse_fixtures(SAMPLE_FIXTURE_RESPONSE, league="Premier League", country="England")
        assert fixtures[0].league == "Premier League"

    def test_parse_fixtures_country_populated(self):
        fixtures = self.parser.parse_fixtures(SAMPLE_FIXTURE_RESPONSE, league="Premier League", country="England")
        assert fixtures[0].country == "England"

    def test_parse_fixtures_scheduled_status(self):
        fixtures = self.parser.parse_fixtures(SAMPLE_FIXTURE_RESPONSE, league="Premier League", country="England")
        assert fixtures[0].status == "scheduled"

    def test_parse_fixtures_no_scores_for_scheduled(self):
        fixtures = self.parser.parse_fixtures(SAMPLE_FIXTURE_RESPONSE, league="Premier League", country="England")
        assert fixtures[0].home_score is None
        assert fixtures[0].away_score is None

    def test_parse_results_finished_status(self):
        fixtures = self.parser.parse_fixtures(SAMPLE_RESULT_RESPONSE, league="Premier League", country="England")
        assert fixtures[0].status == "finished"

    def test_parse_results_scores_extracted(self):
        fixtures = self.parser.parse_fixtures(SAMPLE_RESULT_RESPONSE, league="Premier League", country="England")
        assert fixtures[0].home_score == 2
        assert fixtures[0].away_score == 1

    def test_parse_empty_response_returns_empty_list(self):
        fixtures = self.parser.parse_fixtures("", league="Premier League", country="England")
        assert fixtures == []

    def test_parse_malformed_entry_skipped(self):
        malformed = "ZEE\xf7\xacAA\xf7\xacAB\xf7\xac~"
        fixtures = self.parser.parse_fixtures(malformed, league="Test", country="Test")
        assert isinstance(fixtures, list)

    def test_fixture_to_dict_has_required_keys(self):
        fixtures = self.parser.parse_fixtures(SAMPLE_FIXTURE_RESPONSE, league="Premier League", country="England")
        d = fixtures[0].to_dict()
        assert "match_id" in d
        assert "home_team" in d
        assert "away_team" in d
        assert "league" in d
        assert "country" in d
        assert "match_datetime" in d
        assert "status" in d
        assert "home_score" in d
        assert "away_score" in d

