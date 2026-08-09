"""Tests for the openfootball ``football.txt`` parser.

Every excerpt below is copied verbatim from the openfootball/champions-league
repository, because the format's traps are not guessable:

* the parenthesised scores are ``(90-minute, half-time)``, so the *first*
  number pair on an extra-time line is the score after extra time and the
  shootout comes first of all — a naive read records a 4-3 penalty win as a
  4-3 thrashing;
* date lines carry the year only when it changes, so a parser must carry it
  forward and roll it over in January;
* a match line inherits the kick-off time of the line above when it has none.

The corpus keeps the **90-minute** score, matching football-data.co.uk's
``FTHG``/``FTAG`` semantics, so domestic and European rows mean the same thing.
"""

import pandas as pd

from src.corpus.openfootball.parser import OpenFootballParser, SeasonSpec

_HEADER = """= UEFA Champions League 2024/25

# Date       Tue Sep 17 2024 - Sat May 31 2025 (256d)
# Teams      36


▪ League, Matchday 1
  Tue Sep 17 2024
    18:45  BSC Young Boys (SUI)    v Aston Villa FC (ENG)     0-3 (0-2)
           Juventus FC (ITA)       v PSV (NED)                3-1 (2-0)
  Wed Sep 18
    21:00  Manchester City FC (ENG) v FC Internazionale Milano (ITA)  0-0
"""


def _parse(
    text: str,
    competition: str = "CL",
    season: str = "2024-25",
    years: tuple[int, int] = (2024, 2025),
):
    spec = SeasonSpec(competition=competition, season=season, years=years)
    return OpenFootballParser().parse(text, spec)


class TestBasicParsing:
    def test_reads_every_match(self):
        assert len(_parse(_HEADER)) == 3

    def test_ignores_headers_and_comments(self):
        rows = _parse(_HEADER)
        assert all(row["HomeTeam"] for row in rows)

    def test_extracts_team_names_without_country_code(self):
        rows = _parse(_HEADER)
        assert rows[0]["HomeTeam"] == "BSC Young Boys"
        assert rows[0]["AwayTeam"] == "Aston Villa FC"

    def test_keeps_the_country_code_separately(self):
        """The code is what makes a suggestion safe to offer later."""
        rows = _parse(_HEADER)
        assert rows[0]["HomeCountry"] == "SUI"
        assert rows[0]["AwayCountry"] == "ENG"

    def test_extracts_full_time_goals(self):
        rows = _parse(_HEADER)
        assert (rows[0]["FTHG"], rows[0]["FTAG"]) == (0, 3)

    def test_extracts_half_time_goals(self):
        rows = _parse(_HEADER)
        assert (rows[0]["HTHG"], rows[0]["HTAG"]) == (0, 2)

    def test_match_without_half_time_score_leaves_it_none(self):
        rows = _parse(_HEADER)
        goalless = rows[2]
        assert (goalless["FTHG"], goalless["FTAG"]) == (0, 0)
        assert goalless["HTHG"] is None

    def test_tags_rows_with_the_competition_code(self):
        rows = _parse(_HEADER, competition="CL")
        assert {row["Div"] for row in rows} == {"CL"}


class TestDates:
    def test_uses_the_date_line(self):
        rows = _parse(_HEADER)
        assert rows[0]["Date"] == pd.Timestamp("2024-09-17")

    def test_carries_the_year_onto_a_bare_date_line(self):
        rows = _parse(_HEADER)
        assert rows[2]["Date"] == pd.Timestamp("2024-09-18")

    def test_rolls_the_year_over_in_january(self):
        """A European season spans two calendar years; only the first is given."""
        text = """▪ Finals, Round of 16
  Tue Dec 10 2024
    21:00  Arsenal FC (ENG)        v PSV (NED)                2-1 (1-0)
  Wed Jan 15
    21:00  PSV (NED)               v Arsenal FC (ENG)         0-1 (0-1)
"""
        rows = _parse(text)
        assert rows[0]["Date"] == pd.Timestamp("2024-12-10")
        assert rows[1]["Date"] == pd.Timestamp("2025-01-15")

    def test_does_not_roll_over_within_the_same_half_of_the_season(self):
        text = """▪ League, Matchday 1
  Tue Sep 17 2024
    21:00  Arsenal FC (ENG)        v PSV (NED)                2-1 (1-0)
  Wed Nov 06
    21:00  PSV (NED)               v Arsenal FC (ENG)         0-1 (0-1)
"""
        rows = _parse(text)
        assert rows[1]["Date"] == pd.Timestamp("2024-11-06")

    def test_match_before_any_date_line_is_skipped(self):
        text = "    21:00  Arsenal FC (ENG)  v PSV (NED)  2-1 (1-0)\n"
        assert _parse(text) == []


class TestExtraTimeAndPenalties:
    """The variants that would silently corrupt ratings if misread."""

    def test_extra_time_keeps_the_ninety_minute_score(self):
        # 2014 final: 1-1 after 90, Real won 4-1 in extra time.
        text = """▪ Finals, Final
  Sat May 24 2014
    20:45  Real Madrid (ESP)       v Atlético Madrid (ESP)    4-1 a.e.t. (1-1, 0-1)
"""
        row = _parse(text)[0]
        assert (row["FTHG"], row["FTAG"]) == (1, 1)

    def test_extra_time_keeps_the_half_time_score(self):
        text = """▪ Finals, Final
  Sat May 24 2014
    20:45  Real Madrid (ESP)       v Atlético Madrid (ESP)    4-1 a.e.t. (1-1, 0-1)
"""
        row = _parse(text)[0]
        assert (row["HTHG"], row["HTAG"]) == (0, 1)

    def test_penalty_shootout_is_not_mistaken_for_the_result(self):
        # 2012 final: 1-1 after 90, Chelsea won 4-3 on penalties.
        text = """▪ Finals, Final
  Sat May 19 2012
    20:45  Bayern München (GER)    v Chelsea FC (ENG)         3-4 pen. 1-1 a.e.t. (1-1, 0-0)
"""
        row = _parse(text)[0]
        assert (row["FTHG"], row["FTAG"]) == (1, 1)
        assert (row["HTHG"], row["HTAG"]) == (0, 0)

    def test_penalty_shootout_records_a_draw_not_a_win(self):
        text = """▪ Finals, Final
  Sat May 19 2012
    20:45  Bayern München (GER)    v Chelsea FC (ENG)         3-4 pen. 1-1 a.e.t. (1-1, 0-0)
"""
        row = _parse(text)[0]
        assert row["FTHG"] == row["FTAG"]

    def test_extra_time_win_after_a_ninety_minute_lead(self):
        text = """▪ Finals, Semifinals
  Tue Apr 24 2012
    20:45  APOEL Nikosia (CYP)     v Olympique Lyonnais (FRA)  4-3 pen. 1-0 a.e.t. (1-0, 1-0)
"""
        row = _parse(text)[0]
        assert (row["FTHG"], row["FTAG"]) == (1, 0)


class TestNonResults:
    def test_awarded_matches_are_excluded(self):
        """A forfeit carries no performance signal — including it would tell
        the ratings that one side outplayed the other."""
        text = """▪ Group A
  Tue Sep 17 2024
           Villarreal CF (ESP)     v Qarabağ FK (AZE)         3-0    [awarded]
           Juventus FC (ITA)       v PSV (NED)                3-1 (2-0)
"""
        rows = _parse(text)
        assert len(rows) == 1
        assert rows[0]["HomeTeam"] == "Juventus FC"

    def test_cancelled_matches_are_excluded(self):
        text = """▪ Group A
  Tue Sep 17 2024
           RB Leipzig (GER)        v Spartak Moskva (RUS)     [cancelled]
           Juventus FC (ITA)       v PSV (NED)                3-1 (2-0)
"""
        rows = _parse(text)
        assert len(rows) == 1

    def test_unplayed_fixture_is_excluded(self):
        text = """▪ Group A
  Tue Sep 17 2024
    21:00  Arsenal FC (ENG)        v PSV (NED)
"""
        assert _parse(text) == []


class TestLineContinuation:
    def test_match_without_a_time_inherits_the_previous_one(self):
        rows = _parse(_HEADER)
        assert rows[1]["HomeTeam"] == "Juventus FC"
        assert rows[1]["Date"] == pd.Timestamp("2024-09-17")


class TestMalformedInput:
    def test_empty_text_yields_nothing(self):
        assert _parse("") == []

    def test_unparseable_lines_are_skipped_not_fatal(self):
        text = """▪ Group A
  Tue Sep 17 2024
    this is not a match line at all
           Juventus FC (ITA)       v PSV (NED)                3-1 (2-0)
"""
        assert len(_parse(text)) == 1

    def test_unparseable_date_line_does_not_strand_later_matches(self):
        text = """▪ Group A
  Tue Sep 17 2024
           Juventus FC (ITA)       v PSV (NED)                3-1 (2-0)
  Notaday Xyz 99
           Arsenal FC (ENG)        v PSV (NED)                2-1 (1-0)
"""
        rows = _parse(text)
        assert len(rows) == 2
        assert rows[1]["Date"] == pd.Timestamp("2024-09-17")


class TestGroupSectionsRestartTheCalendar:
    """openfootball lists each group in full before starting the next.

    So a bare ``Wed Sep 14`` follows a ``Wed Dec 7`` several times per file.
    Reading that as "the month went backwards, so the year advanced" walks the
    2011-12 season forward into 2018. The season's own two calendar years are
    the only safe authority for a bare date.
    """

    SECTIONS = """▪ Group A
  Wed Sep 14 2011
    20:45  Bayern München (GER)    v Villarreal CF (ESP)      2-0 (1-0)
  Wed Dec 7
    20:45  Villarreal CF (ESP)     v Bayern München (GER)     0-2 (0-1)
▪ Group B
  Wed Sep 14
    20:45  Arsenal FC (ENG)        v Olympique Lyonnais (FRA)  1-0 (0-0)
  Wed Dec 7
    20:45  Olympique Lyonnais (FRA) v Arsenal FC (ENG)         2-1 (1-0)
"""

    def _rows(self):
        return _parse(self.SECTIONS, season="2011-12", years=(2011, 2012))

    def test_second_group_stays_in_the_first_year(self):
        rows = self._rows()
        assert rows[2]["Date"] == pd.Timestamp("2011-09-14")

    def test_no_match_escapes_the_season(self):
        rows = self._rows()
        assert all(
            pd.Timestamp("2011-07-01") <= row["Date"] <= pd.Timestamp("2012-06-30")
            for row in rows
        )

    def test_autumn_months_take_the_first_year(self):
        rows = self._rows()
        assert rows[1]["Date"] == pd.Timestamp("2011-12-07")

    def test_spring_months_take_the_second_year(self):
        text = """▪ Finals, Final
  Sat May 19
    20:45  Bayern München (GER)    v Chelsea FC (ENG)         1-1 (0-0)
"""
        rows = _parse(text, season="2011-12", years=(2011, 2012))
        assert rows[0]["Date"] == pd.Timestamp("2012-05-19")

    def test_an_explicit_year_always_wins(self):
        rows = self._rows()
        assert rows[0]["Date"] == pd.Timestamp("2011-09-14")


class TestScoreShapesFoundInTheRealCorpus:
    """All six notations that actually occur across the 15 shipped seasons."""

    def _score(self, notation: str):
        text = f"""▪ Finals, Final
  Sat May 19 2012
    20:45  Bayern München (GER)    v Chelsea FC (ENG)         {notation}
"""
        rows = _parse(text)
        return rows[0] if rows else None

    def test_full_time_with_half_time(self):
        row = self._score("0-3 (0-2)")
        assert (row["FTHG"], row["FTAG"], row["HTHG"], row["HTAG"]) == (0, 3, 0, 2)

    def test_full_time_only(self):
        row = self._score("0-0")
        assert (row["FTHG"], row["FTAG"]) == (0, 0)
        assert row["HTHG"] is None

    def test_extra_time_with_both_bracket_values(self):
        row = self._score("4-1 a.e.t. (1-1, 0-1)")
        assert (row["FTHG"], row["FTAG"], row["HTHG"], row["HTAG"]) == (1, 1, 0, 1)

    def test_penalties_with_both_bracket_values(self):
        row = self._score("3-4 pen. 1-1 a.e.t. (1-1, 0-0)")
        assert (row["FTHG"], row["FTAG"], row["HTHG"], row["HTAG"]) == (1, 1, 0, 0)

    def test_extra_time_with_only_the_ninety_minute_value(self):
        """``1-0 a.e.t. (0-0)`` — level at 90, settled in extra time."""
        row = self._score("1-0 a.e.t. (0-0)")
        assert (row["FTHG"], row["FTAG"]) == (0, 0)
        assert row["HTHG"] is None

    def test_penalties_with_only_the_ninety_minute_value(self):
        row = self._score("4-2 pen. 1-1 a.e.t. (1-1)")
        assert (row["FTHG"], row["FTAG"]) == (1, 1)
        assert row["HTHG"] is None


class TestParserErrorBranches:
    """Malformed lines are skipped, never guessed at."""

    def test_unknown_month_is_not_a_date_line(self):
        text = """▪ Group A
  Tue Xxx 17 2024
           Juventus FC (ITA)       v PSV (NED)                3-1 (2-0)
"""
        assert _parse(text) == []

    def test_impossible_day_is_rejected(self):
        text = """▪ Group A
  Tue Feb 30 2024
           Juventus FC (ITA)       v PSV (NED)                3-1 (2-0)
"""
        assert _parse(text) == []

    def test_score_with_no_numbers_is_skipped(self):
        text = """▪ Group A
  Tue Sep 17 2024
           Juventus FC (ITA)       v PSV (NED)                postponed
"""
        assert _parse(text) == []

    def test_team_without_a_country_code_still_parses(self):
        text = """▪ Group A
  Tue Sep 17 2024
           Juventus FC             v PSV                      3-1 (2-0)
"""
        row = _parse(text)[0]
        assert row["HomeTeam"] == "Juventus FC"
        assert row["HomeCountry"] is None

    def test_extra_time_without_a_bracket_is_skipped(self):
        """Regulation score is unknowable, so the row is dropped, not guessed."""
        text = """▪ Finals, Final
  Sat May 19 2012
    20:45  Bayern München (GER)    v Chelsea FC (ENG)         1-1 a.e.t.
"""
        assert _parse(text) == []

    def test_abandoned_matches_are_excluded(self):
        text = """▪ Group A
  Tue Sep 17 2024
           Juventus FC (ITA)       v PSV (NED)                1-0    [abandoned]
"""
        assert _parse(text) == []
