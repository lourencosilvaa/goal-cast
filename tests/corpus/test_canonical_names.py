"""Tests for rewriting European rows into canonical team keys.

This is the step that makes an approved alias mean something. ELO is keyed by
the team-name string, so until "Sport Lisboa e Benfica" becomes "Benfica" the
European result creates a *separate* rating and the domestic pools never link
— the calibration runs cleanly and achieves nothing.

Rows that cannot be translated are kept, not dropped. A club from a league this
project does not carry (Shakhtar, Salzburg, Bodø/Glimt) legitimately has no
canonical key, and its matches still carry information: they let that club
build a rating of its own from European play, which is what makes a
Benfica-vs-Shakhtar result informative rather than noise.
"""

import pandas as pd

from config.config_loader import EuropeanConfig, TeamAliasConfig
from src.corpus.canonical_names import CanonicalCorpusTranslator, TranslationReport
from src.teams.european_names import EuropeanNameResolver
from src.teams.resolver import TeamAlias, TeamAliasRepository, TeamNameResolver

_REGISTRY = {
    "P1": ["Benfica", "Sp Lisbon", "Porto"],
    "E0": ["Arsenal", "Liverpool"],
}


class _Aliases(TeamAliasRepository):
    def __init__(self, aliases: list[TeamAlias]) -> None:
        self._aliases = aliases

    def get_aliases(self) -> list[TeamAlias]:
        return self._aliases


def _translator(aliases: list[TeamAlias] | None = None) -> CanonicalCorpusTranslator:
    config = EuropeanConfig(
        country_leagues={"POR": ["P1"], "ENG": ["E0"]}, alias_scope="EU"
    )
    names = TeamNameResolver(_REGISTRY, _Aliases(aliases or []), TeamAliasConfig())
    return CanonicalCorpusTranslator(EuropeanNameResolver(config, names))


def _corpus() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Div": ["CL", "CL", "CL"],
            "Season": ["2024-25"] * 3,
            "Date": pd.to_datetime(["2024-09-17", "2024-09-18", "2024-09-19"]),
            "HomeTeam": [
                "Sport Lisboa e Benfica",
                "Liverpool FC",
                "FK Shakhtar Donetsk",
            ],
            "AwayTeam": ["Liverpool FC", "Arsenal", "Sport Lisboa e Benfica"],
            "HomeCountry": ["POR", "ENG", "UKR"],
            "AwayCountry": ["ENG", "ENG", "POR"],
            "FTHG": [2, 1, 0],
            "FTAG": [1, 1, 3],
            "FTR": ["H", "D", "A"],
            "HTHG": [1, 0, 0],
            "HTAG": [0, 0, 2],
        }
    )


_BENFICA = TeamAlias(
    league_code="EU-POR", raw_name="Sport Lisboa e Benfica", canonical_name="Benfica"
)
_LIVERPOOL = TeamAlias(
    league_code="EU-ENG", raw_name="Liverpool FC", canonical_name="Liverpool"
)


class TestTranslation:
    def test_approved_alias_is_applied(self):
        result = _translator([_BENFICA]).translate(_corpus())
        assert result.frame.iloc[0]["HomeTeam"] == "Benfica"

    def test_applies_on_both_sides(self):
        result = _translator([_BENFICA]).translate(_corpus())
        assert result.frame.iloc[2]["AwayTeam"] == "Benfica"

    def test_exact_matches_need_no_alias(self):
        """'Arsenal' is already the canonical spelling."""
        result = _translator([]).translate(_corpus())
        assert result.frame.iloc[1]["AwayTeam"] == "Arsenal"

    def test_unapproved_name_is_left_alone(self):
        result = _translator([]).translate(_corpus())
        assert result.frame.iloc[0]["HomeTeam"] == "Sport Lisboa e Benfica"

    def test_untrackable_club_keeps_its_own_name(self):
        """Shakhtar has no domestic history here, so it is its own team."""
        result = _translator([_BENFICA, _LIVERPOOL]).translate(_corpus())
        assert result.frame.iloc[2]["HomeTeam"] == "FK Shakhtar Donetsk"

    def test_no_rows_are_dropped(self):
        result = _translator([_BENFICA]).translate(_corpus())
        assert len(result.frame) == len(_corpus())

    def test_results_are_untouched(self):
        result = _translator([_BENFICA, _LIVERPOOL]).translate(_corpus())
        assert list(result.frame["FTHG"]) == [2, 1, 0]
        assert list(result.frame["FTR"]) == ["H", "D", "A"]

    def test_other_columns_survive(self):
        result = _translator([_BENFICA]).translate(_corpus())
        assert list(result.frame.columns) == list(_corpus().columns)

    def test_input_frame_is_not_mutated(self):
        corpus = _corpus()
        _translator([_BENFICA]).translate(corpus)
        assert corpus.iloc[0]["HomeTeam"] == "Sport Lisboa e Benfica"

    def test_empty_corpus_is_handled(self):
        result = _translator([]).translate(pd.DataFrame())
        assert result.frame.empty
        assert result.report.translated == 0

    def test_a_name_maps_the_same_way_everywhere_it_appears(self):
        result = _translator([_BENFICA]).translate(_corpus())
        names = set(result.frame["HomeTeam"]) | set(result.frame["AwayTeam"])
        assert "Sport Lisboa e Benfica" not in names


class TestReport:
    def test_counts_translated_names(self):
        result = _translator([_BENFICA, _LIVERPOOL]).translate(_corpus())
        assert isinstance(result.report, TranslationReport)
        assert result.report.translated == 3  # Benfica, Liverpool, Arsenal

    def test_counts_untranslated_names(self):
        result = _translator([_BENFICA, _LIVERPOOL]).translate(_corpus())
        assert result.report.untranslated == 1  # Shakhtar

    def test_lists_names_that_could_still_be_linked(self):
        """The actionable set — an approval would connect each of these."""
        result = _translator([]).translate(_corpus())
        assert "Sport Lisboa e Benfica" in result.report.linkable
        assert "Liverpool FC" in result.report.linkable

    def test_untrackable_names_are_not_listed_as_linkable(self):
        result = _translator([_BENFICA, _LIVERPOOL]).translate(_corpus())
        assert "FK Shakhtar Donetsk" not in result.report.linkable

    def test_counts_the_matches_still_unlinked(self):
        """Match appearances, not names — what the calibration actually loses."""
        result = _translator([]).translate(_corpus())
        assert result.report.unlinked_appearances == 4

    def test_full_translation_leaves_nothing_linkable(self):
        result = _translator([_BENFICA, _LIVERPOOL]).translate(_corpus())
        assert result.report.linkable == []
        assert result.report.unlinked_appearances == 0

    def test_exposes_the_mapping_it_applied(self):
        result = _translator([_BENFICA]).translate(_corpus())
        assert result.report.mapping["Sport Lisboa e Benfica"] == "Benfica"

    def test_mapping_omits_untranslated_names(self):
        result = _translator([]).translate(_corpus())
        assert "Sport Lisboa e Benfica" not in result.report.mapping


class TestBuildTranslator:
    """The wiring function every caller uses.

    It must produce a working translator without Supabase, because training
    and the offline scripts run without it — approvals then come from the
    committed seed alone rather than the run failing outright.
    """

    def _config(self, tmp_path, registry: dict, seed: str = "aliases: {}\n"):
        import json

        from config.config_loader import TeamsConfig

        registry_path = tmp_path / "historical.json"
        registry_path.write_text(json.dumps(registry), encoding="utf-8")
        seed_path = tmp_path / "aliases.yaml"
        seed_path.write_text(seed, encoding="utf-8")

        class _Config:
            teams = TeamsConfig(
                registry_path=str(registry_path),
                historical_registry_path=str(registry_path),
                aliases=TeamAliasConfig(seed_path=str(seed_path)),
            )
            european = EuropeanConfig(
                country_leagues={"POR": ["P1"], "ENG": ["E0"]}, alias_scope="EU"
            )

        return _Config()

    def test_builds_without_supabase(self, tmp_path):
        from unittest.mock import patch

        from src.corpus.canonical_names import build_translator

        config = self._config(tmp_path, _REGISTRY)
        with patch(
            "src.backend.core.supabase_client.get_supabase_client",
            side_effect=RuntimeError("not configured"),
        ):
            translator = build_translator(config)
        assert isinstance(translator, CanonicalCorpusTranslator)

    def test_uses_the_historical_registry(self, tmp_path):
        """A relegated club must stay matchable to its own history."""
        from unittest.mock import patch

        from src.corpus.canonical_names import build_translator

        config = self._config(tmp_path, {"N1": ["Vitesse", "Ajax"]})
        config.european = EuropeanConfig(
            country_leagues={"NED": ["N1"]}, alias_scope="EU"
        )
        corpus = pd.DataFrame(
            {
                "Div": ["CL"],
                "Season": ["2023-24"],
                "Date": pd.to_datetime(["2023-09-17"]),
                "HomeTeam": ["Vitesse"],
                "AwayTeam": ["Ajax"],
                "HomeCountry": ["NED"],
                "AwayCountry": ["NED"],
                "FTHG": [1],
                "FTAG": [2],
                "FTR": ["A"],
                "HTHG": [0],
                "HTAG": [1],
            }
        )
        with patch(
            "src.backend.core.supabase_client.get_supabase_client",
            side_effect=RuntimeError("not configured"),
        ):
            result = build_translator(config).translate(corpus)
        assert result.report.linkable == []

    def test_reads_approved_aliases_from_the_seed(self, tmp_path):
        from unittest.mock import patch

        from src.corpus.canonical_names import build_translator

        seed = 'aliases:\n  EU-POR:\n    "Sport Lisboa e Benfica": "Benfica"\n'
        config = self._config(tmp_path, _REGISTRY, seed=seed)
        with patch(
            "src.backend.core.supabase_client.get_supabase_client",
            side_effect=RuntimeError("not configured"),
        ):
            result = build_translator(config).translate(_corpus())
        assert result.report.mapping["Sport Lisboa e Benfica"] == "Benfica"
