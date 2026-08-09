"""The model label must survive the API boundary.

``run_inference`` records which model produced each prediction — since the
European weight sweep (docs §14) domestic fixtures come from the calibrated
ensemble and European ones from ELO alone, and they are displayed identically.
The label exists so a UI can never present one as the other.

It was being dropped. ``MatchPredictionResponse`` had no ``model`` field, and
FastAPI serialises through the response model, so the key written to Supabase
never reached the frontend. The label was correct at every step except the one
that mattered.

Optional rather than required: domestic ``MatchPrediction.to_dict()`` does not
emit it, and rows already in Supabase predate it. ``None`` means the domestic
ensemble — what every prediction was until this branch.
"""

from src.backend.api.predictions import (
    LeaguePredictionsResponse,
    MatchPredictionResponse,
    _build_league_response,
)


def _match(**overrides) -> dict:
    """A Supabase match dict, stated in full (§7.3)."""
    match = {
        "home_team": "Sp Lisbon",
        "away_team": "Man City",
        "league": "Champions League",
        "time": "20:00",
        "probabilities": {"home_win": 0.383, "draw": 0.22, "away_win": 0.397},
        "predicted_outcome": "Away Win",
        "confidence": 0.397,
    }
    match.update(overrides)
    return match


class _Result:
    """Stands in for the service's LeaguePredictions."""

    def __init__(self, matches: list[dict]) -> None:
        self.league_code = "UECL"
        self.league_name = "Conference League"
        self.matches = matches


class TestTheLabelSurvivesSerialisation:
    def test_the_response_model_keeps_it(self):
        """The exact check that failed before: model_dump() dropped the key."""
        response = MatchPredictionResponse(**_match(model="elo"))
        assert response.model_dump()["model"] == "elo"

    def test_it_survives_the_supabase_translation(self):
        built = _build_league_response(_Result([_match(model="elo")]))
        assert built.matches[0].model == "elo"

    def test_it_reaches_the_serialised_league_payload(self):
        built = _build_league_response(_Result([_match(model="elo")]))
        assert built.model_dump()["matches"][0]["model"] == "elo"

    def test_a_blended_label_passes_through_unchanged(self):
        """The API must not interpret the label, only carry it."""
        built = _build_league_response(_Result([_match(model="dixon-coles+elo")]))
        assert built.matches[0].model == "dixon-coles+elo"


class TestAbsenceIsNotAnError:
    def test_a_match_without_a_model_still_builds(self):
        """Every domestic prediction, and every row written before this field."""
        built = _build_league_response(_Result([_match()]))
        assert built.matches[0].model is None

    def test_the_response_model_defaults_it(self):
        assert MatchPredictionResponse(**_match()).model is None

    def test_a_league_may_mix_labelled_and_unlabelled_matches(self):
        built = _build_league_response(
            _Result([_match(model="elo"), _match(home_team="Benfica")])
        )
        assert [m.model for m in built.matches] == ["elo", None]

    def test_an_empty_label_is_carried_as_given(self):
        """Not coerced to None: the API reports what was stored."""
        built = _build_league_response(_Result([_match(model="")]))
        assert built.matches[0].model == ""

    def test_the_rest_of_the_response_is_untouched(self):
        built = _build_league_response(_Result([_match(model="elo")]))
        assert isinstance(built, LeaguePredictionsResponse)
        assert built.matches[0].predicted_outcome == "Away Win"
        assert built.matches[0].probabilities.away_win == 0.397
