"""``/predict-custom`` when the two teams come from different leagues.

The domestic ensemble is trained only on rows where both sides share a league.
It has never seen a cross-league fixture and carries no feature that would tell
it one had arrived, so answering with it would be extrapolation presented as a
prediction. These tests pin the routing, the refusals, and the label that says
which model actually produced the number.
"""

from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ── stubs ────────────────────────────────────────────────────────────────


class _Triple:
    """A 1X2 triple that is already normalised.

    Duck-typed rather than the real ``OutcomeProbabilities``: that class exists
    twice — once in the project, once mirrored inside ``hf_space`` — and a test
    holding the wrong copy would compare two identical-looking types.
    """

    def __init__(self, home_win: float, draw: float, away_win: float) -> None:
        self.home_win = home_win
        self.draw = draw
        self.away_win = away_win

    def normalized(self) -> "_Triple":
        return self


class _DixonColes:
    """Records which teams it has parameters for, and what it would say."""

    def __init__(
        self, known: tuple[str, ...] = (), outcome: _Triple | None = None
    ) -> None:
        self._known = set(known)
        self._outcome = outcome or _Triple(0.5, 0.3, 0.2)

    def knows(self, team: str) -> bool:
        return team in self._known

    def predict_outcome(self, home_team: str, away_team: str) -> _Triple:
        return self._outcome


class _Elo:
    """Ratings on one scale, as the cross-league corpus calibration leaves them."""

    def __init__(self, ratings: dict[str, float], home_advantage: int = 65) -> None:
        self.ratings = ratings
        self.config = SimpleNamespace(home_advantage=home_advantage)

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, 1500.0)

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


class _EnsemblePrediction:
    """What the domestic ``MatchPredictor`` hands back."""

    @staticmethod
    def to_dict() -> dict[str, object]:
        return {
            "predicted_outcome": "Home Win",
            "confidence": 0.61,
            "probabilities": {"home_win": 0.61, "draw": 0.24, "away_win": 0.15},
        }


class _Predictor:
    def __init__(self, poisson: object | None) -> None:
        self.poisson = poisson
        self.feature_names = ["elo_home"]
        self.calls: list[str] = []

    def predict(self, feature_df: pd.DataFrame) -> list[_EnsemblePrediction]:
        self.calls.append("predict")
        return [_EnsemblePrediction()]


# ── explicit configuration ───────────────────────────────────────────────

_LEAGUES = {"E0": "Premier League", "P1": "Liga Portugal"}


def _space_config(
    space_app,
    *,
    enabled: bool = True,
    dixon_coles_weight: float = 0.0,
    min_matches_per_team: int = 10,
    elo_draw_rate: float = 0.22,
):
    """Every value the endpoint reads, stated rather than defaulted (§7.3)."""
    return space_app.SpaceConfig(
        data={
            "base_url": "https://example.invalid",
            "seasons": ["2526"],
            "leagues": dict(_LEAGUES),
            "columns_to_keep": ["Date", "HomeTeam", "AwayTeam"],
        },
        features={
            "rolling_window": 5,
            "elo": {"k_factor": 32, "home_advantage": 65, "initial_rating": 1500.0},
            "xg_proxy": {"sot_conversion": 0.30, "shot_conversion": 0.03},
            "fatigue": {
                "max_rest_days": 30,
                "default_rest_days": 14,
                "fatigue_threshold": 3,
            },
        },
        european_prediction={
            "enabled": enabled,
            "dixon_coles_weight": dixon_coles_weight,
            "min_matches_per_team": min_matches_per_team,
            "elo_draw_rate": elo_draw_rate,
        },
    )


def _install(
    space_app,
    *,
    poisson: object | None = None,
    elo: _Elo | None = None,
    match_counts: dict[str, int] | None = None,
    config=None,
) -> _Predictor:
    """Put the module into the state ``lifespan`` would leave it in."""
    predictor = _Predictor(poisson)
    space_app._PREDICTOR = predictor
    space_app._CONFIG = config if config is not None else _space_config(space_app)
    space_app._ENRICHED_DATA = pd.DataFrame()
    space_app._ELO = elo
    space_app._MATCH_COUNTS = match_counts or {}
    space_app.DIVISION_MAP.clear()
    space_app.DIVISION_MAP.update(_LEAGUES)
    return predictor


def _client(space_app) -> TestClient:
    """No ``with``: entering the context would run the real lifespan, which
    downloads a model from HuggingFace."""
    return TestClient(space_app.app, raise_server_exceptions=False)


def _post(space_app, **body) -> object:
    return _client(space_app).post("/predict-custom", json=body)


# ── tests ────────────────────────────────────────────────────────────────


class TestRouting:
    """Which model answers is decided by the *pairing*, not by a badge."""

    def test_same_league_uses_the_ensemble(self, space_app):
        predictor = _install(space_app, poisson=_DixonColes(known=("Arsenal",)))
        res = _post(
            space_app,
            home_team="Arsenal",
            away_team="Chelsea",
            league_code="E0",
            away_league_code="E0",
        )
        assert res.status_code == 200
        assert predictor.calls == ["predict"]

    def test_absent_away_league_means_the_same_league(self, space_app):
        """What every existing caller has always meant by sending one code."""
        predictor = _install(space_app, poisson=_DixonColes())
        res = _post(
            space_app, home_team="Arsenal", away_team="Chelsea", league_code="E0"
        )
        assert res.status_code == 200
        assert predictor.calls == ["predict"]

    def test_domestic_answer_carries_no_model_label(self, space_app):
        """The ensemble is the default, and naming it on every response would
        make the cross-league label mean nothing by contrast."""
        _install(space_app, poisson=_DixonColes())
        res = _post(
            space_app, home_team="Arsenal", away_team="Chelsea", league_code="E0"
        )
        assert res.json()["model"] is None

    def test_different_leagues_bypass_the_ensemble(self, space_app):
        predictor = _install(
            space_app,
            poisson=_DixonColes(known=("Arsenal", "Benfica")),
            elo=_Elo({"Arsenal": 1800.0, "Benfica": 1600.0}),
            match_counts={"Arsenal": 300, "Benfica": 300},
        )
        res = _post(
            space_app,
            home_team="Arsenal",
            away_team="Benfica",
            league_code="E0",
            away_league_code="P1",
        )
        assert res.status_code == 200
        assert predictor.calls == [], "the ensemble must not see a cross-league tie"

    def test_cross_league_answer_names_both_leagues(self, space_app):
        _install(
            space_app,
            poisson=_DixonColes(known=("Arsenal", "Benfica")),
            elo=_Elo({"Arsenal": 1800.0, "Benfica": 1600.0}),
            match_counts={"Arsenal": 300, "Benfica": 300},
        )
        body = _post(
            space_app,
            home_team="Arsenal",
            away_team="Benfica",
            league_code="E0",
            away_league_code="P1",
        ).json()
        assert body["league"] == "Premier League"
        assert body["away_league"] == "Liga Portugal"


class TestTheLabelNamesWhatPredicted:
    """A number produced by ELO alone must not arrive labelled as a blend."""

    def test_zero_weight_is_labelled_elo(self, space_app):
        _install(
            space_app,
            poisson=_DixonColes(known=("Arsenal", "Benfica")),
            elo=_Elo({"Arsenal": 1800.0, "Benfica": 1600.0}),
            match_counts={"Arsenal": 300, "Benfica": 300},
            config=_space_config(space_app, dixon_coles_weight=0.0),
        )
        body = _post(
            space_app,
            home_team="Arsenal",
            away_team="Benfica",
            league_code="E0",
            away_league_code="P1",
        ).json()
        assert body["model"] == "elo"

    def test_full_weight_is_labelled_dixon_coles(self, space_app):
        _install(
            space_app,
            poisson=_DixonColes(
                known=("Arsenal", "Benfica"), outcome=_Triple(0.2, 0.3, 0.5)
            ),
            elo=_Elo({"Arsenal": 1800.0, "Benfica": 1600.0}),
            match_counts={"Arsenal": 300, "Benfica": 300},
            config=_space_config(space_app, dixon_coles_weight=1.0),
        )
        body = _post(
            space_app,
            home_team="Arsenal",
            away_team="Benfica",
            league_code="E0",
            away_league_code="P1",
        ).json()
        assert body["model"] == "dixon-coles"
        assert body["predicted_outcome"] == "Away Win"


class TestTheNumbersComeFromTheRatings:

    def test_the_stronger_side_is_favoured(self, space_app):
        _install(
            space_app,
            poisson=_DixonColes(known=("Arsenal", "Benfica")),
            elo=_Elo({"Arsenal": 1900.0, "Benfica": 1500.0}),
            match_counts={"Arsenal": 300, "Benfica": 300},
        )
        body = _post(
            space_app,
            home_team="Arsenal",
            away_team="Benfica",
            league_code="E0",
            away_league_code="P1",
        ).json()
        probabilities = body["probabilities"]
        assert probabilities["home_win"] > probabilities["away_win"]
        assert body["predicted_outcome"] == "Home Win"

    def test_the_configured_draw_rate_is_used(self, space_app):
        """ELO models no draws at all, so the share is stated, not derived."""
        _install(
            space_app,
            poisson=_DixonColes(known=("Arsenal", "Benfica")),
            elo=_Elo({"Arsenal": 1700.0, "Benfica": 1700.0}),
            match_counts={"Arsenal": 300, "Benfica": 300},
            config=_space_config(space_app, elo_draw_rate=0.30),
        )
        body = _post(
            space_app,
            home_team="Arsenal",
            away_team="Benfica",
            league_code="E0",
            away_league_code="P1",
        ).json()
        assert body["probabilities"]["draw"] == pytest.approx(0.30, abs=1e-6)


class TestRefusals:
    """A club with no history gets a stated reason, never an invented number."""

    def test_untracked_club_is_refused_with_its_name(self, space_app):
        _install(
            space_app,
            poisson=_DixonColes(known=("Arsenal",)),
            elo=_Elo({"Arsenal": 1800.0}),
            match_counts={"Arsenal": 300},
        )
        res = _post(
            space_app,
            home_team="Arsenal",
            away_team="Kairat",
            league_code="E0",
            away_league_code="P1",
        )
        assert res.status_code == 422
        assert "Kairat" in res.json()["detail"]

    def test_thin_history_is_refused_with_the_count(self, space_app):
        _install(
            space_app,
            poisson=_DixonColes(known=("Arsenal", "Benfica")),
            elo=_Elo({"Arsenal": 1800.0, "Benfica": 1600.0}),
            match_counts={"Arsenal": 300, "Benfica": 4},
            config=_space_config(space_app, min_matches_per_team=10),
        )
        res = _post(
            space_app,
            home_team="Arsenal",
            away_team="Benfica",
            league_code="E0",
            away_league_code="P1",
        )
        assert res.status_code == 422
        detail = res.json()["detail"]
        assert "4" in detail and "10" in detail


class TestUnavailability:
    """Cross-league needs pieces the domestic path does not. Say so."""

    def test_missing_poisson_is_503_not_a_guess(self, space_app):
        """Dixon-Coles answers the *history* question even at weight zero — it
        is the only component that knows which clubs exist."""
        _install(
            space_app,
            poisson=None,
            elo=_Elo({"Arsenal": 1800.0, "Benfica": 1600.0}),
            match_counts={"Arsenal": 300, "Benfica": 300},
        )
        res = _post(
            space_app,
            home_team="Arsenal",
            away_team="Benfica",
            league_code="E0",
            away_league_code="P1",
        )
        assert res.status_code == 503

    def test_missing_elo_is_503(self, space_app):
        _install(
            space_app,
            poisson=_DixonColes(known=("Arsenal", "Benfica")),
            elo=None,
            match_counts={"Arsenal": 300, "Benfica": 300},
        )
        res = _post(
            space_app,
            home_team="Arsenal",
            away_team="Benfica",
            league_code="E0",
            away_league_code="P1",
        )
        assert res.status_code == 503

    def test_disabled_track_is_503(self, space_app):
        _install(
            space_app,
            poisson=_DixonColes(known=("Arsenal", "Benfica")),
            elo=_Elo({"Arsenal": 1800.0, "Benfica": 1600.0}),
            match_counts={"Arsenal": 300, "Benfica": 300},
            config=_space_config(space_app, enabled=False),
        )
        res = _post(
            space_app,
            home_team="Arsenal",
            away_team="Benfica",
            league_code="E0",
            away_league_code="P1",
        )
        assert res.status_code == 503


class TestAnAmbiguousAwayLeagueStaysDomestic:
    """Every unclear case resolves to the path that can refuse an unknown team.

    Routing to the cross-league model on a blank or malformed second code would
    bypass the ensemble on the strength of a typo.
    """

    def _predictor_with(self, space_app) -> _Predictor:
        return _install(
            space_app,
            poisson=_DixonColes(known=("Arsenal", "Chelsea")),
            elo=_Elo({"Arsenal": 1800.0, "Chelsea": 1600.0}),
            match_counts={"Arsenal": 300, "Chelsea": 300},
        )

    def test_whitespace_only_away_league(self, space_app):
        predictor = self._predictor_with(space_app)
        res = _post(
            space_app,
            home_team="Arsenal",
            away_team="Chelsea",
            league_code="E0",
            away_league_code="   ",
        )
        assert res.status_code == 200
        assert predictor.calls == ["predict"]

    def test_empty_away_league(self, space_app):
        predictor = self._predictor_with(space_app)
        res = _post(
            space_app,
            home_team="Arsenal",
            away_team="Chelsea",
            league_code="E0",
            away_league_code="",
        )
        assert res.status_code == 200
        assert predictor.calls == ["predict"]

    def test_the_same_code_with_stray_whitespace(self, space_app):
        """`" E0"` and `"E0"` are one league, not two."""
        predictor = self._predictor_with(space_app)
        res = _post(
            space_app,
            home_team="Arsenal",
            away_team="Chelsea",
            league_code="E0",
            away_league_code=" E0 ",
        )
        assert res.status_code == 200
        assert predictor.calls == ["predict"]

    def test_a_null_away_league(self, space_app):
        predictor = self._predictor_with(space_app)
        res = _post(
            space_app,
            home_team="Arsenal",
            away_team="Chelsea",
            league_code="E0",
            away_league_code=None,
        )
        assert res.status_code == 200
        assert predictor.calls == ["predict"]


class TestBoundaries:

    def test_an_unmapped_code_shows_the_code_itself(self, space_app):
        """Better than a blank: the raw division is a usable answer, and it is
        the right display for a league the config does not cover."""
        _install(
            space_app,
            poisson=_DixonColes(known=("Arsenal", "Benfica")),
            elo=_Elo({"Arsenal": 1800.0, "Benfica": 1600.0}),
            match_counts={"Arsenal": 300, "Benfica": 300},
        )
        body = _post(
            space_app,
            home_team="Arsenal",
            away_team="Benfica",
            league_code="E0",
            away_league_code="XX9",
        ).json()
        assert body["league"] == "Premier League"
        assert body["away_league"] == "XX9"

    def test_a_club_absent_from_the_counts_is_not_refused(self, space_app):
        """The count gate is evidence *for* a refusal, so its absence can only
        make the gate more permissive. Whether the club exists at all is a
        different question, and Dixon-Coles has already answered it."""
        _install(
            space_app,
            poisson=_DixonColes(known=("Arsenal", "Benfica")),
            elo=_Elo({"Arsenal": 1800.0, "Benfica": 1600.0}),
            match_counts={"Arsenal": 300},
        )
        res = _post(
            space_app,
            home_team="Arsenal",
            away_team="Benfica",
            league_code="E0",
            away_league_code="P1",
        )
        assert res.status_code == 200

    def test_the_triple_sums_to_one(self, space_app):
        _install(
            space_app,
            poisson=_DixonColes(known=("Arsenal", "Benfica")),
            elo=_Elo({"Arsenal": 2100.0, "Benfica": 1450.0}),
            match_counts={"Arsenal": 300, "Benfica": 300},
        )
        probabilities = _post(
            space_app,
            home_team="Arsenal",
            away_team="Benfica",
            league_code="E0",
            away_league_code="P1",
        ).json()["probabilities"]
        assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-3)

    def test_a_config_predating_the_block_still_loads(self, space_app):
        """The Space and this repo deploy separately; an older ``config.yaml``
        on the box must not take the endpoint down."""
        config = space_app.SpaceConfig(
            data={
                "base_url": "https://example.invalid",
                "seasons": ["2526"],
                "leagues": dict(_LEAGUES),
                "columns_to_keep": ["Date"],
            },
            features={
                "rolling_window": 5,
                "elo": {"k_factor": 32, "home_advantage": 65, "initial_rating": 1500.0},
                "xg_proxy": {"sot_conversion": 0.30, "shot_conversion": 0.03},
                "fatigue": {
                    "max_rest_days": 30,
                    "default_rest_days": 14,
                    "fatigue_threshold": 3,
                },
            },
        )
        assert config.european_prediction.enabled is True
        assert config.european_prediction.dixon_coles_weight == 0.0
        assert config.european_prediction.elo_draw_rate == 0.22


class TestCalibratedRatingsSurviveStartup:
    """``build`` returns domestic rows only.

    A European result updates the ratings without ever appearing in that
    output, so recomputing ELO from it would silently discard every
    cross-league link — leaving exactly the uncalibrated ratings the corpus
    work exists to replace. The fitted instance is the only place they live.
    """

    def test_engineer_features_hands_back_the_fitted_instance(self, space_app):
        frame, elo = space_app._engineer_features(
            pd.DataFrame(), _space_config(space_app)
        )
        assert isinstance(frame, pd.DataFrame)
        assert elo is not None
        assert elo.get_rating("Arsenal") == 1500.0

    def test_the_walk_writes_into_that_instance(self, space_app):
        _, elo = space_app._engineer_features(_domestic(), _space_config(space_app))
        assert elo.ratings, "the returned ELO carries no ratings from the walk"


def _domestic() -> pd.DataFrame:
    """A corpus small enough to read, large enough to move a rating.

    Carries the shot/foul/corner columns because the ensemble's feature matrix
    is built from them — this is exactly the set the European feed lacks, which
    is why European results feed ELO and Dixon-Coles but never the ensemble.
    """
    rows = 4
    return pd.DataFrame(
        {
            "Div": ["E0"] * rows,
            "League": ["Premier League"] * rows,
            "Date": pd.to_datetime(
                ["2024-08-01", "2024-08-08", "2024-08-15", "2024-08-22"]
            ),
            "HomeTeam": ["Arsenal", "Chelsea", "Arsenal", "Chelsea"],
            "AwayTeam": ["Chelsea", "Arsenal", "Chelsea", "Arsenal"],
            "FTHG": [3, 0, 2, 1],
            "FTAG": [0, 1, 1, 1],
            "FTR": ["H", "A", "H", "D"],
            "HS": [18, 9, 15, 11],
            "AS": [7, 14, 8, 12],
            "HST": [8, 3, 6, 4],
            "AST": [2, 5, 3, 4],
            "HC": [9, 4, 7, 5],
            "AC": [3, 6, 4, 5],
            "HF": [10, 12, 9, 11],
            "AF": [13, 9, 12, 10],
        }
    )
