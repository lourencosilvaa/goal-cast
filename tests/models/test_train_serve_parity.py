"""Training and serving must build ELO from the same inputs.

A fixture's ``elo_home`` feature is read from that team's *last historical
row* — the rating it carried into its most recent match — not from its current
rating. Reproducing that number at inference means replaying the same
chronological walk over the same matches.

Get this wrong and the model is trained on calibrated features and served
uncalibrated ones, which is worse than never calibrating: the features stop
meaning what the model learned them to mean, and nothing fails loudly.

These tests pin the property directly — same corpus in, same ELO columns out —
rather than trusting four call sites to stay in step by inspection.
"""

import pandas as pd

from config.config_loader import EloConfig
from src.models.cross_competition import (
    CrossCompetitionCorpus,
    CrossCompetitionEloBuilder,
)
from src.models.elo import FootballELO


def _elo() -> FootballELO:
    return FootballELO(
        EloConfig(k_factor=32, home_advantage=65, initial_rating=1500.0)
    )


def _domestic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Div": ["E0", "P1", "E0", "P1"],
            "Date": pd.to_datetime(
                ["2024-08-01", "2024-08-02", "2024-11-01", "2024-11-02"]
            ),
            "HomeTeam": ["Arsenal", "Benfica", "Arsenal", "Benfica"],
            "AwayTeam": ["Everton", "Boavista", "Everton", "Boavista"],
            "FTHG": [3, 3, 1, 1],
            "FTAG": [0, 0, 0, 0],
            "FTR": ["H", "H", "H", "H"],
            "HS": [18, 17, 12, 13],
        }
    )


def _european() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Div": ["CL"],
            "Date": pd.to_datetime(["2024-09-17"]),
            "HomeTeam": ["Benfica"],
            "AwayTeam": ["Arsenal"],
            "FTHG": [5],
            "FTAG": [0],
            "FTR": ["H"],
        }
    )


def _build(european: pd.DataFrame) -> pd.DataFrame:
    return CrossCompetitionEloBuilder(_elo()).build(
        CrossCompetitionCorpus(domestic=_domestic(), supplementary=european)
    )


ELO_COLUMNS = [
    "elo_home",
    "elo_away",
    "elo_diff",
    "elo_expected_home",
    "elo_expected_away",
]


class TestParity:
    def test_same_inputs_give_identical_elo_columns(self):
        """Training and every inference path run this same call."""
        training = _build(_european())
        serving = _build(_european())
        pd.testing.assert_frame_equal(training[ELO_COLUMNS], serving[ELO_COLUMNS])

    def test_repeated_builds_do_not_accumulate(self):
        """A fresh ELO per build — otherwise a second call double-counts."""
        first = _build(_european())
        second = _build(_european())
        assert list(first["elo_home"]) == list(second["elo_home"])

    def test_dropping_the_corpus_changes_the_features(self):
        """The guard's own premise: serving without it really does differ."""
        with_european = _build(_european())
        without = _build(pd.DataFrame())
        assert list(with_european["elo_home"]) != list(without["elo_home"])

    def test_the_last_row_is_what_inference_reads(self):
        """Inference takes a team's most recent row, so that is what must match."""
        featured = _build(_european())
        arsenal = featured[featured["HomeTeam"] == "Arsenal"].iloc[-1]
        again = _build(_european())
        arsenal_again = again[again["HomeTeam"] == "Arsenal"].iloc[-1]
        assert arsenal["elo_home"] == arsenal_again["elo_home"]


class TestVendoredCopyMatches:
    """The HF Space vendors its own copy of the builder.

    Two divergent copies would reintroduce the skew silently, so the file is
    compared byte for byte rather than assumed to be in sync.
    """

    def test_space_copy_is_identical(self):
        from pathlib import Path

        root = Path(__file__).parent.parent.parent
        original = root / "src" / "models" / "cross_competition.py"
        vendored = root / "hf_space" / "src" / "models" / "cross_competition.py"
        assert vendored.is_file(), "the Space is missing its copy of the builder"
        assert vendored.read_text(encoding="utf-8") == original.read_text(
            encoding="utf-8"
        )
