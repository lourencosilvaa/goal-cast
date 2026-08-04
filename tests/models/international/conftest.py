import pandas as pd
import pytest


@pytest.fixture
def raw_international_rows() -> pd.DataFrame:
    """Raw Kaggle-shaped international results (pre-normalization)."""
    return pd.DataFrame(
        {
            "date": [
                "1872-11-30",
                "2018-06-14",
                "2018-06-15",
                "2022-12-18",
                "2024-07-14",
                "2026-03-21",
            ],
            "home_team": [
                "Scotland",
                "Russia",
                "Portugal",
                "Argentina",
                "Spain",
                "Brazil",
            ],
            "away_team": [
                "England",
                "Saudi Arabia",
                "Spain",
                "France",
                "England",
                "Portugal",
            ],
            "home_score": [0, 5, 3, 3, 2, 1],
            "away_score": [0, 0, 3, 3, 1, 1],
            "tournament": [
                "Friendly",
                "FIFA World Cup",
                "FIFA World Cup",
                "FIFA World Cup",
                "UEFA Euro",
                "Friendly",
            ],
            "city": [
                "Glasgow",
                "Moscow",
                "Sochi",
                "Lusail",
                "Berlin",
                "Lisbon",
            ],
            "country": [
                "Scotland",
                "Russia",
                "Russia",
                "Qatar",
                "Germany",
                "Portugal",
            ],
            "neutral": [False, False, True, True, True, False],
        }
    )
