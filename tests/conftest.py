import pytest
from pathlib import Path


@pytest.fixture
def config_path() -> Path:
    return Path(__file__).parent.parent / "config" / "config.yaml"


@pytest.fixture
def sample_match_data():
    import pandas as pd

    data = {
        "Date": pd.to_datetime([
            "2024-01-06", "2024-01-07", "2024-01-13", "2024-01-14",
            "2024-01-20", "2024-01-21", "2024-01-27", "2024-01-28",
            "2024-02-03", "2024-02-04",
        ]),
        "HomeTeam": [
            "Arsenal", "Liverpool", "Arsenal", "Man City",
            "Liverpool", "Arsenal", "Chelsea", "Liverpool",
            "Arsenal", "Man City",
        ],
        "AwayTeam": [
            "Chelsea", "Man City", "Liverpool", "Arsenal",
            "Chelsea", "Man City", "Arsenal", "Man City",
            "Chelsea", "Liverpool",
        ],
        "FTHG": [2, 1, 3, 0, 2, 1, 0, 2, 1, 3],
        "FTAG": [0, 1, 1, 2, 1, 1, 1, 0, 0, 1],
        "FTR": ["H", "D", "H", "A", "H", "D", "A", "H", "H", "H"],
        "HTHG": [1, 0, 1, 0, 1, 0, 0, 1, 0, 2],
        "HTAG": [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
        "HTR": ["H", "A", "H", "A", "H", "A", "D", "H", "D", "H"],
        "HS": [15, 12, 18, 8, 14, 10, 9, 16, 13, 20],
        "AS": [8, 14, 10, 15, 9, 13, 12, 8, 7, 11],
        "HST": [6, 4, 8, 3, 5, 4, 3, 7, 5, 9],
        "AST": [3, 5, 4, 6, 3, 5, 4, 3, 2, 4],
        "HF": [10, 12, 9, 14, 11, 13, 8, 10, 12, 9],
        "AF": [12, 10, 11, 9, 13, 10, 11, 12, 10, 11],
        "HC": [7, 5, 8, 3, 6, 4, 5, 7, 6, 9],
        "AC": [3, 6, 4, 7, 4, 6, 5, 3, 3, 5],
        "HY": [2, 1, 1, 3, 2, 1, 2, 1, 2, 1],
        "AY": [1, 2, 2, 1, 1, 2, 1, 2, 1, 2],
        "HR": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "AR": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "B365H": [1.50, 2.20, 1.80, 3.50, 1.90, 2.50, 3.00, 1.70, 1.60, 1.40],
        "B365D": [4.00, 3.40, 3.80, 3.60, 3.50, 3.30, 3.40, 3.80, 4.00, 4.50],
        "B365A": [6.50, 3.10, 4.20, 2.00, 3.80, 2.80, 2.30, 4.50, 5.50, 7.00],
    }

    return pd.DataFrame(data)
