import pandas as pd

from src.models.data_cleaner import DataCleaner


class TestDataCleaner:

    def test_clean_converts_dates(self, sample_match_data):
        cleaner = DataCleaner()
        result = cleaner.clean(sample_match_data)
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])

    def test_clean_adds_result_column(self, sample_match_data):
        cleaner = DataCleaner()
        result = cleaner.clean(sample_match_data)
        assert "Result" in result.columns

    def test_clean_result_encoding(self, sample_match_data):
        cleaner = DataCleaner()
        result = cleaner.clean(sample_match_data)
        assert set(result["Result"].unique()).issubset({0, 1, 2})

    def test_clean_home_win_is_2(self, sample_match_data):
        cleaner = DataCleaner()
        result = cleaner.clean(sample_match_data)
        home_wins = result[result["FTR"] == "H"]["Result"]
        assert all(home_wins == 2)

    def test_clean_draw_is_1(self, sample_match_data):
        cleaner = DataCleaner()
        result = cleaner.clean(sample_match_data)
        draws = result[result["FTR"] == "D"]["Result"]
        assert all(draws == 1)

    def test_clean_away_win_is_0(self, sample_match_data):
        cleaner = DataCleaner()
        result = cleaner.clean(sample_match_data)
        away_wins = result[result["FTR"] == "A"]["Result"]
        assert all(away_wins == 0)

    def test_clean_sorts_by_date(self, sample_match_data):
        cleaner = DataCleaner()
        result = cleaner.clean(sample_match_data)
        dates = result["Date"].values
        assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))

    def test_clean_numeric_columns(self, sample_match_data):
        cleaner = DataCleaner()
        result = cleaner.clean(sample_match_data)
        for col in ["FTHG", "FTAG", "HS", "AS"]:
            assert pd.api.types.is_numeric_dtype(result[col])

    def test_clean_drops_rows_without_date(self):
        cleaner = DataCleaner()
        data = pd.DataFrame({
            "Date": ["2024-01-01", "invalid_date"],
            "HomeTeam": ["A", "B"],
            "AwayTeam": ["C", "D"],
            "FTHG": [1, 2],
            "FTAG": [0, 1],
            "FTR": ["H", "A"],
        })
        result = cleaner.clean(data)
        assert len(result) == 1

    def test_clean_preserves_all_valid_rows(self, sample_match_data):
        cleaner = DataCleaner()
        result = cleaner.clean(sample_match_data)
        assert len(result) == len(sample_match_data)
