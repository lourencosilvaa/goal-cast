import pandas as pd


class DataCleaner:
    """Data cleaning and standardization for football match data."""

    NUMERIC_COLS = [
        "FTHG",
        "FTAG",
        "HTHG",
        "HTAG",
        "HS",
        "AS",
        "HST",
        "AST",
        "HF",
        "AF",
        "HC",
        "AC",
        "HY",
        "AY",
        "HR",
        "AR",
        "B365H",
        "B365D",
        "B365A",
    ]

    RESULT_MAP = {"H": 2, "D": 1, "A": 0}

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize raw football data."""
        df = df.copy()

        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        for col in self.NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Result"] = df["FTR"].map(self.RESULT_MAP)
        df = df.dropna(subset=["Result"])
        df["Result"] = df["Result"].astype(int)

        return df
