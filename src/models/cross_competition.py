"""Cross-league calibration of the goals-only rating models.

Domestic leagues are near-closed pools: a team plays its own division and
almost nothing else. ELO is zero-sum within a pool, so every league's ratings
drift around the same initial value no matter how strong the league really is —
measured on the real corpus, five of them sit at a mean of exactly 1500.0.
Dixon-Coles is worse: cross-league attack and defence strengths are outright
*unidentified*, because no domestic match connects a Portuguese team to an
English one, so no amount of domestic data can settle them.

Real cross-league matches are the only fix, and UEFA club competitions are
where they happen. This module feeds those results into the two models that
need nothing but goals, while keeping them out of the ensemble's feature
matrix, whose rows must carry the shot/foul/corner columns the European feed
does not have.
"""

from dataclasses import dataclass, field
from typing import ClassVar

import pandas as pd

from src.models.elo import FootballELO


@dataclass(frozen=True)
class CrossCompetitionCorpus:
    """The two corpora a goals-only model should see together.

    Bundled into one object so consumers take a single argument rather than a
    repeated pair (§4.3).

    Attributes:
        domestic: Fully featured domestic matches — the rows that train the
            ensemble.
        supplementary: Goals-only matches (European competitions) that exist
            purely to link the domestic pools.
    """

    #: Columns a goals-only model consumes.
    GOALS_COLUMNS: ClassVar[tuple[str, ...]] = (
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
    )

    domestic: pd.DataFrame
    supplementary: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def has_supplementary(self) -> bool:
        return not self.supplementary.empty

    def combined_goals(self) -> pd.DataFrame:
        """Both corpora as one chronological goals-only frame.

        This is what Dixon-Coles is fitted on: including the European rows is
        what makes one league's attack strength comparable with another's.
        """
        frames = [self._goals(self.domestic)]
        if self.has_supplementary:
            frames.append(self._goals(self.supplementary))
        combined = pd.concat(frames, ignore_index=True)
        if combined.empty:
            return combined
        return combined.sort_values("Date").reset_index(drop=True)

    @classmethod
    def _goals(cls, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame({column: [] for column in cls.GOALS_COLUMNS})
        available = [c for c in cls.GOALS_COLUMNS if c in frame.columns]
        return frame[available].copy()


@dataclass(frozen=True)
class CrossCompetitionFeatures:
    """Both corpora after one shared ELO walk, each side kept separate.

    ``domestic`` is what trains the ensemble. ``supplementary`` is what the
    blend-weight backtest scores: European rows carrying the ratings each team
    took *into* the tie, which is the only honest thing to predict it from.
    """

    domestic: pd.DataFrame
    supplementary: pd.DataFrame


class CrossCompetitionEloBuilder:
    """Computes ELO across both corpora, returning features for domestic rows.

    The ratings themselves are pooled — a European result updates the same
    ``FootballELO`` instance a domestic result does — but only domestic rows
    come back out, so the training matrix is unchanged in shape and content
    apart from better-calibrated ELO columns.

    ``FootballELO`` is injected rather than constructed here, so the fitted
    ratings stay reachable afterwards (inference needs them) and any other
    rating implementation can be substituted.
    """

    #: Helper columns, added to survive the chronological sort and removed
    #: before the frame is returned. Underscore-prefixed so they cannot
    #: collide with a football-data column.
    ROW_ID_COLUMN: ClassVar[str] = "_cc_row_id"
    SOURCE_COLUMN: ClassVar[str] = "_cc_source"

    DOMESTIC: ClassVar[str] = "domestic"
    SUPPLEMENTARY: ClassVar[str] = "supplementary"

    def __init__(self, elo: FootballELO) -> None:
        self.elo = elo

    def build(self, corpus: CrossCompetitionCorpus) -> pd.DataFrame:
        """Domestic rows with ELO features computed against both corpora."""
        return self.build_all(corpus).domestic

    def build_all(self, corpus: CrossCompetitionCorpus) -> CrossCompetitionFeatures:
        """Both sides of the walk, for callers that need the European rows.

        ``build`` exists for the training path and hands back domestic rows
        only, which is correct for a feature matrix and useless for measuring
        how well the models predict European ties. Recomputing those rows
        afterwards is not an option: ELO walked over the domestic output alone
        reverts to the uncalibrated ratings the corpus exists to correct.
        """
        if corpus.domestic.empty:
            return CrossCompetitionFeatures(
                domestic=corpus.domestic, supplementary=pd.DataFrame()
            )
        if not corpus.has_supplementary:
            # No European history: behave exactly as the plain ELO path.
            return CrossCompetitionFeatures(
                domestic=self.elo.compute_elo_features(corpus.domestic),
                supplementary=pd.DataFrame(),
            )

        combined = self._tagged_combination(corpus)
        featured = self.elo.compute_elo_features(combined)
        return CrossCompetitionFeatures(
            domestic=self._domestic_rows(featured),
            supplementary=self._supplementary_rows(featured),
        )

    def _tagged_combination(self, corpus: CrossCompetitionCorpus) -> pd.DataFrame:
        """Both corpora in one frame, each row traceable back to its source."""
        domestic = corpus.domestic.copy()
        domestic[self.SOURCE_COLUMN] = self.DOMESTIC
        domestic[self.ROW_ID_COLUMN] = range(len(domestic))

        supplementary = CrossCompetitionCorpus._goals(corpus.supplementary).copy()
        supplementary = supplementary.dropna(
            subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        )
        supplementary[self.SOURCE_COLUMN] = self.SUPPLEMENTARY
        supplementary[self.ROW_ID_COLUMN] = -1

        return pd.concat([domestic, supplementary], ignore_index=True)

    def _domestic_rows(self, featured: pd.DataFrame) -> pd.DataFrame:
        """Domestic rows only, chronological and with helper columns removed.

        The order matters beyond tidiness: downstream cross-validation splits
        on time, so this path and the no-European path must hand back the same
        chronological ordering. The row id breaks ties within a date, which
        ``sort_values`` alone does not do deterministically.
        """
        domestic = featured[featured[self.SOURCE_COLUMN] == self.DOMESTIC]
        return (
            domestic.sort_values(["Date", self.ROW_ID_COLUMN])
            .drop(columns=[self.SOURCE_COLUMN, self.ROW_ID_COLUMN])
            .reset_index(drop=True)
        )

    def _supplementary_rows(self, featured: pd.DataFrame) -> pd.DataFrame:
        """European rows only, chronological and with helper columns removed.

        These share a row id of -1, so ``Date`` alone orders them. Their
        columns are the goals-only set plus the ELO features, since that is
        all the European feed carries.
        """
        supplementary = featured[featured[self.SOURCE_COLUMN] == self.SUPPLEMENTARY]
        return (
            supplementary.sort_values("Date")
            .drop(columns=[self.SOURCE_COLUMN, self.ROW_ID_COLUMN])
            .dropna(axis=1, how="all")
            .reset_index(drop=True)
        )
