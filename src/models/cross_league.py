"""Fixtures whose two teams come from different leagues.

Two small shared pieces: deciding that a pairing *is* cross-league, and
counting how much history each club has. Both are needed by the scheduled job
and by the HuggingFace Space, and both were on their way to being written
twice — the mistake :mod:`src.models.fixture_features` exists to undo.

The routing question is deliberately not "is this a UEFA competition?". What
makes the ensemble invalid is the *pairing*: it is trained only on domestic
rows, so it has never seen a fixture whose sides come from different pools and
has no feature that would tell it one had arrived. A Liga Portugal club against
a Premier League club is that fixture whether or not a European badge is
attached, which is why this asks about leagues rather than competitions.

Kept to pandas and nothing else: it is mirrored into ``hf_space/`` (see
``tests/test_hf_space_contract.py``).
"""

import pandas as pd

#: Columns a corpus must have before appearances can be counted.
_TEAM_COLUMNS = ("HomeTeam", "AwayTeam")


def is_cross_league(home_league: str, away_league: str) -> bool:
    """Whether the two sides come from different leagues.

    An unstated away league means "the same one" — that is what a domestic
    caller sending a single league code has always meant, and it must keep
    meaning it. An unstated *home* league is not a licence to guess either:
    with nothing to compare against, the answer is no, and the domestic path
    (which can at least refuse on an unknown team) handles it.
    """
    home = home_league.strip()
    away = away_league.strip()
    if not home or not away:
        return False
    return home != away


def match_counts(featured_data: pd.DataFrame) -> dict[str, int]:
    """How many matches each team appears in, home or away.

    The gate behind refusing a prediction: a Dixon-Coles parameter fitted on
    three matches is noise wearing the shape of knowledge.

    A frame missing the team columns counts nothing rather than raising. The
    counts are evidence *for* a refusal, so their absence can only make the
    gate more permissive — and taking a prediction down because a snapshot was
    malformed would be the worse failure.
    """
    if featured_data.empty:
        return {}
    if any(column not in featured_data.columns for column in _TEAM_COLUMNS):
        return {}
    appearances = pd.concat(
        [featured_data[column] for column in _TEAM_COLUMNS], ignore_index=True
    )
    counts: dict[str, int] = appearances.value_counts().to_dict()
    return counts
