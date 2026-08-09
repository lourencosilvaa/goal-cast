"""One way to load the European corpus, shared by training and inference.

Training and inference must build ELO from *identical* inputs. They do not
share a code path — training runs offline, inference runs in three separate
places — so the only thing keeping them honest is that all four call this.

Why identical inputs matter more than it might seem: a fixture's ``elo_home``
feature is read from that team's **last historical row**, i.e. the rating it
carried into its most recent match. Reproducing that number requires replaying
the same chronological ELO walk over the same matches. A model trained on
calibrated ratings and served uncalibrated ones is worse than one that was
never calibrated, because the features no longer mean what the model learned
them to mean.

Reads only the on-disk cache. Fetching here would put a live website in the
inference path, where an outage produces silently wrong predictions rather
than a loud failure.
"""

from typing import Any

import pandas as pd

from src.corpus.canonical_names import build_translator
from src.corpus.supplementary import StaticFileCorpusSource


class MissingEuropeanCorpusError(RuntimeError):
    """Calibration was required but its inputs were absent.

    Raised only when ``european.required`` is set — training sets it, inference
    does not. The distinction matters: an inference path that cannot find the
    corpus should still serve predictions, whereas a retrain that cannot find
    it would overwrite a calibrated model with an uncalibrated one.
    """


#: Pointer used in every diagnostic, so the fix is one command away.
_BUILD_HINT = "Run: uv run python scripts/build_european_corpus.py"
_ALIAS_HINT = (
    "Run: uv run python scripts/export_team_aliases.py (approved alias mappings "
    "must be committed to the seed — Supabase is not reachable from CI)"
)


def load_european_corpus(config: Any, verbose: bool = True) -> pd.DataFrame:
    """Cached European results, translated to canonical team keys.

    Returns an empty frame when the track is disabled, the cache is missing,
    or the config predates the European section — every one of which must
    leave the caller working exactly as it did before, just uncalibrated.

    Unless ``european.required`` is set, in which case each of those becomes a
    :class:`MissingEuropeanCorpusError`. Silence is the failure mode this guard
    exists to remove: it is how a scheduled retrain shipped an uncalibrated
    model and still reported success.

    The translation is not optional: ELO is keyed by the team-name string, so
    an untranslated "Sport Lisboa e Benfica" builds a rating separate from
    Benfica's and links nothing.
    """
    european = getattr(config, "european", None)
    if european is None or not european.enabled:
        return pd.DataFrame()

    required = getattr(european, "required", False)

    corpus = StaticFileCorpusSource(european.cache_path).load()
    if corpus.empty:
        if required:
            raise MissingEuropeanCorpusError(
                f"No European corpus at {european.cache_path}, but "
                f"european.required is set. {_BUILD_HINT}"
            )
        if verbose:
            print(
                "No European corpus found — ELO and Dixon-Coles ratings will "
                "NOT be comparable across leagues. "
                "See docs/european-competitions.md."
            )
        return corpus

    try:
        result = build_translator(config).translate(corpus)
    except Exception as exc:
        # A translator that cannot be built (no Supabase, no registry) must not
        # take inference down; it just means nothing links.
        if required:
            raise MissingEuropeanCorpusError(
                f"European corpus could not be translated ({exc}), but "
                f"european.required is set. {_ALIAS_HINT}"
            ) from exc
        if verbose:
            print(f"European corpus could not be translated ({exc}) — skipping")
        return pd.DataFrame()

    if required and not result.report.translated:
        # The corpus is present but nothing in it resolves to a canonical key,
        # so every European row builds a rating of its own and the domestic
        # pools stay exactly as disconnected as they were.
        raise MissingEuropeanCorpusError(
            f"European corpus has {len(corpus)} matches but not one team name "
            f"maps to a canonical key, so no leagues are linked. {_ALIAS_HINT}"
        )

    if verbose:
        report = result.report
        print(
            f"European corpus: {len(corpus)} matches, "
            f"{report.translated} names mapped to canonical keys"
        )
        if report.linkable:
            print(
                f"  WARNING: {len(report.linkable)} names are still unapproved, "
                f"costing {report.unlinked_appearances} match appearances of "
                f"cross-league linkage. Run scripts/review_team_names.py."
            )
    return result.frame
