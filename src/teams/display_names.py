"""Human-facing names for canonically-keyed teams.

Every model, dataset and artefact in this project is keyed by football-data's
spelling — ``Sp Lisbon``, ``Man City``, ``Ein Frankfurt`` — and changing that
would mean remapping the whole corpus and retraining. But those keys are terse
to the point of ambiguity for a reader: ``Sp Lisbon`` is Sporting, not Benfica,
which is exactly the kind of confusion worth designing out of the interface.

So the key and the label are separated. openfootball's unambiguous full names
become *display* names, shown wherever a person reads one, while the key
underneath never moves.

A team with no mapping displays its canonical key, so the interface degrades to
today's behaviour rather than to a blank.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Mapping

import yaml


class DisplayNameRepository(ABC):
    """Read-only source of canonical-key → human-facing name."""

    @abstractmethod
    def all(self) -> Mapping[str, str]:
        """Every mapping, or an empty mapping when unavailable."""

    def display_for(self, canonical: str) -> str:
        """The name to show for a canonical key, falling back to the key."""
        return self.all().get(canonical, canonical)

    def apply(self, canonical_names: list[str]) -> list[str]:
        """Display names for a list of keys, order preserved."""
        return [self.display_for(name) for name in canonical_names]


class StaticDisplayNameRepository(DisplayNameRepository):
    """Display names committed to a YAML file and shipped with the image.

    Degrades to an empty mapping rather than raising, so a missing or
    malformed file leaves the UI showing canonical keys instead of failing.
    """

    ENCODING: ClassVar[str] = "utf-8"
    NAMES_KEY: ClassVar[str] = "display_names"

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._cache: Mapping[str, str] | None = None

    def all(self) -> Mapping[str, str]:
        if self._cache is None:
            self._cache = self._read()
        return self._cache

    def _read(self) -> Mapping[str, str]:
        try:
            raw = yaml.safe_load(self._path.read_text(encoding=self.ENCODING))
        except (OSError, yaml.YAMLError):
            return {}
        if not isinstance(raw, dict):
            return {}
        names = raw.get(self.NAMES_KEY)
        if not isinstance(names, dict):
            return {}
        return {
            str(canonical): str(display)
            for canonical, display in names.items()
            if str(canonical).strip() and str(display).strip()
        }


class NullDisplayNameRepository(DisplayNameRepository):
    """No mappings — every team displays its canonical key."""

    def all(self) -> Mapping[str, str]:
        return {}
