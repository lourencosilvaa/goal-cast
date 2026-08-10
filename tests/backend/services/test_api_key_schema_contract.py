"""The upsert and the documented schema have to agree.

They did not, and nothing caught it. ``ApiKeyService.set_user_key`` upserts
with ``on_conflict="user_id,service"``; the ``CREATE TABLE`` in the README
declared ``user_id UUID PRIMARY KEY``. Postgres answers that pair with
``42P10 — there is no unique or exclusion constraint matching the ON CONFLICT
specification``, so **every** save of a Gemini or NVIDIA key returned HTTP 500,
in production, silently, from the day the second service was added.

The schema was also wrong on its own terms: a user holds one key per service,
and a primary key on ``user_id`` alone can physically store only one of them.

No test could have caught this by mocking Supabase — a mock accepts any
``on_conflict`` string. What can be checked is that the two artefacts still
describe the same key, so this reads both and compares them.
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SERVICE = PROJECT_ROOT / "src/backend/services/api_key_service.py"
README = PROJECT_ROOT / "README.md"
TABLE = "user_api_keys"


def _conflict_columns() -> list[str]:
    """The columns the service's upsert relies on being unique together."""
    match = re.search(r'on_conflict\s*=\s*"([^"]+)"', SERVICE.read_text())
    assert match, "ApiKeyService no longer declares an on_conflict target"
    return [column.strip() for column in match.group(1).split(",")]


def _create_table() -> str:
    """The documented CREATE TABLE for user_api_keys."""
    body = README.read_text()
    match = re.search(
        rf"CREATE TABLE public\.{TABLE}\s*\((.*?)\);", body, re.DOTALL
    )
    assert match, f"README no longer documents the {TABLE} schema"
    return match.group(1)


class TestUpsertTarget:
    def test_the_service_upserts_on_user_and_service(self):
        assert _conflict_columns() == ["user_id", "service"]


class TestDocumentedSchema:
    def test_the_primary_key_covers_the_upsert_target(self):
        """Without this, Postgres rejects every save with 42P10."""
        create = _create_table()
        declared = re.search(r"PRIMARY KEY\s*\(([^)]+)\)", create)
        assert declared, (
            "user_api_keys must declare a composite PRIMARY KEY; a column-level "
            "`user_id UUID PRIMARY KEY` is what caused the 500s"
        )
        columns = [column.strip() for column in declared.group(1).split(",")]
        assert columns == _conflict_columns()

    def test_user_id_is_not_a_primary_key_on_its_own(self):
        """It cannot be: one user holds a Gemini key *and* an NVIDIA key."""
        assert not re.search(r"user_id\s+UUID\s+PRIMARY KEY", _create_table())

    def test_the_service_column_is_required(self):
        assert re.search(r"service\s+TEXT\s+NOT NULL", _create_table())

    def test_the_table_still_cascades_on_user_deletion(self):
        """A deleted account must not leave its encrypted keys behind."""
        assert "ON DELETE CASCADE" in _create_table()


class TestMigrationIsDocumented:
    """Existing projects were created with the broken schema, so the fix has
    to be written down — the corrected CREATE TABLE alone helps nobody who
    already ran the old one."""

    def test_the_readme_explains_how_to_repair_an_existing_table(self):
        body = README.read_text()
        assert "ALTER TABLE public.user_api_keys" in body
        assert "ADD PRIMARY KEY (user_id, service)" in body

    def test_the_readme_names_the_error_operators_will_actually_see(self):
        assert "42P10" in README.read_text()
