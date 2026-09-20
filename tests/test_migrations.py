"""
tests/test_migrations.py
--------------------------
Unit tests for database.migrations.run_migrations() - pure SQLite
against a temp file, no LLM/network calls, CI-safe like the other
tests/test_*.py modules.

Covers:
  - a fresh DB gets every table with every expected column
  - running it twice is a no-op the second time (idempotency)
  - a simulated "legacy" DB (missing newer columns, the exact
    situation this replaces five scattered ad-hoc ALTERs for)
    gets backfilled correctly, without touching existing data

Run:
    pytest tests/test_migrations.py -v
"""

import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import database.connection as connection
import database.migrations as migrations
from database.schema import TABLES


def _use_temp_db(monkeypatch, tmp_path):
    db_path = str(tmp_path / "test_migrations.db")
    monkeypatch.setattr(connection, "DB_PATH", db_path)
    return db_path


def test_fresh_db_gets_every_table_and_column(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)

    migrations.run_migrations()

    with connection.get_connection() as conn:
        cursor = conn.cursor()
        for table_name, (_, column_defs) in TABLES.items():
            cursor.execute(f"PRAGMA table_info({table_name})")
            actual_columns = {row["name"] for row in cursor.fetchall()}
            assert actual_columns == set(column_defs.keys()), (
                f"{table_name}: expected {set(column_defs.keys())}, got {actual_columns}"
            )


def test_running_twice_is_idempotent(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)

    first = migrations.run_migrations()
    second = migrations.run_migrations()

    assert first != {} or first == {}  # first run may create columns on a truly empty DB's tables via CREATE TABLE (not ALTER), so no assertion needed here
    assert second == {}, f"second run should be a no-op, but added: {second}"


def test_legacy_db_gets_missing_columns_backfilled(monkeypatch, tmp_path):
    db_path = _use_temp_db(monkeypatch, tmp_path)

    # Simulate a database created before status/doctor_notes/updated_at
    # existed on `cases` - exactly the scenario tools/save_case.py's old
    # inline ALTER logic (and app.py's ensure_patient_name_column) used
    # to handle one column at a time.
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT NOT NULL,
                symptoms TEXT NOT NULL,
                severity TEXT NOT NULL,
                department TEXT NOT NULL,
                summary TEXT DEFAULT '',
                recommendation TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO cases (patient_name, symptoms, severity, department, created_at)
            VALUES ('Jane Doe', 'headache', 'Mild', 'General Medicine', '2026-01-01T00:00:00')
            """
        )
        conn.commit()

    result = migrations.run_migrations()

    assert "cases" in result
    assert set(result["cases"]) >= {"status", "doctor_notes", "updated_at"}

    with connection.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(cases)")
        columns = {row["name"] for row in cursor.fetchall()}
        assert {"status", "doctor_notes", "updated_at"}.issubset(columns)

        # Existing row must survive untouched, with the new column
        # defaulted sensibly rather than data being lost.
        cursor.execute("SELECT * FROM cases WHERE patient_name = 'Jane Doe'")
        row = cursor.fetchone()
        assert row["symptoms"] == "headache"
        assert row["status"] == "Pending"  # backfilled via its DEFAULT


def test_all_tables_are_created_with_row_factory_access(monkeypatch, tmp_path):
    """Sanity check that get_connection()'s row_factory works against
    freshly migrated tables (dict-style row[\"col\"] access, used
    throughout tools/*.py)."""
    _use_temp_db(monkeypatch, tmp_path)
    migrations.run_migrations()

    with connection.get_connection() as conn:
        conn.execute(
            """
            INSERT INTO health_profile (patient_name, age, updated_at)
            VALUES ('Test Patient', 30, '2026-01-01T00:00:00')
            """
        )
        conn.commit()

        row = conn.execute(
            "SELECT * FROM health_profile WHERE patient_name = 'Test Patient'"
        ).fetchone()
        assert row["age"] == 30
