"""
database/schema.py
--------------------
Single source of truth for MediAgent AI's SQLite schema.

Before this file, each table's CREATE TABLE statement (and its own
column-presence-check migration) lived inline in whichever tools/*.py
module happened to need it first (tools/auth_tools.py, save_case.py,
lab_report_tools.py, health_profile_tools.py) plus one more ad-hoc
ALTER in app.py itself. Five different files each hand-rolled their
own DB_PATH + get_connection() too. That worked, but it meant the
actual shape of the database was ambiguous - fully "grep the whole
repo" to answer the question, and this module was that grep with a
completely different (unused, out of date) schema in it.

This file now describes every table exactly as it exists in
production today - it does not change any table's shape or
relationships. The one real limitation this does NOT fix (see the note
at the bottom) is left as-is deliberately, because fixing it is a data
migration, not a schema cleanup - see the note for why that's a
separate, carefully-planned piece of work rather than something to
fold in silently here.

Each entry below is (create_table_sql, {column_name: column_ddl_fragment}).
database/migrations.py uses the per-column DDL fragment to add any
column that's missing from an existing database with the same type/
default it would have gotten in a fresh CREATE TABLE - see that module.
"""

TABLES = {
    "users": (
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            display_name TEXT,
            salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'staff',
            patient_name TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT
        )
        """,
        {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "username": "TEXT NOT NULL UNIQUE COLLATE NOCASE",
            "display_name": "TEXT",
            "salt": "TEXT NOT NULL",
            "password_hash": "TEXT NOT NULL",
            "role": "TEXT NOT NULL DEFAULT 'staff'",
            "patient_name": "TEXT",
            "created_at": "TEXT NOT NULL",
            "updated_at": "TEXT",
        },
    ),

    "cases": (
        """
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            symptoms TEXT NOT NULL,
            severity TEXT NOT NULL,
            department TEXT NOT NULL,
            summary TEXT DEFAULT '',
            recommendation TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'Pending',
            doctor_notes TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT
        )
        """,
        {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "patient_name": "TEXT NOT NULL DEFAULT 'Unknown'",
            "symptoms": "TEXT NOT NULL",
            "severity": "TEXT NOT NULL",
            "department": "TEXT NOT NULL",
            "summary": "TEXT DEFAULT ''",
            "recommendation": "TEXT DEFAULT ''",
            "status": "TEXT NOT NULL DEFAULT 'Pending'",
            "doctor_notes": "TEXT DEFAULT ''",
            "created_at": "TEXT NOT NULL",
            "updated_at": "TEXT",
        },
    ),

    "health_profile": (
        """
        CREATE TABLE IF NOT EXISTS health_profile (
            patient_name TEXT PRIMARY KEY COLLATE NOCASE,
            age INTEGER,
            gender TEXT,
            blood_group TEXT,
            height_cm REAL,
            weight_kg REAL,
            chronic_conditions TEXT,
            allergies TEXT,
            current_medications TEXT,
            updated_at TEXT
        )
        """,
        {
            "patient_name": "TEXT PRIMARY KEY COLLATE NOCASE",
            "age": "INTEGER",
            "gender": "TEXT",
            "blood_group": "TEXT",
            "height_cm": "REAL",
            "weight_kg": "REAL",
            "chronic_conditions": "TEXT",
            "allergies": "TEXT",
            "current_medications": "TEXT",
            "updated_at": "TEXT",
        },
    ),

    "lab_reports": (
        """
        CREATE TABLE IF NOT EXISTS lab_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            file_name TEXT NOT NULL,
            raw_text TEXT DEFAULT '',
            ai_summary TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT
        )
        """,
        {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "patient_name": "TEXT NOT NULL",
            "file_name": "TEXT NOT NULL",
            "raw_text": "TEXT DEFAULT ''",
            "ai_summary": "TEXT DEFAULT ''",
            "created_at": "TEXT NOT NULL",
            "updated_at": "TEXT",
        },
    ),

    "lab_values": (
        """
        CREATE TABLE IF NOT EXISTS lab_values (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id INTEGER NOT NULL,
            patient_name TEXT NOT NULL,
            parameter TEXT NOT NULL,
            value REAL,
            unit TEXT DEFAULT '',
            ref_low REAL,
            ref_high REAL,
            flag TEXT DEFAULT 'unknown',
            created_at TEXT NOT NULL,
            FOREIGN KEY (report_id) REFERENCES lab_reports (id)
        )
        """,
        {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "report_id": "INTEGER NOT NULL",
            "patient_name": "TEXT NOT NULL",
            "parameter": "TEXT NOT NULL",
            "value": "REAL",
            "unit": "TEXT DEFAULT ''",
            "ref_low": "REAL",
            "ref_high": "REAL",
            "flag": "TEXT DEFAULT 'unknown'",
            "created_at": "TEXT NOT NULL",
        },
    ),
}

# agent_runs (agents/observability.py) is deliberately NOT listed here -
# it's pure telemetry with its own lifecycle (best-effort logging that
# must never block a triage run), not application data, so it keeps its
# own table-creation call rather than going through the startup
# migration path every other table uses.


# ============================================================
# KNOWN LIMITATION: patient_name as the join key, not a real FK
# ============================================================
#
# Every table above keys off `patient_name` (a free-text, case-
# insensitive string) rather than a `patient_id` foreign key into a
# `patients` table. `users.patient_name` is the sole source of truth
# for which records a patient-role login can see (tools/authorization.py)
# - so this isn't just an inconsistency, it's the actual identity model
# the whole app is built on.
#
# This is a real normalization gap (two patients who share a name would
# collide; renaming a patient means updating every table). It is NOT
# fixed here on purpose: doing it properly means (1) adding a `patients`
# table, (2) backfilling a `patient_id` on every existing row in cases /
# health_profile / lab_reports / lab_values / users against whatever
# real data already lives in data/hospital.db on the deployed instance,
# and (3) rewriting every query site (tools/*.py, app.py, tools/
# chat_rag_tools.py, tools/authorization.py) to join on the new id.
# That's a live-data migration with real corruption risk if the name-
# matching backfill logic gets an edge case wrong (e.g. two differently-
# cased spellings of the same patient), not a schema-only cleanup - it
# deserves its own pass with a backup of the real database in hand,
# not one folded silently into this one.
