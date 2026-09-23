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
    "patients": (
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            display_name TEXT NOT NULL,
            normalized_name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        )
        """,
        {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "display_name": "TEXT NOT NULL",
            "normalized_name": "TEXT NOT NULL",
            "created_at": "TEXT NOT NULL",
        },
    ),

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
            patient_id INTEGER,
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
            "patient_id": "INTEGER",
            "created_at": "TEXT NOT NULL",
            "updated_at": "TEXT",
        },
    ),

    "cases": (
        """
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            patient_id INTEGER,
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
            "patient_id": "INTEGER",
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
            patient_id INTEGER,
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
            "patient_id": "INTEGER",
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
            patient_id INTEGER,
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
            "patient_id": "INTEGER",
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
            patient_id INTEGER,
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
            "patient_id": "INTEGER",
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
# IDENTITY MODEL: patient_id migration, in progress
# ============================================================
#
# Every table above now HAS a `patient_id` column pointing at the new
# `patients` table (see database/patients.py for get_or_create_patient()
# / find_patient_id()), and database/migrations.py backfills it
# automatically on every startup for any row where it's still NULL,
# matching by a normalized (trimmed, lowercased, whitespace-collapsed)
# version of patient_name. tools/auth_tools.py now also resolves and
# stores patient_id on every patient-role signup, and refuses to create
# a second patient-role login against a patient_id that's already
# claimed - the actual name-collision security hole this migration
# exists to close.
#
# What this does NOT do yet: none of the read/write query sites in
# tools/*.py, app.py, or ui/*.py have been switched to filter or join
# on patient_id instead of patient_name - they still read/write
# patient_name exactly as before, so nothing about existing behavior
# changes yet. `patient_id` is populated and available, but
# `patient_name` remains the column every query actually uses. That
# query-by-query rewrite (tools/save_case.py, tools/lab_report_tools.py,
# tools/health_profile_tools.py, tools/chat_rag_tools.py,
# tools/authorization.py's can_access_patient_record(), etc.) is a
# separate, larger pass - deliberately not folded in here, since it
# touches how every screen reads data and deserves to be done (and
# tested against a real backup of data/hospital.db) on its own, not
# silently bundled into a schema change.
