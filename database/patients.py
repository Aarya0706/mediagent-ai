"""
database/patients.py
----------------------
Resolves a stable `patient_id` for a display name, backed by the new
`patients` table (database/schema.py). This is the first piece of the
patient_id migration described in the KNOWN LIMITATION note at the
bottom of schema.py: a `patients` table now exists, and this module is
the one place that knows how to normalize a name and look up or create
its row - so every future call site that needs a patient_id (the
startup backfill in migrations.py, signup in tools/auth_tools.py, and
eventually the query-site rewrite) goes through the same normalization
logic instead of five different ad-hoc `.strip().lower()` calls.

Normalization: trim, collapse internal whitespace, lowercase. This is
deliberately simple (no fuzzy matching, no nickname handling) so it's
predictable and auditable - two names that normalize to the same string
are treated as the same patient; anything else is a different patient,
even if a human would recognize them as the same person. That's a
reasonable default for a portfolio project's demo data, but it's worth
being explicit that it's a real limitation: a typo'd name creates a new
patient rather than matching an existing one.
"""

import re
from datetime import datetime
from zoneinfo import ZoneInfo

from database.connection import get_connection

IST = ZoneInfo("Asia/Kolkata")


def normalize_name(name):
    """Trim, collapse internal whitespace, lowercase. Returns "" for
    None/blank input rather than raising, so callers can check
    truthiness instead of catching exceptions."""
    if not name:
        return ""
    return re.sub(r"\s+", " ", name.strip()).lower()


def find_patient_id(name, conn=None):
    """Returns the existing patient_id for `name`, or None if no
    patient with that normalized name exists yet. Never creates a row -
    use get_or_create_patient() when a new patient should be
    provisioned."""
    normalized = normalize_name(name)
    if not normalized:
        return None

    owns_connection = conn is None
    conn = conn or get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM patients WHERE normalized_name = ?",
            (normalized,),
        )
        row = cursor.fetchone()
        return row["id"] if row else None
    finally:
        if owns_connection:
            conn.close()


def get_or_create_patient(name, conn=None):
    """Returns the patient_id for `name`, creating a new `patients` row
    if one doesn't exist yet for this normalized name. Returns None for
    blank/None input rather than creating a row for an empty identity.

    Safe to call concurrently: normalized_name is UNIQUE, so a race
    that tries to insert the same normalized name twice raises
    sqlite3.IntegrityError on the second insert, which is caught here
    and re-resolved with a lookup instead of propagating."""
    normalized = normalize_name(name)
    if not normalized:
        return None

    owns_connection = conn is None
    conn = conn or get_connection()
    try:
        existing = find_patient_id(name, conn=conn)
        if existing is not None:
            return existing

        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO patients (display_name, normalized_name, created_at) VALUES (?, ?, ?)",
                (name.strip(), normalized, datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")),
            )
            if owns_connection:
                conn.commit()
            return cursor.lastrowid
        except Exception:
            # Most likely a UNIQUE constraint race (another call created
            # the same normalized_name between the lookup above and this
            # insert) - re-resolve rather than fail the caller.
            return find_patient_id(name, conn=conn)
    finally:
        if owns_connection:
            conn.close()
