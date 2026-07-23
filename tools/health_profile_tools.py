import os
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DB_PATH = os.path.join(
    ROOT,
    "data",
    "hospital.db"
)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# SCHEMA
# ============================================================
#
# PRD 5.3 describes health_profile as 1:1 with `users` (allergies, chronic
# conditions, height/weight). This app has no auth/users table - patients
# are identified by name throughout (cases, lab_reports). So health_profile
# is keyed on patient_name here instead, same identity convention as every
# other table in this app. If real auth is added later, this is the table
# that would need a user_id foreign key swapped in.

def ensure_health_profile_table():
    with get_connection() as conn:
        conn.execute(
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
            """
        )
        conn.commit()


ensure_health_profile_table()


# ============================================================
# READ / WRITE
# ============================================================

def get_profile(patient_name):
    """Returns the stored health profile for this patient as a dict, or
    None if they don't have one saved yet."""
    if not patient_name or not patient_name.strip():
        return None

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM health_profile WHERE LOWER(patient_name) = LOWER(?)",
            (patient_name.strip(),),
        )
        row = cursor.fetchone()

    return dict(row) if row else None


def upsert_profile(
    patient_name,
    age=None,
    gender=None,
    blood_group=None,
    height_cm=None,
    weight_kg=None,
    chronic_conditions=None,
    allergies=None,
    current_medications=None,
):
    """Creates or updates this patient's stored health profile.

    Any field passed as None leaves the existing stored value untouched
    (fetched from the current row first) rather than blanking it out. This
    matters because two different screens write to this table: the full
    Health Profile tab (sets everything at once) and the Symptom Checker
    (only ever touches age/gender/conditions/allergies/medications after a
    triage session) - a triage-session sync shouldn't erase height/weight
    that was set earlier from the profile tab.
    """
    if not patient_name or not patient_name.strip():
        raise ValueError("patient_name is required")

    name = patient_name.strip()
    existing = get_profile(name) or {}

    def pick(new_value, key):
        return new_value if new_value is not None else existing.get(key)

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO health_profile (
                patient_name, age, gender, blood_group, height_cm, weight_kg,
                chronic_conditions, allergies, current_medications, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(patient_name) DO UPDATE SET
                age = excluded.age,
                gender = excluded.gender,
                blood_group = excluded.blood_group,
                height_cm = excluded.height_cm,
                weight_kg = excluded.weight_kg,
                chronic_conditions = excluded.chronic_conditions,
                allergies = excluded.allergies,
                current_medications = excluded.current_medications,
                updated_at = excluded.updated_at
            """,
            (
                name,
                pick(age, "age"),
                pick(gender, "gender"),
                pick(blood_group, "blood_group"),
                pick(height_cm, "height_cm"),
                pick(weight_kg, "weight_kg"),
                pick(chronic_conditions, "chronic_conditions"),
                pick(allergies, "allergies"),
                pick(current_medications, "current_medications"),
                datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()


def get_all_profiled_patients():
    """Distinct patient names that already have a saved health profile."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT patient_name FROM health_profile ORDER BY patient_name COLLATE NOCASE"
        )
        return [row["patient_name"] for row in cursor.fetchall()]