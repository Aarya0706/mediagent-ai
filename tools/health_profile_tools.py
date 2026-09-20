from datetime import datetime
from zoneinfo import ZoneInfo

from database.connection import get_connection, DB_PATH  # noqa: F401 - DB_PATH kept for callers that import it from here

IST = ZoneInfo("Asia/Kolkata")


# ============================================================
# SCHEMA
# ============================================================
#
# PRD 5.3 describes health_profile as 1:1 with `users` (allergies, chronic
# conditions, height/weight). This app identifies patients by name
# throughout (cases, lab_reports, users.patient_name), so health_profile
# is keyed on patient_name here too, same convention as every other
# table - see database/schema.py's note on why that's a known,
# deliberately-deferred limitation rather than a real user_id FK.
#
# The table itself is now defined once in database/schema.py and
# created/migrated by database/migrations.py.

def ensure_health_profile_table():
    """Kept as a function (rather than inlining the call below) so
    existing callers/imports of ensure_health_profile_table() elsewhere
    keep working unchanged."""
    from database.migrations import run_migrations
    run_migrations()


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