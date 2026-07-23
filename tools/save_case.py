import os
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

IST = ZoneInfo("Asia/Kolkata")

ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DB_PATH = os.path.join(
    ROOT,
    "data",
    "hospital.db"
)

os.makedirs(
    os.path.dirname(DB_PATH),
    exist_ok=True
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    """
    Creates a SQLite connection.

    row_factory allows rows to be accessed like dictionaries:
    row["patient_name"]
    """

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    return conn


# ============================================================
# INITIALIZE / MIGRATE DATABASE
# ============================================================

def initialize_database():
    """
    Creates the cases table if it does not exist.

    Also safely adds the status column to existing databases
    without deleting old cases.
    """

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cases
            (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                patient_name TEXT NOT NULL,

                symptoms TEXT NOT NULL,

                severity TEXT NOT NULL,

                department TEXT NOT NULL,

                summary TEXT DEFAULT '',

                recommendation TEXT DEFAULT '',

                created_at TEXT NOT NULL,

                status TEXT NOT NULL DEFAULT 'Pending'
            )
            """
        )


        # ----------------------------------------------------
        # MIGRATE EXISTING DATABASE
        # ----------------------------------------------------

        cursor.execute("PRAGMA table_info(cases)")

        columns = {
            row["name"]
            for row in cursor.fetchall()
        }


        if "status" not in columns:

            cursor.execute(
                """
                ALTER TABLE cases
                ADD COLUMN status TEXT NOT NULL DEFAULT 'Pending'
                """
            )


        if "doctor_notes" not in columns:

            cursor.execute(
                """
                ALTER TABLE cases
                ADD COLUMN doctor_notes TEXT DEFAULT ''
                """
            )


        conn.commit()


# Run initialization automatically whenever imported

initialize_database()


# ============================================================
# SAVE NEW CASE
# ============================================================

def save_case_to_db(
    patient_name,
    symptoms,
    severity,
    department,
    summary="",
    recommendation=""
):
    """
    Saves a new patient case.

    New cases automatically receive Pending status.
    """

    patient_name = str(patient_name).strip() or "Unknown"

    symptoms = str(symptoms).strip()

    severity = str(severity).strip()

    department = str(department).strip()

    summary = str(summary or "").strip()

    recommendation = str(recommendation or "").strip()


    created_at = datetime.now(IST).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO cases
            (
                patient_name,
                symptoms,
                severity,
                department,
                summary,
                recommendation,
                created_at,
                status
            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_name,
                symptoms,
                severity,
                department,
                summary,
                recommendation,
                created_at,
                "Pending"
            )
        )


        conn.commit()


        case_id = cursor.lastrowid


    return case_id


# ============================================================
# GET ALL CASES
# ============================================================

def get_all_cases():
    """
    Returns all patient cases.

    Newest cases appear first.
    """

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT

                id,

                patient_name,

                symptoms,

                severity,

                department,

                summary,

                recommendation,

                created_at,

                status

            FROM cases

            ORDER BY datetime(created_at) DESC, id DESC
            """
        )


        cases = cursor.fetchall()


    return [
        dict(case)
        for case in cases
    ]


# ============================================================
# UPDATE CASE STATUS
# ============================================================

def update_case_status(case_id, status):
    """
    Updates the workflow status of a patient case.

    Allowed values:
    Pending
    In Progress
    Resolved
    """

    allowed_statuses = {
        "Pending",
        "In Progress",
        "Resolved"
    }


    if status not in allowed_statuses:

        raise ValueError(
            f"Invalid case status: {status}"
        )


    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE cases

            SET status = ?

            WHERE id = ?
            """,
            (
                status,
                case_id
            )
        )


        conn.commit()


        updated = cursor.rowcount


    return updated > 0


# ============================================================
# UPDATE DOCTOR NOTES
# ============================================================

def update_case_notes(case_id, notes):
    """
    Saves a doctor's free-text consultation notes against a case.

    Overwrites any previous notes for this case (single note field,
    not a log) - callers that want history should read the existing
    value first and append if that's the desired behaviour.
    """

    notes = str(notes or "").strip()

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE cases

            SET doctor_notes = ?

            WHERE id = ?
            """,
            (
                notes,
                case_id
            )
        )


        conn.commit()


        updated = cursor.rowcount


    return updated > 0


# ============================================================
# DELETE ONE CASE
# ============================================================

def delete_case(case_id):
    """
    Permanently deletes a single case by id.

    Returns True if a row was deleted, False if no case with
    that id existed (e.g. already deleted by another session).
    """

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM cases

            WHERE id = ?
            """,
            (case_id,)
        )


        conn.commit()


        deleted = cursor.rowcount


    return deleted > 0


# ============================================================
# GET PENDING CRITICAL CASE COUNT
# ============================================================

def get_pending_critical_count():
    """
    Counts Critical cases that have NOT been resolved.

    The counter automatically decreases when a Critical
    case is marked Resolved.
    """

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(*)

            FROM cases

            WHERE LOWER(TRIM(severity)) = 'critical'

            AND status != 'Resolved'
            """
        )


        count = cursor.fetchone()[0]


    return count


# ============================================================
# GET AVAILABLE DEPARTMENTS
# ============================================================

def get_departments():
    """
    Returns unique departments currently present in the DB.

    Used by the Doctor Portal department filter.
    """

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT DISTINCT department

            FROM cases

            WHERE department IS NOT NULL

            AND TRIM(department) != ''

            ORDER BY department ASC
            """
        )


        departments = [
            row[0]
            for row in cursor.fetchall()
        ]


    return departments


# ============================================================
# GET CASES BY DEPARTMENT
# ============================================================

def get_cases_by_department(department):
    """
    Returns cases belonging to one department.
    """

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT

                id,

                patient_name,

                symptoms,

                severity,

                department,

                summary,

                recommendation,

                created_at,

                status

            FROM cases

            WHERE department = ?

            ORDER BY datetime(created_at) DESC, id DESC
            """,
            (department,)
        )


        cases = cursor.fetchall()


    return [
        dict(case)
        for case in cases
    ]


# ============================================================
# DELETE ALL CASES
# ============================================================

def clear_all_cases():
    """
    Deletes all patient cases.

    Keep this only if your Admin Dashboard currently uses
    a Clear Cases button.
    """

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM cases
            """
        )


        conn.commit()


    return True