from datetime import datetime
from zoneinfo import ZoneInfo

from database.connection import get_connection, DB_PATH  # noqa: F401 - DB_PATH kept for callers that import it from here


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

IST = ZoneInfo("Asia/Kolkata")


# ============================================================
# INITIALIZE / MIGRATE DATABASE
# ============================================================

def initialize_database():
    """
    Creates every table (cases included) if it doesn't exist yet, and
    backfills any column an existing database is missing (e.g. status,
    doctor_notes on older cases rows) without touching existing data.

    Kept as a function (rather than inlining the call below) so
    existing callers/imports of initialize_database() elsewhere keep
    working unchanged. The actual table/column definitions now live in
    database/schema.py, applied by database/migrations.py.
    """
    from database.migrations import run_migrations
    run_migrations()


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
# GET CASES FOR ONE PATIENT (DB-level ownership scoping)
# ============================================================
#
# Security note: this exists because get_all_cases() must never be
# called on behalf of a patient-role session. A patient-role user's
# view of case history has to be enforced by a WHERE clause here, not
# by filtering the full result set in Python/UI after the fact or by
# hiding a tab - either of those still pulls every other patient's
# record into memory and one missed branch away from being shown.
# Callers should combine this with tools.authorization.require_patient_access().

def get_cases_for_patient(patient_name):
    """
    Returns only the cases belonging to patient_name, newest first.

    Use this (never get_all_cases()) for any view a patient-role
    session can reach.
    """

    patient_name = (patient_name or "").strip()
    if not patient_name:
        return []

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

            WHERE LOWER(patient_name) = LOWER(?)

            ORDER BY datetime(created_at) DESC, id DESC
            """,
            (patient_name,),
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