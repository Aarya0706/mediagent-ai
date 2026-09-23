"""
database/migrations.py
------------------------
One idempotent entrypoint that creates every table (if missing) and
adds any column an existing database doesn't have yet (if missing).

Previously this logic - CREATE TABLE IF NOT EXISTS, then PRAGMA
table_info + ALTER TABLE ADD COLUMN for anything new - was correct but
duplicated across tools/auth_tools.py (ensure_users_table),
tools/save_case.py (initialize_database), tools/lab_report_tools.py
(initialize_lab_tables), tools/health_profile_tools.py
(ensure_health_profile_table), and app.py (ensure_patient_name_column).
Each one only knew about its own table, so a column added to `cases`
in one place couldn't be seen or reasoned about from another.

run_migrations() now does all of it, once, from database/schema.py's
single table/column definitions. It's safe to call on every app
startup (and does - see database/connection.py callers): CREATE TABLE
IF NOT EXISTS and "add column only if missing" are both no-ops on a
database that's already up to date.

Usage:
    from database.migrations import run_migrations
    run_migrations()
"""

from database.connection import get_connection
from database.schema import TABLES

# Every table (other than `patients` itself) that carries a
# patient_name + patient_id pair needing backfill. `users` is included
# because patient-role logins are also part of the identity model - see
# _backfill_patient_ids() below and the note at the bottom of schema.py.
_PATIENT_SCOPED_TABLES = ("users", "cases", "health_profile", "lab_reports", "lab_values")


def run_migrations():
    """Creates every table in TABLES if missing, adds any column
    present in TABLES but missing from an existing table (using that
    column's real type/default, not a guess), and backfills patient_id
    on every patient-scoped row that doesn't have one yet. Returns a
    dict of {table_name: [columns_added]} for anything that changed, so
    callers/tests can see what happened - usually empty."""
    added = {}

    with get_connection() as conn:
        cursor = conn.cursor()

        for table_name, (create_sql, column_defs) in TABLES.items():
            cursor.execute(create_sql)

            cursor.execute(f"PRAGMA table_info({table_name})")
            existing_columns = {row["name"] for row in cursor.fetchall()}

            for column, ddl_fragment in column_defs.items():
                if column in existing_columns:
                    continue

                # SQLite can't ALTER TABLE ADD COLUMN with a PRIMARY KEY
                # or UNIQUE constraint - every column that's actually
                # missing on a legacy DB is a plain nullable/defaulted
                # column added after the table already existed (see
                # schema.py's comments), so this is safe for every real
                # case; a genuinely new PK/UNIQUE column would need a
                # real migration (new table + copy), not this path.
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column} {ddl_fragment}")
                added.setdefault(table_name, []).append(column)

        conn.commit()

        backfilled = _backfill_patient_ids(conn)
        if backfilled:
            added.setdefault("patients", []).append(f"backfilled {backfilled} row(s)")

    return added


def _backfill_patient_ids(conn):
    """For every row in _PATIENT_SCOPED_TABLES with a NULL patient_id
    and a non-blank patient_name, resolves (or creates) a `patients`
    row via database.patients.get_or_create_patient() and sets
    patient_id. Idempotent: rows that already have a patient_id are
    skipped, so this is a no-op on every run after the first. Returns
    the total number of rows updated, for callers/tests.

    Import is local (not top-of-file) to avoid a circular import -
    database.patients imports get_connection from this package too."""
    from database.patients import get_or_create_patient

    cursor = conn.cursor()
    total_updated = 0

    for table_name in _PATIENT_SCOPED_TABLES:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = {row["name"] for row in cursor.fetchall()}
        if "patient_id" not in columns or "patient_name" not in columns:
            continue

        id_column = "id" if "id" in columns else "patient_name"

        cursor.execute(
            f"SELECT {id_column} AS row_key, patient_name FROM {table_name} "
            f"WHERE patient_id IS NULL AND patient_name IS NOT NULL AND TRIM(patient_name) != ''"
        )
        rows = cursor.fetchall()

        for row in rows:
            patient_id = get_or_create_patient(row["patient_name"], conn=conn)
            if patient_id is None:
                continue
            cursor.execute(
                f"UPDATE {table_name} SET patient_id = ? WHERE {id_column} = ?",
                (patient_id, row["row_key"]),
            )
            total_updated += 1

    if total_updated:
        conn.commit()

    return total_updated


if __name__ == "__main__":
    result = run_migrations()
    if result:
        print("Migrations applied:")
        for table, columns in result.items():
            print(f"  {table}: added {columns}")
    else:
        print("Database already up to date - no migrations needed.")
