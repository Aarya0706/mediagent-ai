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


def run_migrations():
    """Creates every table in TABLES if missing, and adds any column
    present in TABLES but missing from an existing table (using that
    column's real type/default, not a guess). Returns a dict of
    {table_name: [columns_added]} for anything that changed, so
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

    return added


if __name__ == "__main__":
    result = run_migrations()
    if result:
        print("Migrations applied:")
        for table, columns in result.items():
            print(f"  {table}: added {columns}")
    else:
        print("Database already up to date - no migrations needed.")
