"""
database/db.py
----------------
Manual entrypoint for setting up (or checking) the database without
starting the Streamlit app - e.g. right after cloning, or in a
container build step.

The previous version of this file predated the real app (it imported
PATIENTS_TABLE/APPOINTMENTS_TABLE/AGENT_LOGS_TABLE that don't exist,
via a relative `from schema import ...` that only worked if you were
already `cd`'d into database/, and pointed at a relative "../data/
hospital.db" path that broke depending on your cwd). It was never
actually wired into app.py's startup, so it silently drifted out of
sync with the real schema. This version just calls the same
run_migrations() the app itself runs on every startup, so there's only
one implementation of "what does the database look like" - see
database/schema.py.

Usage:
    python -m database.db
    # or
    python database/db.py
"""

from database.migrations import run_migrations

if __name__ == "__main__":
    result = run_migrations()
    if result:
        print("Database created/updated. Migrations applied:")
        for table, columns in result.items():
            print(f"  {table}: added {columns}")
    else:
        print("Database already up to date - no migrations needed.")
