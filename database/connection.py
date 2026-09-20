"""
database/connection.py
------------------------
The one place DB_PATH and get_connection() are defined. Previously
this exact pair (DB_PATH + get_connection()) was copy-pasted in
app.py, tools/auth_tools.py, tools/health_profile_tools.py,
tools/lab_report_tools.py and tools/save_case.py - all pointing at the
same file, all written independently. Harmless while they agreed, but
a change to one (e.g. switching databases, adding connection options)
had to be remembered in five places.

Usage:
    from database.connection import get_connection, DB_PATH

    with get_connection() as conn:
        conn.execute("SELECT ...")
"""

import os
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "hospital.db")


def get_connection():
    """Row-factory-enabled connection to the app's single SQLite database.
    Ensures data/ exists first, so this is always safe to call first."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
