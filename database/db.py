import sqlite3

from schema import (
    PATIENTS_TABLE,
    CASES_TABLE,
    APPOINTMENTS_TABLE,
    AGENT_LOGS_TABLE
)

DB_PATH = "../data/hospital.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(PATIENTS_TABLE)
    cursor.execute(CASES_TABLE)
    cursor.execute(APPOINTMENTS_TABLE)
    cursor.execute(AGENT_LOGS_TABLE)

    conn.commit()
    conn.close()

    print("Database Created Successfully")


if __name__ == "__main__":
    init_db()