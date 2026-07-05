import sqlite3
import os
from datetime import datetime
from zoneinfo import ZoneInfo
IST = ZoneInfo("Asia/Kolkata")

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hospital.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def save_case_to_db(symptoms: str, severity: str, department: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symptoms TEXT,
            severity TEXT,
            department TEXT,
            summary TEXT,
            created_at TEXT
        )
    """)
    cur.execute("""
        INSERT INTO cases (symptoms, severity, department, summary, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        symptoms,
        severity,
        department,
        f"Patient reported: {symptoms}. Severity: {severity}.",
        datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()
    return "Case saved."