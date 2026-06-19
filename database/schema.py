PATIENTS_TABLE = """
CREATE TABLE IF NOT EXISTS patients(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    gender TEXT,
    phone TEXT
);
"""

CASES_TABLE = """
CREATE TABLE IF NOT EXISTS cases(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    symptoms TEXT,
    severity TEXT,
    department TEXT,
    summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

APPOINTMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS appointments(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    department TEXT,
    slot_time TEXT,
    status TEXT
);
"""

AGENT_LOGS_TABLE = """
CREATE TABLE IF NOT EXISTS agent_logs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER,
    action TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""