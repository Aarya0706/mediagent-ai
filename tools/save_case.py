from langchain.tools import tool
import sqlite3
from datetime import datetime
def generate_summary(symptoms, severity):

    symptoms = symptoms.lower()

    if "chest pain" in symptoms:
        return "Possible cardiac condition. Immediate evaluation recommended."

    elif "difficulty breathing" in symptoms:
        return "Possible respiratory distress. Monitor oxygen levels."

    elif "headache" in symptoms:
        return "Neurological symptoms detected. Further assessment advised."

    elif "fever" in symptoms:
        return "Possible infection. Monitor temperature and hydration."

    elif "rash" in symptoms:
        return "Skin-related symptoms detected."

    return f"Patient reported: {symptoms}. Severity assessed as {severity}."


@tool
def save_case_to_db(symptoms: str,
                    severity: str,
                    department: str) -> str:
     
    
    """
    Save patient case details into the hospital database.

Use this after determining:
- symptoms
- severity
- department

Always save the case before ending the conversation.
    """
    print("SAVE TOOL CALLED")

    conn = sqlite3.connect("data/hospital.db")
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
    summary = generate_summary(symptoms, severity)

    cur.execute("""
        INSERT INTO cases
        (symptoms, severity, department, summary, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        symptoms,
        severity,
        department,
        summary,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    print("CASE SAVED:", symptoms, severity, department)
    conn.close()

    return f"Patient reported: {symptoms}"

    return "Case successfully saved."