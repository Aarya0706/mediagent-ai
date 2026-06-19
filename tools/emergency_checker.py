from langchain.tools import tool


@tool
def check_emergency_severity(symptoms: str) -> str:
    """
    MUST be called before routing a patient.
    Checks if symptoms indicate a medical emergency.
    """

    symptoms = symptoms.lower()

    critical = [
        "chest pain",
        "difficulty breathing",
        "heart attack",
        "stroke",
        "unconscious",
        "severe bleeding"
    ]

    high = [
        "high fever",
        "vomiting blood",
        "fainting"
    ]

    for item in critical:
        if item in symptoms:
            return "Critical"

    for item in high:
        if item in symptoms:
            return "High"

    return "Low"