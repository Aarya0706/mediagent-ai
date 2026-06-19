from langchain.tools import tool


@tool
def get_department_reference_list(symptoms: str) -> str:
    """
    Determines the correct department.
    """

    symptoms = symptoms.lower()
    
    if "emergency" in symptoms or (
        "chest pain" in symptoms and "difficulty breathing" in symptoms
    ):
        return "Emergency"

    if "chest pain" in symptoms or "heart" in symptoms:
        return "Cardiology"

    elif "difficulty breathing" in symptoms or "shortness of breath" in symptoms or "cough" in symptoms:
        return "Pulmonology"

    elif "headache" in symptoms or "migraine" in symptoms or "dizziness" in symptoms:
        return "Neurology"

    elif "skin" in symptoms or "rash" in symptoms or "itching" in symptoms:
        return "Dermatology"

    elif "eye" in symptoms or "vision" in symptoms:
        return "Ophthalmology"

    elif "bone" in symptoms or "fracture" in symptoms or "joint pain" in symptoms:
        return "Orthopedics"

    elif "ear" in symptoms or "nose" in symptoms or "throat" in symptoms:
        return "ENT"

    elif "child" in symptoms or "baby" in symptoms:
        return "Pediatrics"

    elif "stomach" in symptoms or "abdomen" in symptoms or "vomiting" in symptoms:
        return "Gastroenterology"

    elif "pregnancy" in symptoms or "obstetrics" in symptoms or "gynecology" in symptoms:
        return "Obstetrics & Gynecology"
    elif "fever" in symptoms:
        return "General Medicine"
    elif "diabetes" in symptoms or "blood sugar" in symptoms:
        return "Endocrinology"
    elif "anxiety" in symptoms or "depression" in symptoms or "stress" in symptoms:
        return "Psychiatry"
    return "General Medicine"