import sys
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from tools.emergency_checker import check_emergency_severity
from tools.department_router import get_department_reference_list
from tools.save_case import save_case_to_db

# Load API key from .env
from pathlib import Path

 

load_dotenv(dotenv_path=".env")

 

# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2
)

# Available tools
tools = [
    check_emergency_severity,
    get_department_reference_list,
    save_case_to_db
]

# System prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an intelligent healthcare triage assistant.

            Analyze the patient's symptoms, symptom duration, onset, pain level,
            known medical conditions, medications, and allergies.

            You MUST call check_emergency_severity first.
            Use the available tools when required.

            Your final answer MUST follow this exact format:

            Severity: <Mild/Moderate/Critical>

            Department: <Recommended Department>

            Summary: <2-4 sentence patient-friendly assessment explaining the symptoms,
            possible concerns, and why the recommended department is appropriate.
            Do not claim a confirmed diagnosis.>

            Actions:
            1. <A safe, immediate self-care or precautionary action specific to the patient's symptoms.>
            2. <A second symptom-specific action explaining what the patient should avoid, monitor, or do next.>
            3. <A clear recommendation about when and where to seek medical evaluation based on severity and duration.>
            4. <Specific red-flag symptoms to watch for that would require urgent medical attention.>

            Rules:
            - Always assess severity first.
            - Recommend the most appropriate department.
            - Give exactly 4 actions.
            - Every action MUST be specific to the patient's symptoms, body area, severity, duration, and onset.
            - Use the patient's known conditions, medications, and allergies when relevant.
            - Do not give generic advice such as "stay calm", "drink water", "get rest", or "monitor symptoms" unless it is specifically relevant to the patient's symptoms.
            - Do not repeatedly tell every patient to come to the hospital.
            - For Mild cases, prioritize safe self-care, symptom monitoring, and appropriate routine follow-up.
            - For Moderate cases, provide safe interim care and recommend timely medical evaluation.
            - For Critical cases, clearly instruct the patient to seek emergency medical care immediately.
            - Do not recommend prescription medications.
            - Do not provide medication dosages.
            - Do not recommend over-the-counter medication unless it is clearly appropriate to the symptoms and include a brief safety qualification.
            - Never claim or imply a confirmed diagnosis.
            - Do not exaggerate emergency risk.
            - Red-flag advice must be specific to the patient's symptoms.
            - Avoid repeating the same advice in multiple actions.
            - Be concise, patient-friendly, medically cautious, and professional.
            - Never leave any section empty.
            - Never include tool calls in the final response.
            - Never display <function> tags.
            - Use tools internally only.
            """
            
             
            
             
            
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ]
)

# Create tool-calling agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)
 

def run_triage_pipeline(symptoms: str, patient_context: str) -> dict:
    if not symptoms or not symptoms.strip():
        return {"valid": False, "invalid_reason": "No symptoms provided."}

    intake_text = f"Patient Context: {patient_context}\n\nSymptoms:\n{symptoms}"

    try:
        result = agent_executor.invoke({"input": intake_text})
        output_text = result.get("output", "")
    except Exception as e:
        return {"valid": False, "invalid_reason": f"Agent error: {str(e)}"}

    def extract_field(text, field):
        match = re.search(rf"{field}:\s*(.+)", text)
        return match.group(1).strip() if match else ""

    severity   = extract_field(output_text, "Severity") or "Moderate"
    department = extract_field(output_text, "Department") or "General Medicine"
    summary    = extract_field(output_text, "Summary") or output_text.strip()

    actions = re.findall(r"\d+\.\s*(.+)", output_text)
    if not actions:
        actions = ["Consult a physician for further evaluation."]

    # Pull the actual reported pain/discomfort level out of the intake text
    pain_match = re.search(r"Pain/Discomfort Level:\s*(\d+)/10", intake_text)
    pain_level = int(pain_match.group(1)) if pain_match else 5

    severity_base = {"critical": 8, "moderate": 5, "mild": 2}.get(severity.lower(), 5)

    # Blend severity label with actual reported pain level, so specifics of
    # the case (not just the 3-bucket label) affect the final score
    urgency_score = min(10, max(1, round((severity_base + pain_level) / 2)))

    warning = "Seek emergency care immediately if symptoms worsen." if severity.lower() == "critical" else ""

    return {
        "valid": True,
        "severity": severity,
        "department": department,
        "urgency_score": urgency_score,
        "triage_reasoning": output_text,
        "intake": intake_text,
        "summary": summary,
        "actions": actions,
        "warning": warning,
    }

if __name__ == "__main__":

    while True:
        user_input = input("\nPatient Symptoms (type exit to quit): ")

        if user_input.lower() == "exit":
            break

        result = agent_executor.invoke(
            {
                "input": user_input
            }
        )

        print("\nAI RESPONSE:")
        print(result["output"])