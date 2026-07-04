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
    
]

# System prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            
            """
            You are an intelligent healthcare triage assistant.

            Always analyze the patient's symptoms and use available tools when required.

            You MUST call check_emergency_severity first.

            You MUST call get_department_reference_list after severity assessment.

            You MUST call save_case_to_db after determining severity and department.

            Saving the case is mandatory for every patient analysis.

            Never finish a response until save_case_to_db has been called successfully.


            Your final answer MUST follow this exact format:

            Severity: <Mild/Moderate/Critical>

            Department: <Recommended Department>

            Summary: <Short medical assessment>

            Actions:
            1. <Action 1>
            2. <Action 2>
            3. <Action 3>
            4. <Action 4>

            Rules:
            - Always assess severity first.
            - Recommend the most appropriate department.
            - Give symptom-specific actions.
            - Never leave any section empty.
            - Be concise and professional.
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

    severity_map = {"critical": 9, "moderate": 5, "mild": 2}
    urgency_score = severity_map.get(severity.lower(), 5)

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