import sys
import os

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