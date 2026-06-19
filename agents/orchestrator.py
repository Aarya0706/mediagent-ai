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
    save_case_to_db
]

# System prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an intelligent healthcare triage assistant.

            Rules:
            - Always assess emergency severity first.
            - Use available tools whenever useful.
            - If symptoms are severe, recommend immediate medical attention.
            - If not severe, determine the correct department.
            - Explain your reasoning clearly.
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
    verbose=False
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