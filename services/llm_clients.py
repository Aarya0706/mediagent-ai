"""
services/llm_clients.py
--------------------------
The two cached Groq LLM client factories app.py used to define inline
(get_drug_llm for the Drug Interaction Checker, get_lab_llm for Lab
Report parsing + the AI Health Chat). Both are simple @st.cache_resource
factories with no domain logic - pulled out mainly so the two service
modules that use them (drug_service.py, and app.py's lab-report/chat
call sites) have one place to get an LLM client from, instead of each
needing its own copy of the same six lines.
"""

import os

import streamlit as st
from langchain_groq import ChatGroq


@st.cache_resource
def get_drug_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY"),
        model_kwargs={"reasoning_effort": "low"}
    )


@st.cache_resource
def get_lab_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY"),
        model_kwargs={"reasoning_effort": "low"}
    )
