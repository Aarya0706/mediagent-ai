"""
ui/health_chat.py
--------------------
Tab 8 - "AI Health Chat": RAG-grounded conversational Q&A over a
patient's own records (cases + lab reports), with a PubMed fallback
when nothing in their own history clears the confidence bar. See
tools/chat_rag_tools.py for the retrieval/grounding logic itself -
this module is just the chat UI. Extracted verbatim from app.py's
`with tab8:` block.
"""

import streamlit as st

from services.llm_clients import get_lab_llm
from tools.appointment_prep_tools import get_all_patients_with_history
from tools.chat_rag_tools import (
    get_patient_chunks,
    has_any_chunks,
    get_relevant_context,
    generate_chat_answer,
)

_lab_llm = get_lab_llm()


def render_health_chat_tab():
    st.header("💬 AI Health Chat")
    st.markdown(
        "Ask questions about a patient's own health records - grounded only in their "
        "actual lab reports and symptom-checker history, with sources cited."
    )
    st.caption(
        "This is a v1: retrieval uses TF-IDF word-overlap matching over this patient's own "
        "records (not a trained embedding model), with a PubMed literature fallback when your "
        "own records don't confidently cover the question. Chat history is kept for this "
        "session only - it isn't saved once you close the app."
    )

    st.divider()

    chat_patients = get_all_patients_with_history()
    _is_patient_session = st.session_state.get("auth_role") == "patient"

    if _is_patient_session:
        _own_name = st.session_state.get("auth_patient_name", "")
        _own_has_history = any(p.strip().lower() == _own_name.strip().lower() for p in chat_patients)

        col_patient, col_mode = st.columns([2, 1])
        with col_patient:
            if _own_has_history:
                chat_patient_name = _own_name
                st.text_input("Patient", value=chat_patient_name, disabled=True, key="chat_locked_patient_display", help="Locked to your account's identity.")
            else:
                chat_patient_name = None
                st.info(
                    "No case or lab history on file for your account yet. Complete a "
                    "symptom check or upload a lab report first, then come back here."
                )
        with col_mode:
            chat_mode_label = st.radio(
                "Mode",
                options=["Patient", "Advanced"],
                key="chat_mode_select",
                horizontal=True,
            )
        chat_mode = "advanced" if chat_mode_label == "Advanced" else "patient"

    elif not chat_patients:
        chat_patient_name = None
        st.info(
            "No patients with any case or lab history yet. Use the Patient Triage or "
            "Lab Reports tab first, then come back here."
        )
    else:
        col_patient, col_mode = st.columns([2, 1])
        with col_patient:
            chat_patient_name = st.selectbox(
                "Patient",
                options=chat_patients,
                key="chat_patient_select",
            )
        with col_mode:
            chat_mode_label = st.radio(
                "Mode",
                options=["Patient", "Advanced"],
                key="chat_mode_select",
                horizontal=True,
            )
        chat_mode = "advanced" if chat_mode_label == "Advanced" else "patient"

    if chat_patient_name:
        if not has_any_chunks(chat_patient_name):
            st.caption(
                "No lab reports or symptom-checker cases found for this patient yet - "
                "questions about their personal records won't have anything to draw on, "
                "but general health questions will still be answered."
            )

        history_key = f"chat_history__{chat_patient_name.strip().lower()}"
        if history_key not in st.session_state:
            st.session_state[history_key] = []

        for turn in st.session_state[history_key]:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
                if turn["role"] == "assistant" and turn.get("sources"):
                    with st.expander("Sources"):
                        for s in turn["sources"]:
                            st.caption(f"• {s}")

        question = st.chat_input(
            f"Ask something about {chat_patient_name}'s health records..."
        )

        if question:
            st.session_state[history_key].append(
                {"role": "user", "content": question}
            )
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Looking through the records..."):
                    try:
                        chunks = get_patient_chunks(chat_patient_name)
                        retrieved, used_pubmed = get_relevant_context(question, chunks)
                        result = generate_chat_answer(
                            question, retrieved, chat_mode, _lab_llm
                        )
                    except Exception as e:
                        result = {
                            "answer": "Something went wrong while answering that question.",
                            "sources": [],
                        }
                        used_pubmed = False
                        st.exception(e)

                    if used_pubmed:
                        st.caption(
                            "\U0001F52C This patient's own records didn't fully cover the "
                            "question - also checked PubMed medical literature."
                        )

                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("Sources"):
                        for s in result["sources"]:
                            st.caption(f"• {s}")

            st.session_state[history_key].append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })

        if st.button("🗑 Clear conversation", key="chat_clear_history"):
            st.session_state[history_key] = []
            st.rerun()
