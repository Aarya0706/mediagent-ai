"""
ui/appointment_prep.py
-------------------------
Tab 7 - "Pre-Appointment Prep Generator": pulls a patient's recent
cases and lab reports into an LLM-generated prep sheet ahead of a
doctor visit. Extracted verbatim from app.py's `with tab7:` block.
"""

import streamlit as st

from services.llm_clients import get_lab_llm
from tools.appointment_prep_tools import (
    get_recent_cases,
    get_recent_lab_reports,
    get_lab_values_for_reports,
    has_any_history,
    get_all_patients_with_history,
    generate_appointment_prep,
)

_lab_llm = get_lab_llm()


def render_appointment_prep_tab():
    st.header("🗓️ Pre-Appointment Prep Generator")
    st.markdown(
        "Generate a one-page prep sheet before a doctor's visit: questions to ask, "
        "what's changed since last time, and which reports to bring - built from this "
        "patient's actual case and lab history, not generic advice."
    )
    st.caption(
        "Pulls from the last 5 symptom-checker cases and the last 3 lab reports on file "
        "for the selected patient."
    )

    st.divider()

    prep_patients = get_all_patients_with_history()
    _is_patient_session = st.session_state.get("auth_role") == "patient"

    if _is_patient_session:
        _own_name = st.session_state.get("auth_patient_name", "")
        # Locked to their own identity - only proceed if THEY specifically
        # have case/lab history, never show a picker over every patient.
        _own_has_history = any(p.strip().lower() == _own_name.strip().lower() for p in prep_patients)

        if not _own_has_history:
            st.info(
                "No case or lab history on file for your account yet. Complete a "
                "symptom check or upload a lab report first, then come back here."
            )
            prep_patient_name = None
        else:
            prep_patient_name = _own_name
            st.text_input("Patient", value=prep_patient_name, disabled=True, key="prep_locked_patient_display", help="Locked to your account's identity.")

    elif not prep_patients:
        st.info(
            "No patients with any case or lab history yet. Use the Patient Triage or "
            "Lab Reports tab first, then come back here."
        )
        prep_patient_name = None
    else:
        prep_patient_name = st.selectbox(
            "Patient",
            options=prep_patients,
            key="prep_patient_select",
        )

    if prep_patient_name:
        col_reason, col_specialty = st.columns(2)
        with col_reason:
            visit_reason = st.text_input(
                "Reason for this visit",
                placeholder="e.g. Follow-up on blood sugar and fatigue",
                key="prep_visit_reason",
            )
        with col_specialty:
            doctor_specialty = st.text_input(
                "Doctor's specialty (optional)",
                placeholder="e.g. Endocrinologist",
                key="prep_doctor_specialty",
            )

        if st.button("🗓️ Generate Prep Sheet", width="stretch"):
            if not has_any_history(prep_patient_name):
                st.warning("No case or lab history found for this patient.")
            else:
                with st.spinner("Reviewing case and lab history..."):
                    recent_cases = get_recent_cases(prep_patient_name, limit=5)
                    recent_lab_reports = get_recent_lab_reports(prep_patient_name, limit=3)
                    report_ids = [r["id"] for r in recent_lab_reports]
                    lab_values_by_report = get_lab_values_for_reports(report_ids)

                with st.spinner("Generating questions and summary..."):
                    try:
                        prep_result = generate_appointment_prep(
                            patient_name=prep_patient_name,
                            visit_reason=visit_reason.strip(),
                            doctor_specialty=doctor_specialty.strip(),
                            recent_cases=recent_cases,
                            recent_lab_reports=recent_lab_reports,
                            lab_values_by_report=lab_values_by_report,
                            llm=_lab_llm,
                        )
                        st.session_state["prep_last_result"] = {
                            "patient": prep_patient_name,
                            **prep_result,
                        }
                    except Exception as e:
                        st.error("Something went wrong while generating the prep sheet.")
                        st.exception(e)

        last_prep = st.session_state.get("prep_last_result")
        if last_prep and last_prep["patient"] == prep_patient_name:
            st.divider()

            st.markdown("### 📝 What's Changed")
            st.info(last_prep["change_summary"] or "No summary available.")

            st.markdown("### ❓ Questions to Ask Your Doctor")
            if last_prep["questions"]:
                for i, q in enumerate(last_prep["questions"], 1):
                    st.markdown(f"{i}. {q}")
            else:
                st.caption("No questions were generated.")

            st.markdown("### 📎 Reports to Bring")
            if last_prep["reports_to_bring"]:
                for r in last_prep["reports_to_bring"]:
                    st.markdown(f"- **{r['file_name']}** — {r['date']}")
            else:
                st.caption("No lab reports on file for this patient yet.")

            st.caption(
                "This prep sheet is generated to help you organise the conversation with "
                "your doctor. It is not medical advice and does not replace their assessment."
            )
