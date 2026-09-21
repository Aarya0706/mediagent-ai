"""
ui/health_profile.py
-----------------------
Tab 9 - "Health Profile": view/edit a patient's persistent profile
(demographics, chronic conditions, medications, allergies) reused
across the triage form, lab reports, and appointment prep. Extracted
verbatim from app.py's `with tab9:` block.
"""

import streamlit as st

from tools.appointment_prep_tools import get_all_patients_with_history
from tools.health_profile_tools import get_profile, upsert_profile, get_all_profiled_patients


def render_health_profile_tab():
    st.header("🗂️ Health Profile")
    st.markdown(
        "Persistent patient details - blood group, height/weight, chronic "
        "conditions, allergies, and current medications - saved once and "
        "reused across the app instead of being re-typed every visit."
    )
    st.caption(
        "Loaded automatically wherever it's relevant (e.g. via \"Load saved "
        "health profile\" on the Patient Triage tab). Editing and saving here "
        "always overwrites the full profile; the Patient Triage tab only ever "
        "syncs age, gender, conditions, medications, and allergies."
    )

    st.divider()

    # Same pending-key handoff pattern as the Lab Reports tab: apply any
    # queued patient selection / new-patient-field clear BEFORE the widgets
    # that own those keys are drawn below.
    if "profile_pending_patient_select" in st.session_state:
        st.session_state["profile_patient_select"] = st.session_state.pop(
            "profile_pending_patient_select"
        )
    if st.session_state.pop("profile_pending_new_patient_clear", False):
        st.session_state["profile_new_patient_name"] = ""

    _is_patient_session = st.session_state.get("auth_role") == "patient"

    if _is_patient_session:
        # Patient sessions manage only their own profile - locked to the
        # identity set at signup, never a picker over every patient.
        profile_patient_name = st.session_state.get("auth_patient_name", "")
        st.text_input(
            "Patient",
            value=profile_patient_name,
            disabled=True,
            key="profile_locked_patient_display",
            help="Locked to your account's identity.",
        )
    else:
        known_patients = sorted(
            set(get_all_patients_with_history()) | set(get_all_profiled_patients()),
            key=str.lower,
        )
        profile_patient_options = ["+ Add new patient..."] + known_patients

        selected_profile_option = st.selectbox(
            "Patient",
            options=profile_patient_options,
            key="profile_patient_select",
        )

        if selected_profile_option == "+ Add new patient...":
            profile_patient_name = st.text_input(
                "New patient name",
                key="profile_new_patient_name",
                placeholder="e.g. Rohan Sharma",
            )
        else:
            profile_patient_name = selected_profile_option

    if not profile_patient_name.strip():
        st.caption("Enter or select a patient name above to view or edit their health profile.")
    else:
        existing_profile = get_profile(profile_patient_name.strip()) or {}

        if existing_profile:
            st.success(f"Showing saved profile - last updated {existing_profile.get('updated_at', 'unknown')}.")
        else:
            st.info("No saved profile yet for this patient - fill in what you know and save.")

        BLOOD_GROUPS = ["Unknown", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
        CHRONIC_CONDITION_OPTIONS = [
            "Diabetes", "Hypertension", "Heart Disease", "Asthma",
            "Thyroid Disorder", "Kidney Disease", "Epilepsy", "Cancer",
            "HIV/AIDS", "Arthritis", "Depression / Anxiety", "None",
        ]

        # Widget keys are namespaced by patient name below. Streamlit only
        # applies a widget's value=/index=/default= argument the FIRST time
        # that key is ever created - on every later rerun it keeps whatever
        # is already in session_state for that key, ignoring the argument.
        # With a fixed key like "profile_age", switching the selected
        # patient would keep showing the PREVIOUS patient's numbers instead
        # of the newly-selected one's. Namespacing the key by patient name
        # gives each patient their own fresh widget instance instead.
        _pf_ns = profile_patient_name.strip().lower()

        pf_col1, pf_col2, pf_col3 = st.columns(3)

        with pf_col1:
            profile_age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                step=1,
                value=int(existing_profile.get("age") or 0),
                key=f"profile_age__{_pf_ns}",
            )
            profile_gender = st.selectbox(
                "Gender",
                ["Male", "Female", "Other"],
                index=["Male", "Female", "Other"].index(existing_profile.get("gender"))
                if existing_profile.get("gender") in ["Male", "Female", "Other"]
                else 0,
                key=f"profile_gender__{_pf_ns}",
            )

        with pf_col2:
            profile_blood_group = st.selectbox(
                "Blood Group",
                BLOOD_GROUPS,
                index=BLOOD_GROUPS.index(existing_profile.get("blood_group"))
                if existing_profile.get("blood_group") in BLOOD_GROUPS
                else 0,
                key=f"profile_blood_group__{_pf_ns}",
            )
            profile_height = st.number_input(
                "Height (cm)",
                min_value=0.0,
                max_value=250.0,
                step=0.5,
                value=float(existing_profile.get("height_cm") or 0.0),
                key=f"profile_height__{_pf_ns}",
            )

        with pf_col3:
            profile_weight = st.number_input(
                "Weight (kg)",
                min_value=0.0,
                max_value=300.0,
                step=0.5,
                value=float(existing_profile.get("weight_kg") or 0.0),
                key=f"profile_weight__{_pf_ns}",
            )

        existing_conditions_list = [
            c.strip()
            for c in (existing_profile.get("chronic_conditions") or "").split(",")
            if c.strip()
        ]
        profile_conditions = st.multiselect(
            "Chronic Conditions",
            CHRONIC_CONDITION_OPTIONS,
            default=[c for c in existing_conditions_list if c in CHRONIC_CONDITION_OPTIONS] or ["None"],
            key=f"profile_conditions__{_pf_ns}",
        )

        pf_col4, pf_col5 = st.columns(2)
        with pf_col4:
            profile_medications = st.text_input(
                "Current Medications",
                value=existing_profile.get("current_medications") or "",
                placeholder="e.g. Metformin, Aspirin, Lisinopril",
                key=f"profile_medications__{_pf_ns}",
            )
        with pf_col5:
            profile_allergies = st.text_input(
                "Known Allergies",
                value=existing_profile.get("allergies") or "",
                placeholder="e.g. Penicillin, Sulfa drugs, Latex",
                key=f"profile_allergies__{_pf_ns}",
            )

        st.divider()

        if st.button("💾 Save Health Profile", width="stretch"):
            conditions_str = ", ".join(c for c in profile_conditions if c != "None") or "None"

            upsert_profile(
                patient_name=profile_patient_name.strip(),
                age=int(profile_age) if profile_age else None,
                gender=profile_gender,
                blood_group=profile_blood_group,
                height_cm=float(profile_height) if profile_height else None,
                weight_kg=float(profile_weight) if profile_weight else None,
                chronic_conditions=conditions_str,
                allergies=profile_allergies.strip() or None,
                current_medications=profile_medications.strip() or None,
            )

            if not _is_patient_session and selected_profile_option == "+ Add new patient...":
                # Same auto-select handoff as the Lab Reports tab - avoids
                # leaving "+ Add new patient..." selected after their first save.
                st.session_state["profile_pending_patient_select"] = profile_patient_name.strip()
                st.session_state["profile_pending_new_patient_clear"] = True

            st.success(f"Health profile saved for {profile_patient_name.strip()}.")
            st.rerun()
