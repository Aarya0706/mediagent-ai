"""
ui/patient_triage.py
-----------------------
Tab 1 - "Patient Triage": the symptom-intake form, the multi-agent
triage pipeline call, and the resulting AI assessment + PDF report
download. Extracted from app.py's `with tab1:` block.

UX fix: the pipeline call used to sit behind a progress bar that
jumped to 10% before the (single, blocking, 10-20 second) pipeline
call and only moved again after it returned - so for the whole real
wait, the UI showed "Agent 1/3" frozen at 10%, which misrepresented
what was actually happening. Replaced with an honest spinner that
stays active for the actual duration of the call and sets accurate
time expectations instead of faking incremental progress.
"""

import os

import streamlit as st
from groq import Groq

from agents.pipeline import run_triage_pipeline
from services.report_service import generate_pdf_report
from tools.health_profile_tools import get_profile, upsert_profile
from tools.save_case import save_case_to_db
from ui.components import show_error_details
from zoneinfo import ZoneInfo
from datetime import datetime

IST = ZoneInfo("Asia/Kolkata")


def now_ist():
    return datetime.now(IST)


groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def render_patient_triage_tab():
    st.header("🩺 Patient Symptom Analysis")

    # Apply any pending health-profile load from a previous rerun, BEFORE
    # the widgets below are instantiated - Streamlit forbids writing to a
    # widget's session_state key after that widget has already been drawn
    # in the same run (same pending-key pattern used in the Lab Reports tab).
    for _pending_key, _real_key in [
        ("triage_pending_age", "triage_age"),
        ("triage_pending_gender", "triage_gender"),
        ("triage_pending_conditions", "triage_known_conditions"),
        ("triage_pending_meds", "triage_current_meds"),
        ("triage_pending_allergies", "triage_allergies"),
    ]:
        if _pending_key in st.session_state:
            st.session_state[_real_key] = st.session_state.pop(_pending_key)

    st.markdown("""
    <div style="background:linear-gradient(135deg,#c0392b,#e74c3c);border-radius:16px;padding:20px 24px;margin-bottom:16px;display:flex;align-items:center;justify-content:space-between;box-shadow:0 4px 20px rgba(231,76,60,0.3);">
        <div>
            <div style="color:white;font-size:20px;font-weight:800;margin-bottom:4px;">🚨 Life-Threatening Emergency?</div>
            <div style="color:#FECACA;font-size:14px;">If you or someone is in immediate danger - do not use this form</div>
        </div>
        <div style="text-align:right;">
            <div style="color:white;font-size:28px;font-weight:900;">📞 112</div>
            <div style="color:#FECACA;font-size:12px;">India Emergency Helpline</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    sos_col1, sos_col2, sos_col3 = st.columns(3)

    with sos_col1:
        st.markdown("""
        <div style="background:#FFF5F5;border:1px solid #FECACA;border-radius:12px;padding:12px;text-align:center;margin-bottom:16px;">
            <div style="font-size:22px;">🚑</div>
            <div style="font-weight:700;color:#c0392b;font-size:14px;">Ambulance</div>
            <div style="font-size:18px;font-weight:800;color:#c0392b;">102</div>
        </div>
        """, unsafe_allow_html=True)

    with sos_col2:
        st.markdown("""
        <div style="background:#FFF5F5;border:1px solid #FECACA;border-radius:12px;padding:12px;text-align:center;margin-bottom:16px;">
            <div style="font-size:22px;">🏥</div>
            <div style="font-weight:700;color:#c0392b;font-size:14px;">Emergency</div>
            <div style="font-size:18px;font-weight:800;color:#c0392b;">108</div>
        </div>
        """, unsafe_allow_html=True)

    with sos_col3:
        st.markdown("""
        <div style="background:#FFF5F5;border:1px solid #FECACA;border-radius:12px;padding:12px;text-align:center;margin-bottom:16px;">
            <div style="font-size:22px;">👮</div>
            <div style="font-weight:700;color:#c0392b;font-size:14px;">Police</div>
            <div style="font-size:18px;font-weight:800;color:#c0392b;">100</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 👤 Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        _is_patient_session = st.session_state.get("auth_role") == "patient"

        if _is_patient_session:
            # Patient sessions never get to type a name here - it's locked
            # to the identity set at signup, so this form can only ever
            # create a case under their own name, never someone else's.
            st.session_state["triage_patient_name"] = st.session_state.get("auth_patient_name", "")

        patient_name = st.text_input(
            "Patient Name",
            key="triage_patient_name",
            disabled=_is_patient_session,
            help="Locked to your account's identity." if _is_patient_session else None,
        )

        if not _is_patient_session and patient_name.strip() and get_profile(patient_name.strip()):
            if st.button(
                "📂 Load saved health profile",
                key="triage_load_profile",
                help="Fills in age, gender, conditions, medications, and allergies from this patient's saved profile.",
            ):
                _profile = get_profile(patient_name.strip())
                st.session_state["triage_pending_age"] = int(_profile.get("age") or 0)
                st.session_state["triage_pending_gender"] = _profile.get("gender") or "Male"

                _saved_conditions = [
                    c.strip()
                    for c in (_profile.get("chronic_conditions") or "").split(",")
                    if c.strip()
                ]
                st.session_state["triage_pending_conditions"] = _saved_conditions or ["None"]
                st.session_state["triage_pending_meds"] = _profile.get("current_medications") or ""
                st.session_state["triage_pending_allergies"] = _profile.get("allergies") or ""
                st.rerun()

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            step=1,
            key="triage_age",
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Other"],
            key="triage_gender",
        )
        phone = st.text_input("Phone Number")

    st.divider()
    st.markdown("#### 🫀 Symptom Details")

    col3, col4 = st.columns(2)

    with col3:
        body_part = st.selectbox(
            "Primary Body Part / Area Affected",
            [
                "Select...",
                "Head / Brain",
                "Eyes",
                "Ears / Nose / Throat",
                "Chest / Heart",
                "Lungs / Breathing",
                "Abdomen / Stomach",
                "Back / Spine",
                "Arms / Hands",
                "Legs / Feet",
                "Skin",
                "Reproductive / Urinary",
                "Mental / Psychological",
                "Whole Body / General",
            ],
        )

        duration = st.selectbox(
            "How long have you had these symptoms?",
            [
                "Just started (< 1 hour)",
                "A few hours (1-6 hours)",
                "Today (6-24 hours)",
                "A few days (2-3 days)",
                "About a week",
                "More than a week",
                "Chronic / ongoing",
            ],
        )

    with col4:
        severity_slider = st.slider(
            "Pain / Discomfort Level",
            min_value=1,
            max_value=10,
            value=5,
            help="1 = barely noticeable, 10 = worst imaginable",
        )

        if severity_slider <= 3:
            st.success(
                f"Level {severity_slider}/10 - Mild discomfort"
            )
        elif severity_slider <= 6:
            st.warning(
                f"Level {severity_slider}/10 - Moderate discomfort"
            )
        else:
            st.error(
                f"Level {severity_slider}/10 - Severe discomfort"
            )

        onset_type = st.radio(
            "How did symptoms start?",
            ["Sudden / Abrupt", "Gradual"],
            horizontal=True,
        )

    # IMPORTANT: outside col4
    # Voice Input + Symptoms Text Area

    st.markdown("#### 🎙️ Voice Input (Optional)")

    audio_value = st.audio_input(
        "Record your symptoms",
        key="symptoms_audio",
    )

    if audio_value is not None:
        if st.button("✨ Transcribe Voice", key="transcribe_symptoms"):
            try:
                with st.spinner("Transcribing your voice..."):
                    transcription = groq_client.audio.transcriptions.create(
                        file=(
                            "symptoms.wav",
                            audio_value.getvalue(),
                        ),
                        model="whisper-large-v3-turbo",
                        response_format="text",
                    )

                    st.session_state.pending_symptoms_text = str(
                        transcription
                    ).strip()

                st.success("Voice transcription completed.")
                st.rerun()

            except Exception as e:
                st.error(f"Voice transcription failed: {e}")


    if "pending_symptoms_text" in st.session_state:
        st.session_state.symptoms_text = (
            st.session_state.pending_symptoms_text
        )
        del st.session_state.pending_symptoms_text


    if "symptoms_text" not in st.session_state:
        st.session_state.symptoms_text = ""


    symptoms_desc = st.text_area(
        "Describe your symptoms in detail",
        placeholder=(
            "e.g. sharp chest pain radiating to left arm, "
            "shortness of breath, dizziness"
        ),
        height=100,
        key="symptoms_text",
    )
    st.divider()
    st.markdown("#### 🏥 Medical Context")

    col5, col6 = st.columns(2)

    with col5:
        if "triage_known_conditions" not in st.session_state:
            st.session_state["triage_known_conditions"] = ["None"]

        known_conditions = st.multiselect(
            "Known Medical Conditions (if any)",
            [
                "Diabetes",
                "Hypertension",
                "Heart Disease",
                "Asthma",
                "Thyroid Disorder",
                "Kidney Disease",
                "Epilepsy",
                "Cancer",
                "HIV/AIDS",
                "Arthritis",
                "Depression / Anxiety",
                "None",
            ],
            key="triage_known_conditions",
        )

    with col6:
        current_medications = st.text_input(
            "Current Medications (optional)",
            placeholder="e.g. Metformin, Aspirin, Lisinopril",
            key="triage_current_meds",
        )

        allergies = st.text_input(
            "Known Allergies (optional)",
            placeholder="e.g. Penicillin, Sulfa drugs, Latex",
            key="triage_allergies",
        )

    st.divider()

    if st.button(
        "🔍 Analyze Symptoms",
        width="stretch",
    ):
        if body_part == "Select...":
            st.warning(
                "Please select the body part affected."
            )

        elif not symptoms_desc.strip():
            st.warning(
                "Please describe your symptoms in the text box."
            )

        else:
            conditions_str = ", ".join(
                c
                for c in known_conditions
                if c != "None"
            ) or "None reported"

            symptoms = f"""Body Part Affected: {body_part}
Symptom Description: {symptoms_desc.strip()}
Duration: {duration}
Onset: {onset_type}
Pain/Discomfort Level: {severity_slider}/10
Known Conditions: {conditions_str}
Current Medications: {current_medications.strip() or "None reported"}
Allergies: {allergies.strip() or "None reported"}""".strip()

            patient_context = (
                f"Age: {age}, Gender: {gender}"
            )

            with st.spinner(
                "🤖 Running the multi-agent triage pipeline "
                "(Intake → Triage → Recommendation)... "
                "this typically takes 10-20 seconds."
            ):
                try:
                    result = run_triage_pipeline(
                        symptoms,
                        patient_context,
                    )

                except Exception as e:
                    st.error(
                        "The triage pipeline failed to run."
                    )
                    show_error_details(e)

                    result = None

            if result is not None:
                if not result.get("valid"):
                    st.error(
                        "Could not process input: "
                        f"{result.get('invalid_reason', 'Unknown reason')}"
                    )

                    st.info(
                        "Please describe your symptoms "
                        "more specifically."
                    )

                else:
                    severity = result["severity"]
                    department = result["department"]
                    urgency = result["urgency_score"]

                    st.markdown("---")
                    st.markdown("### 📊 Assessment Results")

                    res_col1, res_col2, res_col3, res_col4 = st.columns(4)

                    with res_col1:
                        if severity == "Critical":
                            st.error(f"🔴 **{severity}**")

                        elif severity == "Moderate":
                            st.warning(f"🟡 **{severity}**")

                        else:
                            st.success(f"🟢 **{severity}**")

                        st.caption("Severity Level")

                    with res_col2:
                        st.info(f"🏥 **{department}**")
                        st.caption("Recommended Department")

                    with res_col3:
                        st.metric(
                            "Urgency Score",
                            f"{urgency} / 10",
                        )
                    with res_col4:
                        confidence_score = result.get("confidence_score", 50)
                        st.metric("AI Confidence", f"{confidence_score}%")

                    st.progress(urgency / 10)

                    if severity == "Critical":
                        st.error(
                            "🚨 IMMEDIATE MEDICAL ATTENTION REQUIRED "
                            "- Go to Emergency now"
                        )

                    with st.expander(
                        "🧠 View AI Triage Reasoning"
                    ):
                        st.info(
                            result.get(
                                "triage_reasoning",
                                "",
                            )
                        )

                        st.caption(
                            "Structured Intake Sent "
                            "to Triage Agent:"
                        )

                        st.code(
                            result.get("intake", ""),
                            language=None,
                        )

                    st.subheader("🩺 Patient Summary")

                    st.info(f"""
**Patient:** {patient_name}  |  **Age:** {age}  |  **Gender:** {gender}  |  **Phone:** {phone}

**Body Area:** {body_part}  |  **Duration:** {duration}  |  **Pain Level:** {severity_slider}/10  |  **Onset:** {onset_type}

**Symptoms:** {symptoms_desc}

**Known Conditions:** {conditions_str}

**AI Assessment:** {result.get('summary', '')}
                    """)

                    st.caption(
                        "🕒 Analysis generated on "
                        f"{now_ist().strftime('%d-%m-%Y %H:%M')} IST"
                    )

                    st.subheader("📋 Recommended Actions")

                    for i, action in enumerate(
                        result.get("actions", []),
                        1,
                    ):
                        st.markdown(f"{i}. {action}")

                    warning = str(result.get("warning", "")).strip()

                    if warning and warning.upper() not in {"NONE", "N/A", "NULL"}:
                        st.error(
                            "⚠️ **When to go to Emergency immediately:** "
                            f"{warning}"
                        )

                    st.caption(
                        "ℹ️ This is preliminary AI-assisted triage information, "
                        "not a medical diagnosis. It does not replace professional "
                        "medical advice - always consult a qualified healthcare "
                        "provider for medical decisions."
                    )

                    try:
                        save_case_to_db(
                            patient_name=patient_name.strip() or "Unknown",
                            symptoms=f"{body_part}: {symptoms_desc}",
                            severity=severity,
                            department=department,
                            summary=result.get("summary", ""),
                            recommendation=result.get("recommendation", ""),
                        )

                    except Exception as e:
                        st.warning(
                            "Case could not be saved "
                            f"to history: {e}"
                        )

                    if patient_name.strip():
                        try:
                            # Best-effort sync: keeps the patient's saved
                            # health profile current with whatever they just
                            # entered here, without requiring a separate trip
                            # to the Health Profile tab. Only touches the
                            # fields this form actually collects - height/
                            # weight/blood group (set only in the Health
                            # Profile tab) are left untouched by upsert_profile.
                            upsert_profile(
                                patient_name=patient_name.strip(),
                                age=int(age) if age else None,
                                gender=gender,
                                chronic_conditions=(
                                    conditions_str if conditions_str != "None reported" else "None"
                                ),
                                allergies=allergies.strip() or None,
                                current_medications=current_medications.strip() or None,
                            )
                        except Exception:
                            pass  # profile sync is best-effort, never blocks the triage flow

                    try:
                        pdf_bytes = generate_pdf_report(
                            patient_name=patient_name,
                            age=age,
                            gender=gender,
                            phone=phone,
                            body_part=body_part,
                            symptoms_desc=symptoms_desc,
                            duration=duration,
                            onset_type=onset_type,
                            severity_slider=severity_slider,
                            conditions_str=conditions_str,
                            severity=severity,
                            department=department,
                            urgency=urgency,
                            result=result,
                        )

                        st.download_button(
                            label="📄 Download PDF Report",
                            data=pdf_bytes,
                            file_name=(
                                f"report_"
                                f"{(patient_name or 'patient').replace(' ', '_')}"
                                f".pdf"
                            ),
                            mime="application/pdf",
                            key="download_patient_report",
                        )

                    except Exception as e:
                        show_error_details(
                            e,
                            "Your assessment was generated, but the PDF "
                            "report couldn't be created.",
                        )
