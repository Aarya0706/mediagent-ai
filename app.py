import requests
from groq import Groq
import tempfile
import time
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from datetime import datetime
from fpdf import FPDF
from fpdf.enums import WrapMode, XPos, YPos
from dotenv import load_dotenv

import sys
import os
from tools.save_case import save_case_to_db, update_case_status
from zoneinfo import ZoneInfo
load_dotenv()

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

IST = ZoneInfo("Asia/Kolkata")


def now_ist():
    return datetime.now(IST)


ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
DB_PATH = os.path.join(ROOT, "data", "hospital.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def ensure_patient_name_column():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(cases)")
        columns = [row[1] for row in cursor.fetchall()]

        if "patient_name" not in columns:
            cursor.execute(
                "ALTER TABLE cases ADD COLUMN patient_name TEXT DEFAULT 'Unknown'"
            )
            conn.commit()


ensure_patient_name_column()

from agents.pipeline import run_triage_pipeline
from tools.save_case import save_case_to_db

 
st.set_page_config(
    page_title="MediAgent AI",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
.stApp { background-color: #FAF7F2; }
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #F8F4EE 0%, #F1E7D8 50%, #EFE6D8 100%);
}
h1, h2, h3, h4 { color: #2C3E50; }
p, label, div { color: #34495E; }
[data-testid="stTextArea"] textarea {
    background-color: #FFFFFF !important;
    color: #1A1A1A !important;
    border: 1.5px solid #A67C52 !important;
    border-radius: 10px !important;
}
.stButton > button {
    background-color: #A67C52;
    color: white;
    border-radius: 15px;
    border: none;
    font-weight: 600;
    padding: 10px 20px;
    transition: all 0.3s ease;
}
.stButton > button:hover { background: #8B6A45; transform: translateY(-2px); }
button[data-baseweb="tab"] { border-radius: 12px; }
[data-testid="stAlert"] { border-radius: 20px; box-shadow: 0 8px 24px rgba(0,0,0,0.08); }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🏥 MediAgent AI")

st.markdown("""
### Agentic Hospital Triage & Decision Support System
AI-powered emergency assessment, department routing, doctor workflow management, drug interaction checking, and patient analytics.
""")


# ── PDF helpers ──────────────────────────────────────────────────

def clean_text_for_pdf(text, max_word_len=40):
    """Strip/replace characters that fpdf's core (Helvetica) font can't render,
    and break up unbroken long tokens so multi_cell can always wrap them."""
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-",
        "\u2026": "...",
        "\u2022": "- ",
        "\u2192": " -> ",
        "\u00a0": " ",
        "\u00b0": " deg",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Drop any remaining character outside Latin-1 (emojis, other Unicode)
    text = text.encode("latin-1", "ignore").decode("latin-1")

    # Force-break any "word" longer than max_word_len so it never overflows the page width
    words = text.split(" ")
    safe_words = []
    for w in words:
        if len(w) > max_word_len:
            safe_words.append(" ".join(w[i:i + max_word_len] for i in range(0, len(w), max_word_len)))
        else:
            safe_words.append(w)
    return " ".join(safe_words)


def generate_pdf_report(
    patient_name, age, gender, phone, body_part, symptoms_desc,
    duration, onset_type, severity_slider, conditions_str,
    severity, department, urgency, result
):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=8)
    pdf.add_page()

    # ---------- HEADER ----------
    pdf.set_fill_color(166, 124, 82)
    pdf.rect(0, 0, 210, 25, "F")

    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(10, 7)
    pdf.cell(190, 10, "MediAgent AI - Patient Report")

    pdf.set_text_color(40, 40, 40)
    pdf.set_y(30)

    # ---------- GENERATED TIME ----------
    pdf.set_font("Helvetica", "", 10)

    generated_time = clean_text_for_pdf(
        f"Generated: {now_ist().strftime('%d-%m-%Y %H:%M')} IST"
    )

    pdf.cell(0, 7, generated_time)
    pdf.ln(11)

    # ---------- PATIENT DETAILS ----------
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Patient Details")
    pdf.ln(9)

    pdf.set_font("Helvetica", "", 11)

    patient_text = clean_text_for_pdf(
        f"Name: {patient_name}\n"
        f"Age: {age}\n"
        f"Gender: {gender}\n"
        f"Phone: {phone}"
    )

    pdf.multi_cell(0, 7, patient_text)
    pdf.ln(3)

    # ---------- SYMPTOM INTAKE ----------
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Symptom Intake")
    pdf.ln(9)

    pdf.set_font("Helvetica", "", 11)

    symptom_text = clean_text_for_pdf(
        f"Body Area: {body_part}\n"
        f"Description: {symptoms_desc}\n"
        f"Duration: {duration}\n"
        f"Onset: {onset_type}\n"
        f"Pain Level: {severity_slider}/10\n"
        f"Known Conditions: {conditions_str}"
    )

    pdf.multi_cell(0, 7, symptom_text)
    pdf.ln(3)

    # ---------- TRIAGE RESULT ----------
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Triage Result")
    pdf.ln(9)

    pdf.set_font("Helvetica", "", 11)

    triage_text = clean_text_for_pdf(
        f"Severity: {severity}\n"
        f"Department: {department}\n"
        f"Urgency: {urgency}/10"
    )

    pdf.multi_cell(0, 7, triage_text)
    pdf.ln(3)

    # ---------- AI ASSESSMENT ----------
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "AI Assessment")
    pdf.ln(9)

    pdf.set_font("Helvetica", "", 11)

    summary = clean_text_for_pdf(
        result.get("summary", "No assessment available.")
    )

    pdf.multi_cell(0, 7, summary)
    pdf.ln(3)

    # ---------- RECOMMENDED ACTIONS ----------
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Recommended Actions")
    pdf.ln(9)

    pdf.set_font("Helvetica", "", 11)

    actions = result.get("actions", [])

    if actions:
        for i, action in enumerate(actions, 1):
            action_text = clean_text_for_pdf(f"{i}. {action}")
            pdf.multi_cell(0, 7, action_text)
            pdf.ln(1)
    else:
        pdf.multi_cell(0, 7, "No recommended actions available.")

    # ---------- EMERGENCY WARNING ----------
    warning = result.get("warning")

    if (
        warning
        and str(warning).strip()
        and str(warning).strip().upper() != "NONE"
    ):
        pdf.ln(4)

        pdf.set_fill_color(231, 76, 60)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 11)

        warning_text = clean_text_for_pdf(
            f"EMERGENCY WARNING: {str(warning).strip()}"
        )

        pdf.multi_cell(
            0,
            8,
            warning_text,
            fill=True
        )

        pdf.set_text_color(40, 40, 40)
    # ---------- FOOTER ----------
    pdf.ln(3)

    pdf.set_font("Helvetica", "I", 9)

    disclaimer = clean_text_for_pdf(
        "Disclaimer: MediAgent AI provides preliminary AI-assisted triage "
        "information and does not replace professional medical diagnosis "
        "or treatment."
    )

    pdf.multi_cell(0, 6, disclaimer)

    # fpdf2 2.x returns bytearray from output()
    pdf_data = pdf.output()

    return bytes(pdf_data)


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Patient Triage", "Case History", "Dashboard", "Doctor Portal", "💊 Drug Checker",
])

# ── TAB 1 ─────────────────────────────────────────────────────────
with tab1:
    st.header("🩺 Patient Symptom Analysis")

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
        patient_name = st.text_input("Patient Name")
        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            step=1,
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Other"],
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
            default=["None"],
        )

    with col6:
        current_medications = st.text_input(
            "Current Medications (optional)",
            placeholder="e.g. Metformin, Aspirin, Lisinopril",
        )

        allergies = st.text_input(
            "Known Allergies (optional)",
            placeholder="e.g. Penicillin, Sulfa drugs, Latex",
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

            progress_bar = st.progress(0)
            status = st.empty()

            status.markdown(
                "🔍 **Agent 1/3:** "
                "Validating and normalising intake..."
            )
            progress_bar.progress(10)

            try:
                result = run_triage_pipeline(
                    symptoms,
                    patient_context,
                )

            except Exception as e:
                progress_bar.empty()
                status.empty()

                st.error(
                    "The triage pipeline failed to run."
                )
                st.exception(e)

                result = None

            if result is not None:
                progress_bar.progress(70)

                status.markdown(
                    "📋 **Agent 3/3:** "
                    "Generating recommendations..."
                )

                time.sleep(0.3)

                progress_bar.progress(100)

                time.sleep(0.2)

                progress_bar.empty()
                status.empty()

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
                        st.exception(e)
# ── TAB 2 ─────────────────────────────────────────────────────────
with tab2:
    st.header("📋 Patient Case History")

    # Search by patient name, symptoms, or department
    search_query = st.text_input(
        "🔍 Search Cases",
        placeholder="Search by patient name, symptoms, or department...",
        key="case_history_search"
    )

    if st.button(
        "🗑 Clear All Cases",
        key="clear_all_cases"
    ):
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM cases")
            conn.commit()

        st.success("All case records deleted successfully!")
        st.rerun()

    try:
        with sqlite3.connect(DB_PATH) as conn:

            if search_query.strip():

                search_value = f"%{search_query.strip().lower()}%"

                df = pd.read_sql_query(
                    """
                    SELECT *
                    FROM cases
                    WHERE LOWER(COALESCE(patient_name, '')) LIKE ?
                       OR LOWER(COALESCE(symptoms, '')) LIKE ?
                       OR LOWER(COALESCE(department, '')) LIKE ?
                    ORDER BY created_at DESC
                    """,
                    conn,
                    params=(
                        search_value,
                        search_value,
                        search_value
                    )
                )

            else:
                df = pd.read_sql_query(
                    """
                    SELECT *
                    FROM cases
                    ORDER BY created_at DESC
                    """,
                    conn
                )

        if df.empty:

            if search_query.strip():
                st.info("No matching cases found.")
            else:
                st.info("No patient cases available yet.")

        else:
            display_df = df[
                [
                    "patient_name",
                    "symptoms",
                    "severity",
                    "department",
                    "created_at"
                ]
            ].copy()

            display_df.columns = [
                "Patient Name",
                "Symptoms",
                "Severity",
                "Department",
                "Date & Time"
            ]

            display_df["Severity"] = (
                display_df["Severity"]
                .fillna("Unknown")
                .replace({
                    "Critical": "🚨 Critical",
                    "Moderate": "⚠️ Moderate",
                    "Low": "✅ Low",
                    "Mild": "✅ Mild"
                })
            )

            st.dataframe(
                display_df,
                width="stretch",
                hide_index=True
            )

    except Exception as e:
        st.error(f"Could not load case history: {e}")


# ── TAB 3 ─────────────────────────────────────────────────────────
with tab3:
    st.header("📊 Real-Time Hospital Analytics")

    try:
        with sqlite3.connect(DB_PATH) as conn:

            df = pd.read_sql_query(
                "SELECT * FROM cases",
                conn
            )

            dept_df = pd.read_sql_query(
                """
                SELECT department, COUNT(*) AS total
                FROM cases
                GROUP BY department
                ORDER BY total DESC
                """,
                conn
            )

            sev_df = pd.read_sql_query(
                """
                SELECT severity, COUNT(*) AS total
                FROM cases
                GROUP BY severity
                """,
                conn
            )

        # ── Dashboard Metrics ──────────────────────────────────────

        total_cases = len(df)

        critical_cases = len(
            df[
                df["severity"].str.contains(
                    "Critical",
                    case=False,
                    na=False
                )
            ]
        )

        moderate_cases = len(
            df[
                df["severity"].str.contains(
                    "Moderate",
                    case=False,
                    na=False
                )
            ]
        )

        mild_cases = len(
            df[
                df["severity"].str.contains(
                    "Mild",
                    case=False,
                    na=False
                )
            ]
        )

        critical_percent = (
            round(
                (critical_cases / total_cases) * 100,
                1
            )
            if total_cases > 0
            else 0
        )

        if critical_cases > 10:
            st.error("🚨 Hospital Alert: High Emergency Load")
        else:
            st.success("✅ Hospital Status Normal")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("📁 Cases", total_cases)

        with col2:
            st.metric("🚨 Critical", critical_cases)

        with col3:
            st.metric("⚠️ Moderate", moderate_cases)

        with col4:
            st.metric("✅ Mild", mild_cases)

        with col5:
            st.metric(
                "📊 Critical %",
                f"{critical_percent}%"
            )


        # ── Cases Over Time Line Chart ─────────────────────────────

        st.subheader("📈 Cases Over Time")

        if df.empty:

            st.info("No case data available for timeline analytics.")

        else:

            timeline_df = df.copy()

            timeline_df["created_at"] = pd.to_datetime(
                timeline_df["created_at"],
                errors="coerce"
            )

            timeline_df = timeline_df.dropna(
                subset=["created_at"]
            )

            if timeline_df.empty:

                st.info("No valid case timestamps available.")

            else:

                timeline_df["Date"] = (
                    timeline_df["created_at"].dt.date
                )

                daily_cases = (
                    timeline_df
                    .groupby("Date")
                    .size()
                    .reset_index(name="Cases")
                    .sort_values("Date")
                )

                fig_time = px.line(
                    daily_cases,
                    x="Date",
                    y="Cases",
                    markers=True
                )

                fig_time.update_layout(
                    paper_bgcolor="#F5EFE6",
                    plot_bgcolor="#F5EFE6",

                    font=dict(
                        color="#34495E",
                        size=14
                    ),

                    xaxis=dict(
                        title="Date",
                        color="#34495E"
                    ),

                    yaxis=dict(
                        title="Number of Cases",
                        color="#34495E",
                        rangemode="tozero"
                    ),

                    hovermode="x unified"
                )

                fig_time.update_traces(
                    line=dict(width=3),
                    marker=dict(size=9)
                )

                st.plotly_chart(
                    fig_time,
                    width="stretch"
                )


        # ── Department Distribution ────────────────────────────────

        st.subheader("🏥 Cases by Department")

        if dept_df.empty:

            st.info("No department data available.")

        else:

            fig = px.bar(
                dept_df,
                x="department",
                y="total",
                text="total"
            )

            fig.update_layout(
                paper_bgcolor="#F5EFE6",
                plot_bgcolor="#F5EFE6",

                font=dict(
                    color="#34495E",
                    size=14
                ),

                xaxis=dict(
                    title="Department",
                    color="#34495E"
                ),

                yaxis=dict(
                    title="Cases",
                    color="#34495E"
                )
            )

            fig.update_traces(
                marker_color="#B8874E",
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )


        # ── Severity Distribution ──────────────────────────────────

        st.subheader("📊 Severity Distribution")

        if sev_df.empty:

            st.info("No severity data available.")

        else:

            fig2 = px.pie(
                sev_df,
                names="severity",
                values="total"
            )

            fig2.update_layout(
                paper_bgcolor="#F5EFE6",
                plot_bgcolor="#F5EFE6",

                font=dict(
                    color="#34495E",
                    size=14
                )
            )

            st.plotly_chart(
                fig2,
                width="stretch"
            )

    except Exception as e:
        st.error(f"Could not load hospital analytics: {e}")
     

# ── TAB 4 ─────────────────────────────────────────────────────────
with tab4:
    st.header("👨‍⚕️ Doctor Portal")

    # ==========================================================
    # LOAD CASES
    # ==========================================================

    with sqlite3.connect(DB_PATH) as conn:
        doctor_df = pd.read_sql_query(
            """
            SELECT *
            FROM cases
            ORDER BY
                CASE
                    WHEN severity = 'Critical' THEN 1
                    WHEN severity = 'Moderate' THEN 2
                    ELSE 3
                END,
                created_at DESC
            """,
            conn,
        )

    if doctor_df.empty:
        st.info("No patient cases available.")

    else:
        # ======================================================
        # DEPARTMENT FILTER
        # ======================================================

        departments = sorted(
            doctor_df["department"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_department = st.selectbox(
            "🏥 Filter by Department",
            ["All Departments"] + departments,
        )

        if selected_department != "All Departments":
            filtered_df = doctor_df[
                doctor_df["department"] == selected_department
            ].copy()
        else:
            filtered_df = doctor_df.copy()

        # ======================================================
        # CRITICAL PENDING COUNTER
        # ======================================================

        critical_pending = len(
            doctor_df[
                (doctor_df["severity"] == "Critical")
                & (doctor_df["status"] != "Resolved")
            ]
        )

        if critical_pending > 0:
            st.error(
                f"🚨 Critical Cases Pending: {critical_pending}"
            )
        else:
            st.success("✅ No Critical Cases Pending")

        # ======================================================
        # TIME AGO HELPER
        # ======================================================

        def time_ago(timestamp):
            try:
                created = pd.to_datetime(timestamp)

                now = pd.Timestamp.now()

                seconds = int((now - created).total_seconds())

                if seconds < 0:
                    seconds = 0

                if seconds < 60:
                    return "Just now"

                minutes = seconds // 60

                if minutes < 60:
                    return f"{minutes} min ago"

                hours = minutes // 60

                if hours < 24:
                    return f"{hours} hr ago"

                days = hours // 24

                if days < 30:
                    return f"{days} day{'s' if days != 1 else ''} ago"

                return created.strftime("%d-%m-%Y")

            except Exception:
                return str(timestamp)

        # ======================================================
        # STATUS MANAGEMENT
        # ======================================================

        st.subheader("📋 Patient Case Queue")

        if filtered_df.empty:
            st.info("No cases found for this department.")

        else:
            for _, case in filtered_df.iterrows():

                case_id = int(case["id"])

                status = case.get("status", "Pending")

                if pd.isna(status) or not status:
                    status = "Pending"

                time_text = time_ago(case["created_at"])

                severity_icon = {
                    "Critical": "🔴",
                    "Moderate": "🟡",
                    "Mild": "🟢",
                }.get(case["severity"], "⚪")

                status_icon = {
                    "Pending": "⏳",
                    "In Progress": "🩺",
                    "Resolved": "✅",
                }.get(status, "⏳")

                with st.container(border=True):

                    col1, col2, col3, col4 = st.columns(
                        [3, 2, 2, 2]
                    )

                    with col1:
                        st.markdown(
                            f"### {severity_icon} "
                            f"{case['patient_name']}"
                        )

                        st.caption(
                            f"Case #{case_id} • {time_text}"
                        )

                    with col2:
                        st.markdown("**Department**")
                        st.write(case["department"])

                    with col3:
                        st.markdown("**Severity**")
                        st.write(case["severity"])

                    with col4:
                        st.markdown("**Status**")
                        st.write(
                            f"{status_icon} {status}"
                        )

                    st.markdown(
                        f"**Symptoms:** {case['symptoms']}"
                    )

                    # ------------------------------------------
                    # STATUS BUTTONS
                    # ------------------------------------------

                    btn1, btn2, btn3 = st.columns(3)

                    with btn1:
                        if st.button(
                            "⏳ Pending",
                            key=f"pending_{case_id}",
                            disabled=(status == "Pending"),
                            use_container_width=True,
                        ):
                            update_case_status(
                                case_id,
                                "Pending",
                            )

                            st.rerun()

                    with btn2:
                        if st.button(
                            "🩺 In Progress",
                            key=f"progress_{case_id}",
                            disabled=(status == "In Progress"),
                            use_container_width=True,
                        ):
                            update_case_status(
                                case_id,
                                "In Progress",
                            )

                            st.rerun()

                    with btn3:
                        if st.button(
                            "✅ Resolved",
                            key=f"resolved_{case_id}",
                            disabled=(status == "Resolved"),
                            use_container_width=True,
                        ):
                            update_case_status(
                                case_id,
                                "Resolved",
                            )

                            st.rerun()

        st.markdown("---")
    
# ── Drug Checker LLM (cached so it isn't rebuilt on every rerun) ──
@st.cache_resource
def get_drug_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")
    )


_drug_llm = get_drug_llm()

_drug_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a clinical pharmacist explaining drug interactions to a patient in plain English.
You will receive raw OpenFDA adverse event data about two drugs taken together.
Your job is to summarise what risks exist, how serious they are, and what the patient should do.

Respond in this EXACT format:
SEVERITY: <Major | Moderate | Minor | Unknown>
PLAIN_SUMMARY: <2-3 sentences explaining the interaction in simple language a patient can understand. No jargon.>
MECHANISM: <1 sentence explaining WHY this interaction happens, if known.>
PATIENT_ADVICE: <1-2 sentences on what the patient should do.>

If there is no relevant interaction data found, respond EXACTLY:

SEVERITY: Unknown
PLAIN_SUMMARY: Insufficient evidence was found in the OpenFDA drug label database to assess this drug combination. This does not mean the combination is safe or unsafe.
MECHANISM: Not available from the retrieved evidence.
PATIENT_ADVICE: Consult a doctor or pharmacist for guidance specific to this medication combination.

IMPORTANT:
- Never infer that a drug combination is safe because no interaction data was found.
- Never invent potential risks when the retrieved evidence is insufficient.
- Base the explanation only on the retrieved OpenFDA evidence."""),
    ("human", "Drug 1: {drug1}\nDrug 2: {drug2}\nOpenFDA Data Summary: {fda_data}")
])

_drug_chain = _drug_prompt | _drug_llm | StrOutputParser()
DRUG_NAME_ALIASES = {
    "pacemol": "acetaminophen",
    "paracetamol": "acetaminophen",
    "crocin": "acetaminophen",
    "calpol": "acetaminophen",
    "dolo": "acetaminophen",
    "dolo 650": "acetaminophen",

    "ecosprin": "aspirin",
    "disprin": "aspirin",

    "brufen": "ibuprofen",
    "advil": "ibuprofen",

    "augmentin": "amoxicillin clavulanate",
    "amoxyclav": "amoxicillin clavulanate",

    "zithromax": "azithromycin",
    "azee": "azithromycin",
}


def normalize_drug_name(drug_name: str) -> str:
    cleaned_name = drug_name.strip().lower()
    return DRUG_NAME_ALIASES.get(cleaned_name, cleaned_name)


def _query_openfda(drug1: str, drug2: str) -> dict:
    drug1 = normalize_drug_name(drug1)
    drug2 = normalize_drug_name(drug2)

    base = "https://api.fda.gov/drug/label.json"

    # Terms allowed for evidence matching.
    # Keep these conservative to reduce false positives.
    EVIDENCE_TERMS = {
        "warfarin": [
            "warfarin",
            "coumarin anticoagulant",
            "coumarin anticoagulants",
        ],
        "aspirin": [
            "aspirin",
            "acetylsalicylic acid",
        ],
        "ibuprofen": [
            "ibuprofen",
        ],
        "acetaminophen": [
            "acetaminophen",
            "paracetamol",
        ],
        "sertraline": [
            "sertraline",
        ],
        "tramadol": [
            "tramadol",
        ],
    }

    def get_evidence_terms(drug_name: str) -> list:
        return EVIDENCE_TERMS.get(drug_name, [drug_name])

    def extract_evidence(results: list, target_drug: str) -> list:
        evidence_found = []
        target_terms = get_evidence_terms(target_drug)

        for result in results:
            for section in result.get("drug_interactions", []):
                sentences = section.replace("\n", " ").split(".")

                for i, sentence in enumerate(sentences):
                    sentence_lower = sentence.lower()

                    if any(
                        term.lower() in sentence_lower
                        for term in target_terms
                    ):
                        start = max(0, i - 1)
                        end = min(len(sentences), i + 2)

                        evidence = ". ".join(
                            sentences[start:end]
                        ).strip()

                        if evidence:
                            evidence_found.append(evidence)

        return evidence_found

    def fetch_labels(drug_name: str) -> list:
        resp = requests.get(
            base,
            params={
                "search": (
                    f'(openfda.generic_name:"{drug_name}" OR '
                    f'openfda.brand_name:"{drug_name}" OR '
                    f'openfda.substance_name:"{drug_name}")'
                ),
                "limit": 100,
            },
            timeout=10,
        )

        if resp.status_code != 200:
            return []

        return resp.json().get("results", [])

    try:
        # Fetch FDA labels separately
        drug1_results = fetch_labels(drug1)
        drug2_results = fetch_labels(drug2)

        labels_checked = len(drug1_results) + len(drug2_results)

        relevant_interactions = []

        # In Drug 1 labels, search for evidence terms describing Drug 2
        relevant_interactions.extend(
            extract_evidence(drug1_results, drug2)
        )

        # In Drug 2 labels, search for evidence terms describing Drug 1
        relevant_interactions.extend(
            extract_evidence(drug2_results, drug1)
        )

        # Remove duplicate evidence
        relevant_interactions = list(
            dict.fromkeys(relevant_interactions)
        )

        return {
            "found": len(relevant_interactions) > 0,
            "count": len(relevant_interactions),
            "interactions": relevant_interactions[:5],
            "raw": {
                "drug1": drug1,
                "drug2": drug2,
                "labels_checked": labels_checked,
            },
        }

    except Exception as e:
        return {
            "found": False,
            "count": 0,
            "interactions": [],
            "raw": {},
            "error": str(e),
        }


def _parse_drug_field(text: str, field: str) -> str:
    for line in text.split("\n"):
        if line.strip().upper().startswith(field.upper() + ":"):
            return line.split(":", 1)[1].strip()
    return ""


# ── TAB 5 ─────────────────────────────────────────────────────────
with tab5:
    st.header("💊 Drug Interaction Checker")
    st.markdown(
        "Check for potential interactions between two medications. "
        "Results are sourced live from **OpenFDA** - a real pharmacological database. "
        "Interaction evidence is retrieved from FDA drug labels and summarized by AI for easier understanding."
    )
    st.caption("Source: U.S. Food & Drug Administration (api.fda.gov) - No API key required")

    st.divider()

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        drug1 = st.text_input("💊 Drug 1",
            value=st.session_state.get("drug1", ""),
            key="drug1_input",
            placeholder="e.g. Aspirin")
    with d_col2:
        drug2 = st.text_input("💊 Drug 2",
            value=st.session_state.get("drug2", ""),
            key="drug2_input",
            placeholder="e.g. Warfarin")

    st.markdown("**Quick examples:**")
    ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)
    if ex_col1.button("Aspirin + Warfarin", width="stretch"):
        st.session_state["drug1"] = "Aspirin"
        st.session_state["drug2"] = "Warfarin"
        st.rerun()
    if ex_col2.button("Metformin + Ibuprofen", width="stretch"):
        st.session_state["drug1"] = "Metformin"
        st.session_state["drug2"] = "Ibuprofen"
        st.rerun()
    if ex_col3.button("Lisinopril + Potassium", width="stretch"):
        st.session_state["drug1"] = "Lisinopril"
        st.session_state["drug2"] = "Potassium"
        st.rerun()
    if ex_col4.button("Sertraline + Tramadol", width="stretch"):
        st.session_state["drug1"] = "Sertraline"
        st.session_state["drug2"] = "Tramadol"
        st.rerun()

    st.divider()

    if st.button("🔍 Check Interaction", width="stretch"):
        if not drug1.strip() or not drug2.strip():
            st.warning("Please enter both drug names.")
        elif drug1.strip().lower() == drug2.strip().lower():
            st.warning("Please enter two different drug names.")
        else:
            with st.spinner(f"Querying OpenFDA for {drug1} + {drug2}..."):
                fda_result = _query_openfda(drug1.strip(), drug2.strip())

            if "error" in fda_result:
                st.error(f"OpenFDA API error: {fda_result['error']}")
            else:
                if fda_result["found"]:
                    fda_summary = (
                        f"Found {fda_result['count']:,} drug label entries "
                        f"mentioning {drug2} interactions with {drug1}. "
                        f"Interaction text available."
                    )
                else:
                    fda_summary = f"No interaction data found in OpenFDA for {drug1} and {drug2}."

                try:
                    with st.spinner("Generating clinical explanation..."):
                        llm_output = _drug_chain.invoke({
                            "drug1": drug1.strip(),
                            "drug2": drug2.strip(),
                            "fda_data": fda_summary,
                        })
                except Exception as e:
                    st.error("The clinical explanation could not be generated.")
                    st.exception(e)
                    llm_output = ""

                if llm_output:
                    severity_label = _parse_drug_field(llm_output, "SEVERITY")
                    plain_summary  = _parse_drug_field(llm_output, "PLAIN_SUMMARY")
                    mechanism      = _parse_drug_field(llm_output, "MECHANISM")
                    patient_advice = _parse_drug_field(llm_output, "PATIENT_ADVICE")

                    st.markdown("### 📊 Interaction Result")
                    sev_lower = severity_label.lower()
                    if "major" in sev_lower:
                        st.error(f"🔴 **Severity: {severity_label}** - Significant risk. Consult your doctor immediately.")
                    elif "moderate" in sev_lower:
                        st.warning(f"🟡 **Severity: {severity_label}** - Use with caution. Doctor consultation advised.")
                    elif "minor" in sev_lower:
                        st.success(f"🟢 **Severity: {severity_label}** - Low risk. Monitor for any unusual symptoms.")
                    else:
                        st.info(f"⚪ **Severity: {severity_label}** - Insufficient data to assess risk.")

                    st.markdown(
                        f"**Relevant FDA Evidence Snippets Found: {fda_result['count']}**"
                    )

                    relevant_warnings = fda_result.get("interactions", [])

                    if relevant_warnings:
                        for warning in relevant_warnings[:2]:
                                st.warning(
                                    warning[:500] + "..."
                                    if len(warning) > 500
                                    else warning
                                )

                    st.divider()
                    ic1, ic2 = st.columns(2)
                    with ic1:
                        st.markdown("**📝 Plain English Summary**")
                        st.info(plain_summary or "Not available.")
                        st.markdown("**🔬 Mechanism**")
                        st.info(mechanism or "Not available.")
                    with ic2:
                        st.markdown("**✅ What You Should Do**")
                        st.warning(patient_advice or "Consult your doctor or pharmacist.")
                        st.markdown("**💊 Drug Pair Checked**")
                        st.code(f"{drug1.strip()}  +  {drug2.strip()}", language=None)

                    st.divider()
                    st.caption(
                        "Interaction data sourced from OpenFDA. "
                        "Clinical explanation generated by Groq LLaMA3. "
                        "This tool is for informational purposes only - always consult a licensed pharmacist or physician."
                    )
                    
st.markdown("""
<div style="
margin-top: 60px;
padding: 22px 0;
border-top: 1px solid rgba(166, 124, 82, 0.25);
text-align: center;
color: #8B7355;
font-size: 14px;
">
MediAgent AI &nbsp;•&nbsp; Developed by <b>Aarya Shirsath</b>
</div>
""", unsafe_allow_html=True)