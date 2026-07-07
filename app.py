import requests
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

import sys
import os
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

def now_ist():
    return datetime.now(IST)

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
DB_PATH = os.path.join(ROOT, "data", "hospital.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

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
h1, h2, h3 { color: #2C3E50; }
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
<div style="color:#8B7355;font-size:16px;font-weight:500;letter-spacing:1px;margin-top:-10px;margin-bottom:20px;">
Designed & Developed by Aarya Shirsath
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div style="color:#7A6A58;font-size:17px;font-style:italic;margin-bottom:20px;">
AI-Powered Emergency Assessment & Smart Hospital Routing
</div>
""", unsafe_allow_html=True)
st.markdown("""
### Agentic Hospital Triage & Decision Support System
AI-powered emergency assessment, department routing, doctor workflow management, and patient analytics.
""")


def clean_text_for_pdf(text):
    """Strip/replace characters that fpdf's core (Helvetica) font can't render."""
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "\u2018": "'", "\u2019": "'",   # curly single quotes
        "\u201c": '"', "\u201d": '"',   # curly double quotes
        "\u2013": "-", "\u2014": "-",   # en/em dash
        "\u2026": "...",                # ellipsis
        "\u2022": "-",                  # bullet
        "\u2192": "->",                 # arrow
        "\u00a0": " ",                  # non-breaking space
        "\u00b0": " deg",               # degree sign
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    # Drop any remaining character outside Latin-1 (emojis, other Unicode)
    return text.encode("latin-1", "ignore").decode("latin-1")


def generate_pdf_report(patient_name, age, gender, phone, body_part, symptoms_desc,
                         duration, onset_type, severity_slider, conditions_str,
                         severity, department, urgency, result):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(166, 124, 82)
    pdf.rect(0, 0, 210, 25, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(10, 7)
    pdf.cell(0, 10, "MediAgent AI - Patient Report", ln=True)
    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(10, 30)
    pdf.cell(0, 6, clean_text_for_pdf(f"Generated: {now_ist().strftime('%d-%m-%Y %H:%M')} IST"), ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Patient Details", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, clean_text_for_pdf(
        f"Name: {patient_name}   |   Age: {age}   |   Gender: {gender}   |   Phone: {phone}"))
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Symptom Intake", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, clean_text_for_pdf(
        f"Body Area: {body_part}\n"
        f"Description: {symptoms_desc}\n"
        f"Duration: {duration}   |   Onset: {onset_type}   |   Pain Level: {severity_slider}/10\n"
        f"Known Conditions: {conditions_str}"))
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Triage Result", ln=True)
    sev_colors = {"Critical": (231, 76, 60), "Moderate": (241, 196, 15), "Mild": (46, 204, 113)}
    r, g, b = sev_colors.get(severity, (100, 100, 100))
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(50, 8, clean_text_for_pdf(f" Severity: {severity} "), fill=True, ln=False)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 8, clean_text_for_pdf(f"   Department: {department}   |   Urgency: {urgency}/10"), ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "AI Assessment", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, clean_text_for_pdf(result['summary']))
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Recommended Actions", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for i, action in enumerate(result["actions"], 1):
        pdf.multi_cell(0, 7, clean_text_for_pdf(f"{i}. {action}"))
    if result["warning"]:
        pdf.ln(3)
        pdf.set_fill_color(231, 76, 60)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 8, clean_text_for_pdf(f"EMERGENCY WARNING: {result['warning']}"), fill=True)
    return bytes(pdf.output())


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
        st.markdown("""<div style="background:#FFF5F5;border:1px solid #FECACA;border-radius:12px;padding:12px;text-align:center;margin-bottom:16px;">
            <div style="font-size:22px;">🚑</div>
            <div style="font-weight:700;color:#c0392b;font-size:14px;">Ambulance</div>
            <div style="font-size:18px;font-weight:800;color:#c0392b;">102</div>
        </div>""", unsafe_allow_html=True)
    with sos_col2:
        st.markdown("""<div style="background:#FFF5F5;border:1px solid #FECACA;border-radius:12px;padding:12px;text-align:center;margin-bottom:16px;">
            <div style="font-size:22px;">🏥</div>
            <div style="font-weight:700;color:#c0392b;font-size:14px;">Emergency</div>
            <div style="font-size:18px;font-weight:800;color:#c0392b;">108</div>
        </div>""", unsafe_allow_html=True)
    with sos_col3:
        st.markdown("""<div style="background:#FFF5F5;border:1px solid #FECACA;border-radius:12px;padding:12px;text-align:center;margin-bottom:16px;">
            <div style="font-size:22px;">👮</div>
            <div style="font-weight:700;color:#c0392b;font-size:14px;">Police</div>
            <div style="font-size:18px;font-weight:800;color:#c0392b;">100</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("#### 👤 Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        patient_name = st.text_input("Patient Name")
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        phone = st.text_input("Phone Number")

    st.divider()
    st.markdown("#### 🫀 Symptom Details")
    col3, col4 = st.columns(2)
    with col3:
        body_part = st.selectbox("Primary Body Part / Area Affected", [
            "Select...", "Head / Brain", "Eyes", "Ears / Nose / Throat",
            "Chest / Heart", "Lungs / Breathing", "Abdomen / Stomach",
            "Back / Spine", "Arms / Hands", "Legs / Feet", "Skin",
            "Reproductive / Urinary", "Mental / Psychological", "Whole Body / General",
        ])
        duration = st.selectbox("How long have you had these symptoms?", [
            "Just started (< 1 hour)", "A few hours (1-6 hours)", "Today (6-24 hours)",
            "A few days (2-3 days)", "About a week", "More than a week", "Chronic / ongoing",
        ])
    with col4:
        severity_slider = st.slider("Pain / Discomfort Level", min_value=1, max_value=10, value=5,
                                     help="1 = barely noticeable, 10 = worst imaginable")
        if severity_slider <= 3:
            st.success(f"Level {severity_slider}/10 - Mild discomfort")
        elif severity_slider <= 6:
            st.warning(f"Level {severity_slider}/10 - Moderate discomfort")
        else:
            st.error(f"Level {severity_slider}/10 - Severe discomfort")
        onset_type = st.radio("How did symptoms start?", ["Sudden / Abrupt", "Gradual"], horizontal=True)

    symptoms_desc = st.text_area("Describe your symptoms in detail",
        placeholder="e.g. sharp chest pain radiating to left arm, shortness of breath, dizziness...",
        height=100)

    st.divider()
    st.markdown("#### 🏥 Medical Context")
    col5, col6 = st.columns(2)
    with col5:
        known_conditions = st.multiselect("Known Medical Conditions (if any)", [
            "Diabetes", "Hypertension", "Heart Disease", "Asthma", "Thyroid Disorder",
            "Kidney Disease", "Epilepsy", "Cancer", "HIV/AIDS", "Arthritis",
            "Depression / Anxiety", "None",
        ], default=["None"])
    with col6:
        current_medications = st.text_input("Current Medications (optional)",
            placeholder="e.g. Metformin, Aspirin, Lisinopril")
        allergies = st.text_input("Known Allergies (optional)",
            placeholder="e.g. Penicillin, Sulfa drugs, Latex")

    st.divider()

    if st.button("🔍 Analyze Symptoms", use_container_width=True):
        if body_part == "Select...":
            st.warning("Please select the body part affected.")
        elif not symptoms_desc.strip():
            st.warning("Please describe your symptoms in the text box.")
        else:
            conditions_str = ", ".join([c for c in known_conditions if c != "None"]) or "None reported"
            symptoms = f"""Body Part Affected: {body_part}
Symptom Description: {symptoms_desc.strip()}
Duration: {duration}
Onset: {onset_type}
Pain/Discomfort Level: {severity_slider}/10
Known Conditions: {conditions_str}
Current Medications: {current_medications.strip() or "None reported"}
Allergies: {allergies.strip() or "None reported"}""".strip()

            patient_context = f"Age: {age}, Gender: {gender}"
            progress_bar = st.progress(0)
            status = st.empty()
            status.markdown("🔍 **Agent 1/3:** Validating and normalising intake...")
            progress_bar.progress(10)
            
            st.write("Calling pipeline...")

            result = run_triage_pipeline(symptoms, patient_context)

            st.write("Pipeline finished")
            st.write(result)

            progress_bar.progress(70)
            status.markdown("📋 **Agent 3/3:** Generating recommendations...")
            time.sleep(0.3)
            progress_bar.progress(100)
            time.sleep(0.2)
            progress_bar.empty()
            status.empty()

            if not result["valid"]:
                st.error(f"Could not process input: {result['invalid_reason']}")
                st.info("Please describe your symptoms more specifically.")
            else:
                severity = result["severity"]
                department = result["department"]
                urgency = result["urgency_score"]

                st.markdown("---")
                st.markdown("### 📊 Assessment Results")
                res_col1, res_col2, res_col3 = st.columns(3)
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
                    st.metric("Urgency Score", f"{urgency} / 10")

                st.progress(urgency / 10)
                if severity == "Critical":
                    st.error("🚨 IMMEDIATE MEDICAL ATTENTION REQUIRED - Go to Emergency now")

                with st.expander("🧠 View AI Triage Reasoning"):
                    st.info(result["triage_reasoning"])
                    st.caption("Structured Intake Sent to Triage Agent:")
                    st.code(result["intake"], language=None)

                st.subheader("🩺 Patient Summary")
                st.info(f"""
**Patient:** {patient_name}  |  **Age:** {age}  |  **Gender:** {gender}  |  **Phone:** {phone}

**Body Area:** {body_part}  |  **Duration:** {duration}  |  **Pain Level:** {severity_slider}/10  |  **Onset:** {onset_type}

**Symptoms:** {symptoms_desc}

**Known Conditions:** {conditions_str}

**AI Assessment:** {result['summary']}
                """)
                st.caption(f"🕒 Analysis generated on {now_ist().strftime('%d-%m-%Y %H:%M')} IST")

                st.subheader("📋 Recommended Actions")
                for i, action in enumerate(result["actions"], 1):
                    st.markdown(f"{i}. {action}")

                if result["warning"]:
                    st.error(f"⚠️ **When to go to Emergency immediately:** {result['warning']}")

                save_case_to_db(
                    symptoms=f"{body_part}: {symptoms_desc}",
                    severity=severity,
                    department=department
                )

                pdf_bytes = generate_pdf_report(
                    patient_name, age, gender, phone, body_part, symptoms_desc,
                    duration, onset_type, severity_slider, conditions_str,
                    severity, department, urgency, result
                )
                st.download_button(
                    "📄 Download PDF Report",
                    pdf_bytes,
                    file_name=f"report_{patient_name.replace(' ', '_')}_{now_ist().strftime('%d%m%Y')}.pdf",
                    mime="application/pdf"
                )

# ── TAB 2 ─────────────────────────────────────────────────────────
with tab2:
    st.header("📋 Patient Case History")
    if st.button("🗑 Clear All Cases"):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cases")
        conn.commit()
        conn.close()
        st.success("All case records deleted successfully!")
        st.rerun()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM cases ORDER BY created_at DESC", conn)
        display_df = df[["symptoms", "severity", "department", "created_at"]].copy()
        display_df.columns = ["Symptoms", "Severity", "Department", "Date & Time"]
        display_df["Severity"] = display_df["Severity"].replace({
            "Critical": "🚨 Critical", "Moderate": "⚠️ Moderate",
            "Low": "✅ Low", "Mild": "✅ Mild"
        })
        st.dataframe(display_df, use_container_width=True)
        conn.close()
    except Exception as e:
        st.error(str(e))

# ── TAB 3 ─────────────────────────────────────────────────────────
with tab3:
    st.header("📊 Real-Time Hospital Analytics")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM cases", conn)
    total_cases = len(df)
    critical_cases = len(df[df["severity"].str.contains("Critical", case=False, na=False)])
    moderate_cases = len(df[df["severity"].str.contains("Moderate", case=False, na=False)])
    mild_cases = total_cases - critical_cases - moderate_cases
    critical_percent = round((critical_cases / total_cases) * 100, 1) if total_cases > 0 else 0

    if critical_cases > 10:
        st.error("🚨 Hospital Alert: High Emergency Load")
    else:
        st.success("✅ Hospital Status Normal")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("📁 Cases", total_cases)
    with col2: st.metric("🚨 Critical", critical_cases)
    with col3: st.metric("⚠️ Moderate", moderate_cases)
    with col4: st.metric("✅ Mild", mild_cases)
    with col5: st.metric("📊 Critical %", f"{critical_percent}%")

    dept_df = pd.read_sql_query("SELECT department, COUNT(*) as total FROM cases GROUP BY department", conn)
    fig = px.bar(dept_df, x="department", y="total", text="total")
    fig.update_layout(paper_bgcolor="#F5EFE6", plot_bgcolor="#F5EFE6",
                      font=dict(color="#34495E", size=14),
                      xaxis=dict(title="Department", color="#34495E"),
                      yaxis=dict(title="Cases", color="#34495E"))
    fig.update_traces(marker_color="#B8874E", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    sev_df = pd.read_sql_query("SELECT severity, COUNT(*) as total FROM cases GROUP BY severity", conn)
    st.subheader("Severity Distribution")
    fig2 = px.pie(sev_df, names="severity", values="total")
    fig2.update_layout(paper_bgcolor="#F5EFE6", plot_bgcolor="#F5EFE6", font=dict(color="#34495E", size=14))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("<div style='text-align:center;color:#8B7355;padding-top:20px;'>Designed & Developed by Aarya Shirsath</div>", unsafe_allow_html=True)
    conn.close()

# ── TAB 4 ─────────────────────────────────────────────────────────
with tab4:
    st.header("👨‍⚕️ Doctor Portal")
    conn = sqlite3.connect(DB_PATH)
    doctor_df = pd.read_sql_query("""
        SELECT * FROM cases
        ORDER BY CASE WHEN severity='Critical' THEN 1 WHEN severity='Moderate' THEN 2 ELSE 3 END, created_at DESC
    """, conn)
    critical_count = len(doctor_df[doctor_df["severity"] == "Critical"])
    st.error(f"🚨 Critical Cases Pending: {critical_count}")

    display_df = doctor_df[["symptoms", "severity", "department", "created_at"]].copy()
    display_df.columns = ["Symptoms", "Severity", "Department", "Date & Time"]

    def highlight_severity(row):
        if row["Severity"] == "Critical":
            return ["background-color: #FADADD; color: #7A1F1F"] * len(row)
        elif row["Severity"] == "Moderate":
            return ["background-color: #F8E6C1; color: #7A5C00"] * len(row)
        else:
            return ["background-color: #DCEFD8; color: #1F5D2E"] * len(row)

    styled_df = display_df.style.apply(highlight_severity, axis=1)
    st.dataframe(styled_df, use_container_width=True)
    conn.close()
    st.markdown("---")
    st.caption("🏥 MediAgent AI - Agentic Hospital Triage & Decision Support System")

# ── Drug Checker LLM ──────────────────────────────────────────────
_drug_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)

_drug_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a clinical pharmacist explaining drug interactions to a patient in plain English.
You will receive raw OpenFDA adverse event data about two drugs taken together.
Your job is to summarise what risks exist, how serious they are, and what the patient should do.

Respond in this EXACT format:
SEVERITY: <Major | Moderate | Minor | Unknown>
PLAIN_SUMMARY: <2-3 sentences explaining the interaction in simple language a patient can understand. No jargon.>
MECHANISM: <1 sentence explaining WHY this interaction happens, if known.>
PATIENT_ADVICE: <1-2 sentences on what the patient should do.>

If there is no interaction data found, say:
SEVERITY: Unknown
PLAIN_SUMMARY: No significant interaction data was found in the OpenFDA database for this drug combination.
MECHANISM: Not available.
PATIENT_ADVICE: Always consult your doctor or pharmacist before combining medications."""),
    ("human", "Drug 1: {drug1}\nDrug 2: {drug2}\nOpenFDA Data Summary: {fda_data}")
])

_drug_chain = _drug_prompt | _drug_llm | StrOutputParser()


def _query_openfda(drug1: str, drug2: str) -> dict:
    base = "https://api.fda.gov/drug/label.json"
    try:
        resp = requests.get(base, params={
            "search": f'drug_interactions:"{drug2.lower()}"&openfda.brand_name:"{drug1.lower()}"',
            "limit": 3
        }, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            total = data.get("meta", {}).get("results", {}).get("total", 0)
            results = data.get("results", [])
            interactions = []
            for r in results:
                di = r.get("drug_interactions", [])
                if di:
                    interactions.extend(di[:2])
            return {"found": total > 0, "count": total, "reactions": interactions[:5], "raw": data}
        return {"found": False, "count": 0, "reactions": [], "raw": {}}
    except Exception as e:
        return {"found": False, "count": 0, "reactions": [], "raw": {}, "error": str(e)}


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
        "Zero AI hallucinations on interaction data."
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
    if ex_col1.button("Aspirin + Warfarin"):
        st.session_state["drug1"] = "Aspirin"
        st.session_state["drug2"] = "Warfarin"
        st.rerun()
    if ex_col2.button("Metformin + Ibuprofen"):
        st.session_state["drug1"] = "Metformin"
        st.session_state["drug2"] = "Ibuprofen"
        st.rerun()
    if ex_col3.button("Lisinopril + Potassium"):
        st.session_state["drug1"] = "Lisinopril"
        st.session_state["drug2"] = "Potassium"
        st.rerun()
    if ex_col4.button("Sertraline + Tramadol"):
        st.session_state["drug1"] = "Sertraline"
        st.session_state["drug2"] = "Tramadol"
        st.rerun()

    st.divider()

    if st.button("🔍 Check Interaction", use_container_width=True):
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

                with st.spinner("Generating clinical explanation..."):
                    llm_output = _drug_chain.invoke({
                        "drug1": drug1.strip(),
                        "drug2": drug2.strip(),
                        "fda_data": fda_summary,
                    })

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

                st.markdown(f"**OpenFDA Entries Found:** {fda_result['count']:,}")
                if fda_result["reactions"]:
                    st.markdown("**Interaction Warnings from FDA Label:**")
                    for rxn in fda_result["reactions"][:2]:
                        st.warning(rxn[:300] + "..." if len(rxn) > 300 else rxn)

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