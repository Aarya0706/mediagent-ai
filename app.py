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
 

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.pipeline import run_triage_pipeline
from tools.save_case import save_case_to_db

  
st.set_page_config(
    page_title="MediAgent AI",
    page_icon="🏥",
    layout="wide"
)
st.markdown("""
<style>

.stApp {
    background-color: #FAF7F2;
}

h1, h2, h3 {
    color: #2C3E50;
}

p, label, div {
    color: #34495E;
}

[data-testid="stTextArea"] textarea {
    background-color: #FFFDF8 !important;
    color: #2C3E50 !important;
    border: 1px solid #D6CFC7 !important;
    border-radius: 10px;
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

.stButton > button:hover {
    background: #8B6A45;
    transform: translateY(-2px);
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        135deg,
        #F8F4EE 0%,
        #F1E7D8 50%,
        #EFE6D8 100%
    );
}

h1, h2, h3 {
    color: #2C3E50;
}

p, label, div {
    color: #34495E;
}



.stButton > button:hover {
    background: #8B6A45;
    transform: translateY(-2px);
}

[data-testid="stTextArea"] textarea {
    background: rgba(255,255,255,0.85) !important;
    backdrop-filter: blur(10px);
    border: 2px solid #D8C3A5 !important;
    border-radius: 18px !important;
    padding: 15px !important;

}
button[data-baseweb="tab"] {
    border-radius: 12px;
}
[data-testid="stAlert"] {
    border-radius: 20px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
# 🏥 MediAgent AI
""")

st.markdown("""
<div style="
color:#8B7355;
font-size:16px;
font-weight:500;
letter-spacing:1px;
margin-top:-10px;
margin-bottom:20px;
">
Designed & Developed by Aarya Shirsath
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
color:#7A6A58;
font-size:17px;
font-style:italic;
margin-bottom:20px;
">
AI-Powered Emergency Assessment & Smart Hospital Routing
</div>
""", unsafe_allow_html=True)

st.markdown("""
### Agentic Hospital Triage & Decision Support System

AI-powered emergency assessment, department routing,
doctor workflow management, and patient analytics.
""")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Patient Triage",
    "Case History",
    "Dashboard",
    "Doctor Portal",    
    "💊 Drug Checker",
])


# -------------------------
# TAB 1 - TRIAGE
# -------------------------
with tab1:
    st.header("🩺 Patient Symptom Analysis")
 
    # ── Section 1: Patient Info ───────────────────────────────────
    st.markdown("#### 👤 Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        patient_name = st.text_input("Patient Name")
        age          = st.number_input("Age", min_value=0, max_value=120, step=1)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        phone  = st.text_input("Phone Number")
 
    st.divider()
 
    # ── Section 2: Structured Symptom Intake ─────────────────────
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
            ]
        )
 
        duration = st.selectbox(
            "How long have you had these symptoms?",
            [
                "Just started (< 1 hour)",
                "A few hours (1–6 hours)",
                "Today (6–24 hours)",
                "A few days (2–3 days)",
                "About a week",
                "More than a week",
                "Chronic / ongoing",
            ]
        )
 
    with col4:
        severity_slider = st.slider(
            "Pain / Discomfort Level",
            min_value=1,
            max_value=10,
            value=5,
            help="1 = barely noticeable, 10 = worst imaginable"
        )
 
        # Visual label for slider
        if severity_slider <= 3:
            st.success(f"Level {severity_slider}/10 — Mild discomfort")
        elif severity_slider <= 6:
            st.warning(f"Level {severity_slider}/10 — Moderate discomfort")
        else:
            st.error(f"Level {severity_slider}/10 — Severe discomfort")
 
        onset_type = st.radio(
            "How did symptoms start?",
            ["Sudden / Abrupt", "Gradual"],
            horizontal=True
        )
 
    # ── Symptom description ───────────────────────────────────────
    symptoms_desc = st.text_area(
        "Describe your symptoms in detail",
        placeholder="e.g. sharp chest pain radiating to left arm, shortness of breath, dizziness...",
        height=100
    )
 
    st.divider()
 
    # ── Section 3: Medical Context ────────────────────────────────
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
            default=["None"]
        )
 
    with col6:
        current_medications = st.text_input(
            "Current Medications (optional)",
            placeholder="e.g. Metformin, Aspirin, Lisinopril"
        )
 
        allergies = st.text_input(
            "Known Allergies (optional)",
            placeholder="e.g. Penicillin, Sulfa drugs, Latex"
        )
 
    st.divider()
 
    # ── Analyze Button ────────────────────────────────────────────
    if st.button("🔍 Analyze Symptoms", use_container_width=True):
 
        # Validation
        if body_part == "Select...":
            st.warning("Please select the body part affected.")
        elif not symptoms_desc.strip():
            st.warning("Please describe your symptoms in the text box.")
        else:
            # ── Build rich symptom string for the pipeline ────────
            # This gives the LLM far more structured context than free text alone
            conditions_str = (
                ", ".join([c for c in known_conditions if c != "None"])
                or "None reported"
            )
 
            symptoms = f"""
Body Part Affected: {body_part}
Symptom Description: {symptoms_desc.strip()}
Duration: {duration}
Onset: {onset_type}
Pain/Discomfort Level: {severity_slider}/10
Known Conditions: {conditions_str}
Current Medications: {current_medications.strip() or "None reported"}
Allergies: {allergies.strip() or "None reported"}
""".strip()
 
            patient_context = f"Age: {age}, Gender: {gender}"
 
            # ── Run pipeline with progress steps ─────────────────
            progress_bar = st.progress(0)
            status       = st.empty()
 
            status.markdown("🔍 **Agent 1/3:** Validating and normalising intake...")
            progress_bar.progress(10)
 
            import time
            result = run_triage_pipeline(symptoms, patient_context)
 
            progress_bar.progress(70)
            status.markdown("📋 **Agent 3/3:** Generating recommendations...")
            time.sleep(0.3)
            progress_bar.progress(100)
            time.sleep(0.2)
            progress_bar.empty()
            status.empty()
 
            # ── Invalid input ─────────────────────────────────────
            if not result["valid"]:
                st.error(f"⚠️ Could not process input: {result['invalid_reason']}")
                st.info("Please describe your symptoms more specifically.")
 
            else:
                severity   = result["severity"]
                department = result["department"]
                urgency    = result["urgency_score"]
 
                # ── Severity banner ───────────────────────────────
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
 
                # Urgency progress bar
                st.progress(urgency / 10)
 
                if severity == "Critical":
                    st.error("🚨 IMMEDIATE MEDICAL ATTENTION REQUIRED — Go to Emergency now")
 
                # ── Triage reasoning ──────────────────────────────
                with st.expander("🧠 View AI Triage Reasoning"):
                    st.info(result["triage_reasoning"])
                    st.caption("Structured Intake Sent to Triage Agent:")
                    st.code(result["intake"], language=None)
 
                # ── Patient summary ───────────────────────────────
                st.subheader("🩺 Patient Summary")
                st.info(f"""
**Patient:** {patient_name}  |  **Age:** {age}  |  **Gender:** {gender}  |  **Phone:** {phone}
 
**Body Area:** {body_part}  |  **Duration:** {duration}  |  **Pain Level:** {severity_slider}/10  |  **Onset:** {onset_type}
 
**Symptoms:** {symptoms_desc}
 
**Known Conditions:** {conditions_str}
 
**AI Assessment:** {result['summary']}
                """)
 
                st.caption(f"🕒 Analysis generated on {datetime.now().strftime('%d-%m-%Y %H:%M')}")
 
                # ── Recommended actions ───────────────────────────
                st.subheader("📋 Recommended Actions")
                for i, action in enumerate(result["actions"], 1):
                    st.markdown(f"{i}. {action}")
 
                # ── Emergency warning ─────────────────────────────
                if result["warning"]:
                    st.error(f"⚠️ **When to go to Emergency immediately:** {result['warning']}")
 
                # ── Save to DB ────────────────────────────────────
                save_case_to_db(
                        symptoms=f"{body_part}: {symptoms_desc}",
                        severity=severity,
                        department=department
                )
 
                # ── Download report ───────────────────────────────
                report = f"""MediAgent AI — Patient Report
Generated: {datetime.now().strftime('%d-%m-%Y %H:%M')}
 
PATIENT DETAILS
Name     : {patient_name}
Age      : {age}
Gender   : {gender}
Phone    : {phone}
 
SYMPTOM INTAKE
Body Part    : {body_part}
Description  : {symptoms_desc}
Duration     : {duration}
Onset        : {onset_type}
Pain Level   : {severity_slider}/10
Conditions   : {conditions_str}
Medications  : {current_medications or "None"}
Allergies    : {allergies or "None"}
 
TRIAGE RESULT
Severity     : {severity}
Department   : {department}
Urgency      : {urgency}/10
 
TRIAGE REASONING
{result['triage_reasoning']}
 
AI ASSESSMENT
{result['summary']}
 
RECOMMENDED ACTIONS
{chr(10).join(f"{i+1}. {a}" for i, a in enumerate(result['actions']))}
 
{"EMERGENCY WARNING: " + result['warning'] if result['warning'] else ""}
"""
                st.download_button(
                    "📄 Download Full Report",
                    report,
                    file_name=f"report_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%d%m%Y')}.txt"
                )

# -------------------------
# TAB 2 - HISTORY
# -------------------------
with tab2:

    st.header("📋 Patient Case History")
    if st.button("🗑 Clear All Cases"):
        conn = sqlite3.connect("data/hospital.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cases")
        conn.commit()
        conn.close()
        st.success("All case records deleted successfully!")
        st.rerun()

    try:

        conn = sqlite3.connect("data/hospital.db")
        df = pd.read_sql_query(
            "SELECT * FROM cases ORDER BY created_at DESC",
            conn
        )

        display_df = df[
        ["symptoms", "severity", "department", "created_at"]
        ]

        display_df.columns = [
            "Symptoms",
            "Severity",
            "Department",
            "Date & Time"
        ]
        display_df["Severity"] = display_df["Severity"].replace({
            "Critical": "🚨 Critical",
            "Moderate": "⚠️ Moderate",
            "Low": "✅ Low",
            "Mild": "✅ Mild"
        })

        st.dataframe(
        display_df,
        use_container_width=True
        )

        conn.close()

    except Exception as e:
        st.error(str(e))

# -------------------------
# TAB 3 - DASHBOARD

 
with tab3:

    st.header("📊 Real-Time Hospital Analytics")
    conn = sqlite3.connect("data/hospital.db")

    df = pd.read_sql_query(
        "SELECT * FROM cases",
        conn
    )

    total_cases = len(df)

    critical_cases = len(
        df[df["severity"].str.contains("Critical", case=False, na=False)]
    )

    moderate_cases = len(
        df[df["severity"].str.contains("Moderate", case=False, na=False)]
    )

    mild_cases = len(df) - critical_cases - moderate_cases

    if total_cases > 0:
        critical_percent = round(
            (critical_cases / total_cases) * 100,
            1
        )
    else:
        critical_percent = 0
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
        st.metric("📊 Critical %", f"{critical_percent}%")
        
    dept_df = pd.read_sql_query("""
    SELECT department, COUNT(*) as total
    FROM cases
    GROUP BY department
    """, conn)

    title="Cases by Department"
    fig = px.bar(
        dept_df,
        x="department",
        y="total",
        text="total"
    )
    fig.update_layout(
        paper_bgcolor="#F5EFE6",
        plot_bgcolor="#F5EFE6",
        font=dict(color="#34495E", size=14),
         

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

    st.plotly_chart(fig, use_container_width=True)

    sev_df = pd.read_sql_query("""
    SELECT severity, COUNT(*) as total
    FROM cases
    GROUP BY severity
    """, conn)

    st.subheader("Severity Distribution")

    fig2 = px.pie(
        sev_df,
        names="severity",
        values="total"
    )
    fig2.update_layout(
        paper_bgcolor="#F5EFE6",
        plot_bgcolor="#F5EFE6",
        font=dict(color="#34495E", size=14)
    )



    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(
        "<div style='text-align:center;color:#8B7355;padding-top:20px;'>Designed & Developed by Aarya Shirsath</div>",
        unsafe_allow_html=True
    )

    conn.close()
    
with tab4:

    st.header("👨‍⚕️ Doctor Portal")

    conn = sqlite3.connect("data/hospital.db")

    doctor_df = pd.read_sql_query(
        """
        SELECT *
        FROM cases
        ORDER BY
            CASE
                WHEN severity='Critical' THEN 1
                WHEN severity='Moderate' THEN 2
                ELSE 3
            END,
            created_at DESC
        """,
        conn
    )

    critical_count = len(
        doctor_df[doctor_df["severity"] == "Critical"]
    )

    st.error(f"🚨 Critical Cases Pending: {critical_count}")

    display_df = doctor_df[
        ["symptoms", "severity", "department", "created_at"]
    ]
    display_df.columns = [
        "Symptoms",
        "Severity",
        "Department",
        "Date & Time"
    ]

    def highlight_severity(row):
        if row["Severity"] == "Critical":
            return ["background-color: #FADADD; color: #7A1F1F"] * len(row)
        elif row["Severity"] == "Moderate":
            return ["background-color: #F8E6C1; color: #7A5C00"] * len(row)
        else:
            return ["background-color: #DCEFD8; color: #1F5D2E"] * len(row)

    styled_df = display_df.style.apply(highlight_severity, axis=1)

    st.dataframe(
       styled_df,
       use_container_width=True
    )
    
    
    conn.close()
    st.markdown("---")
    st.caption("🏥 MediAgent AI • Agentic Hospital Triage & Decision Support System")

# ── Drug checker LLM (reuses your Groq key) ──────────────────────
_drug_llm = ChatGroq(
    model="llama3-8b-8192",
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
PATIENT_ADVICE: <1-2 sentences on what the patient should do — e.g. consult doctor, avoid combination, monitor symptoms.>
 
If there is no interaction data found, say:
SEVERITY: Unknown
PLAIN_SUMMARY: No significant interaction data was found in the OpenFDA database for this drug combination.
MECHANISM: Not available.
PATIENT_ADVICE: Always consult your doctor or pharmacist before combining medications."""),
    ("human", """Drug 1: {drug1}
Drug 2: {drug2}
OpenFDA Data Summary: {fda_data}""")
])
 
_drug_chain = _drug_prompt | _drug_llm | StrOutputParser()
 
 
def _query_openfda(drug1: str, drug2: str) -> dict:
    """
    Queries OpenFDA drug adverse events API for co-reported events.
    No API key required.
    Returns dict with found: bool, count: int, reactions: list, raw: dict
    """
    base = "https://api.fda.gov/drug/event.json"
    query = f'patient.drug.medicinalproduct:"{drug1}"+AND+patient.drug.medicinalproduct:"{drug2}"'
 
    try:
        resp = requests.get(
            base,
            params={"search": query, "limit": 5},
            timeout=8
        )
 
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            total   = data.get("meta", {}).get("results", {}).get("total", 0)
 
            # Extract top reactions from results
            reactions = set()
            for result in results:
                for rxn in result.get("patient", {}).get("reaction", []):
                    rt = rxn.get("reactionmeddrapt", "")
                    if rt:
                        reactions.add(rt.title())
 
            return {
                "found":     total > 0,
                "count":     total,
                "reactions": list(reactions)[:10],
                "raw":       data,
            }
 
        elif resp.status_code == 404:
            return {"found": False, "count": 0, "reactions": [], "raw": {}}
 
        else:
            return {"found": False, "count": 0, "reactions": [], "raw": {}, "error": f"API status {resp.status_code}"}
 
    except requests.exceptions.Timeout:
        return {"found": False, "count": 0, "reactions": [], "raw": {}, "error": "OpenFDA API timed out"}
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
        "Results are sourced live from **OpenFDA** — a real pharmacological database. "
        "Zero AI hallucinations on interaction data."
    )
    st.caption("ℹ️ Source: U.S. Food & Drug Administration (api.fda.gov) · No API key required")
 
    st.divider()
 
    d_col1, d_col2 = st.columns(2)
 
    with d_col1:
        drug1 = st.text_input(
            "💊 Drug 1",
            placeholder="e.g. Aspirin",
        )
 
    with d_col2:
        drug2 = st.text_input(
            "💊 Drug 2",
            placeholder="e.g. Warfarin",
        )
 
    # Quick example buttons
      
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
 
            # ── API error ─────────────────────────────────────────
            if "error" in fda_result:
                st.error(f"OpenFDA API error: {fda_result['error']}")
 
            else:
                # ── Build FDA summary for LLM ─────────────────────
                if fda_result["found"]:
                    fda_summary = (
                        f"Found {fda_result['count']:,} adverse event reports "
                        f"where {drug1} and {drug2} were taken together. "
                        f"Top reported reactions: {', '.join(fda_result['reactions']) if fda_result['reactions'] else 'not specified'}."
                    )
                else:
                    fda_summary = f"No adverse event co-reports found in OpenFDA for {drug1} and {drug2}."
 
                # ── LLM explanation ───────────────────────────────
                with st.spinner("Generating clinical explanation..."):
                    llm_output = _drug_chain.invoke({
                        "drug1":    drug1.strip(),
                        "drug2":    drug2.strip(),
                        "fda_data": fda_summary,
                    })
 
                severity_label  = _parse_drug_field(llm_output, "SEVERITY")
                plain_summary   = _parse_drug_field(llm_output, "PLAIN_SUMMARY")
                mechanism       = _parse_drug_field(llm_output, "MECHANISM")
                patient_advice  = _parse_drug_field(llm_output, "PATIENT_ADVICE")
 
                # ── Severity badge ────────────────────────────────
                st.markdown("### 📊 Interaction Result")
 
                sev_lower = severity_label.lower()
                if "major" in sev_lower:
                    st.error(f"🔴 **Severity: {severity_label}** — Significant risk. Consult your doctor immediately.")
                elif "moderate" in sev_lower:
                    st.warning(f"🟡 **Severity: {severity_label}** — Use with caution. Doctor consultation advised.")
                elif "minor" in sev_lower:
                    st.success(f"🟢 **Severity: {severity_label}** — Low risk. Monitor for any unusual symptoms.")
                else:
                    st.info(f"⚪ **Severity: {severity_label}** — Insufficient data to assess risk.")
 
                # ── FDA data card ─────────────────────────────────
                st.markdown(f"**OpenFDA Reports Found:** {fda_result['count']:,}")
                if fda_result["reactions"]:
                    st.markdown("**Top Reported Adverse Reactions:**")
                    rxn_cols = st.columns(min(len(fda_result["reactions"]), 5))
                    for i, rxn in enumerate(fda_result["reactions"][:5]):
                        rxn_cols[i % 5].markdown(f"• {rxn}")
 
                st.divider()
 
                # ── Clinical explanation ──────────────────────────
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
 
                # ── Source attribution ────────────────────────────
                st.divider()
                st.caption(
                    "📌 Interaction data sourced from OpenFDA Adverse Event Reporting System (FAERS). "
                    "Clinical explanation generated by Groq LLaMA3. "
                    "This tool is for informational purposes only — always consult a licensed pharmacist or physician."
                )