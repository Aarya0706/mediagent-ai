import streamlit as st
import sqlite3
import pandas as pd
import subprocess
import plotly.express as px
from datetime import datetime
 
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

tab1, tab2, tab3, tab4 = st.tabs([
    "Patient Triage",
    "Case History",
    "Dashboard",
    "Doctor Portal"
])

# -------------------------
# TAB 1 - TRIAGE
# -------------------------
with tab1:

    st.header("🩺 Patient Symptom Analysis")
    patient_name = st.text_input("Patient Name")

    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"]
    )

    phone = st.text_input("Phone Number")

    symptoms = st.text_area(
        "Enter Patient Symptoms",
        height=150
    )

    if st.button("Analyze Symptoms"): 
        

        if symptoms.strip() == "":
            st.warning("Please enter symptoms.")

        else:

            import subprocess

            with st.spinner("Analyzing symptoms..."):

                result = subprocess.run(
                    ["python", "agents/orchestrator.py"],
                    input=f"{symptoms}\nexit\n",
                    text=True,
                    capture_output=True
                )

            st.success("Analysis Complete")
            output = result.stdout

            output = output.replace("Patient Symptoms (type exit to quit):", "")
            output = output.replace("AI RESPONSE:", "")
            output = output.strip()

            response = output.lower()
            st.write(output)
            department = "General Medicine"

            if "cardiology" in response:
                department = "Cardiology"
            elif "neurology" in response:
                department = "Neurology"
            elif "orthopedic" in response:
                department = "Orthopedics"
            elif "emergency" in response:
                department = "Emergency"
            severity_level = "Mild"
            

            severity_level = "Mild"

            if "critical" in response:
                severity_level = "Critical"
            elif "moderate" in response:
                severity_level = "Moderate"
             

            if severity_level == "Critical":
                st.error("🔴 Severity: Critical")
                st.error("🚨 IMMEDIATE MEDICAL ATTENTION REQUIRED")

            elif severity_level == "Moderate":
                st.warning("🟡 Severity: Moderate")
 
            else:
                st.success("🟢 Severity: Mild")

            st.success(f"🏥 Recommended Department: {department}")
            st.subheader("🩺 Patient Summary")

            st.info(f"""
            Patient Name: {patient_name}

            Age: {age}

            Gender: {gender}

            Phone: {phone}

            Symptoms Reported:

            {symptoms}
            """)
            

             
        
            st.caption(
                f"🕒 Analysis generated on {datetime.now().strftime('%d-%m-%Y %H:%M')}"
            )
             
            st.subheader("📋 Recommended Actions")
            symptom_text = symptoms.lower()

            if "fever" in symptom_text:
                actions = """
            1. Take adequate rest
            2. Stay hydrated
            3. Monitor temperature
            4. Consult doctor if fever persists
            """

            elif "headache" in symptom_text:
                actions = """
            1. Rest in a quiet environment
            2. Drink water regularly
            3. Avoid screen exposure
            4. Seek medical advice if severe
            """

            elif "chest pain" in symptom_text:
                actions = """
            1. Seek immediate medical attention
            2. Avoid physical exertion
            3. Call emergency services if severe
            4. Visit nearest emergency department
            """

            elif "vomiting" in symptom_text or "nausea" in symptom_text:
                actions = """
            1. Drink oral rehydration fluids
            2. Avoid heavy meals
            3. Monitor dehydration signs
            4. Consult doctor if symptoms continue
            """

            else:
                actions = """
            1. Rest adequately
            2. Stay hydrated
            3. Monitor symptoms
            4. Consult doctor if symptoms persist
            """
            st.success(actions)

             
            report = f"""
            Symptoms:
            {symptoms}

            Analysis:
            {output}

            Recommended Actions:
            {actions}
            """

            st.download_button(
                "📄 Download Report",
                report,
                file_name="patient_report.txt"
            )

# -------------------------
# TAB 2 - HISTORY
# -------------------------
with tab2:

    st.header("📋 Patient Case History")

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