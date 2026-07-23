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
from textwrap import dedent

import sys
import os
from tools.save_case import save_case_to_db, update_case_status, get_all_cases, delete_case, clear_all_cases, update_case_notes
from tools.lab_report_tools import (
    ExtractionError,
    extract_text_from_file,
    parse_lab_report_with_llm,
    save_lab_report,
    get_reports_for_patient,
    get_values_for_report,
    get_known_parameters,
    get_trend_data,
    get_all_patient_names,
)
from tools.appointment_prep_tools import (
    get_recent_cases,
    get_recent_lab_reports,
    get_lab_values_for_reports,
    has_any_history,
    get_all_patients_with_history,
    generate_appointment_prep,
)
from tools.chat_rag_tools import (
    get_patient_chunks,
    has_any_chunks,
    retrieve_relevant_chunks,
    get_relevant_context,
    generate_chat_answer,
)
from tools.health_profile_tools import (
    get_profile,
    upsert_profile,
    get_all_profiled_patients,
)
from tools.auth_tools import (
    username_exists,
    create_user,
    verify_user,
    get_user_count,
    ensure_demo_user,
    ensure_demo_patient_user,
)
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

/* Tighter spacing overall - Streamlit's default gap between stacked
   elements/columns is generous and compounds fast on pages with many
   small widgets (metrics, labels, buttons), which is what was making
   Case History feel scattered. */
[data-testid="stVerticalBlock"] { gap: 0.5rem !important; }
[data-testid="stHorizontalBlock"] { gap: 0.75rem !important; }
[data-testid="stMetric"] { padding: 2px 0 !important; }
[data-testid="stMetricValue"] { font-size: 1.5rem !important; }

/* st.divider() renders an <hr> with its own margin on top of the flex
   gap above, which was stacking to create a much bigger visible gap
   than intended around every divider on the page. */
hr { margin: 0.5rem 0 !important; }

/* Bordered containers (st.container(border=True)) - used for each patient
   case card - get a real elevated-card treatment with generous spacing
   below, so consecutive cards read as clearly separate units against the
   page's gradient background instead of blending into one continuous list. */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(0,0,0,0.06) !important;
    border-radius: 18px !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.07) !important;
    padding: 6px 4px !important;
    margin-bottom: 24px !important;
    background: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

title_col, user_col = st.columns([5, 1])
with title_col:
    st.markdown("# 🏥 MediAgent AI")
with user_col:
    if st.session_state.get("auth_user"):
        _role_label = "Patient" if st.session_state.get("auth_role") == "patient" else "Staff"
        st.markdown(
            f'<div style="text-align:right; margin-top:22px; margin-bottom:8px; font-size:13px; color:#7F8C8D;">'
            f'👤 {st.session_state["auth_user"]} <span style="opacity:0.7;">({_role_label})</span></div>',
            unsafe_allow_html=True,
        )
        if st.button("🚪 Log Out", key="top_logout_btn"):
            st.session_state["auth_user"] = None
            st.session_state["auth_role"] = None
            st.session_state["auth_patient_name"] = None
            st.rerun()

st.markdown("""
### Agentic Hospital Triage & Decision Support System
AI-powered emergency assessment, department routing, drug interaction checking, lab report analysis,
pre-appointment prep, a personal health chat, and patient analytics - all in one place.
""")


# ============================================================
# LOGIN GATE
# ============================================================
#
# Streamlit-appropriate reinterpretation of the PRD's auth requirement:
# one login per browser session (st.session_state), not a stateless
# JWT-consuming frontend. Nothing below this block renders - no tabs,
# no DB reads beyond the users table itself - until
# st.session_state["auth_user"] is set, so this is a real gate on the
# whole app, not just a banner on top of it.
#
# Two account roles:
#   - "staff"   - can look up and act on any patient (doctors, nurses,
#                 admins). This is what the app was originally built for.
#   - "patient" - locked at signup to exactly one patient_name, which is
#                 the only identity this session is ever allowed to see.
#                 Tabs that show cross-patient data (Case History,
#                 Dashboard, Doctor Portal) are hidden for this role, and
#                 Patient Triage uses the locked name instead of letting
#                 the session type in any name.

for _key, _default in [("auth_user", None), ("auth_role", None), ("auth_patient_name", None)]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

if not st.session_state["auth_user"]:

    st.divider()

    # One-click guest access - anyone evaluating the app (recruiters,
    # interviewers) shouldn't have to register a real account just to
    # look around, for either role. Auth still gates the app for
    # everyone else; this is a deliberate, visible bypass rather than a
    # security hole.
    demo_col1, demo_col2, _spacer = st.columns([2, 2, 2])
    with demo_col1:
        if st.button("🚀 Try Demo (Staff view)", key="demo_staff_btn", width="stretch"):
            demo_info = ensure_demo_user()
            st.session_state["auth_user"] = demo_info["display_name"]
            st.session_state["auth_role"] = demo_info["role"]
            st.session_state["auth_patient_name"] = demo_info["patient_name"]
            st.rerun()
    with demo_col2:
        if st.button("🙋 Try Demo (Patient view)", key="demo_patient_btn", width="stretch"):
            demo_info = ensure_demo_patient_user()
            st.session_state["auth_user"] = demo_info["display_name"]
            st.session_state["auth_role"] = demo_info["role"]
            st.session_state["auth_patient_name"] = demo_info["patient_name"]
            st.rerun()
    st.caption("Skips registration entirely - explore the app as a shared demo account, from either side.")

    login_tab, signup_tab = st.tabs(["🔐 Log In", "🆕 Create Account"])

    with login_tab:
        st.subheader("Log In")

        if get_user_count() == 0:
            st.info(
                "No accounts exist yet - use **Create Account** to set up the first login."
            )

        with st.form("login_form"):
            login_username = st.text_input("Username")
            login_password = st.text_input("Password", type="password")
            login_submitted = st.form_submit_button("Log In", width="stretch")

        if login_submitted:
            user_info = verify_user(login_username, login_password)
            if user_info:
                st.session_state["auth_user"] = user_info["display_name"]
                st.session_state["auth_role"] = user_info["role"]
                st.session_state["auth_patient_name"] = user_info["patient_name"]
                st.rerun()
            else:
                st.error("Incorrect username or password.")

    with signup_tab:
        st.subheader("Create an Account")
        st.caption(
            "No hospital admin approval flow in this version - anyone with app access "
            "can self-register. Intended for demo/internal use, not a public deployment."
        )

        signup_role = st.radio(
            "I am a...",
            ["Hospital Staff (doctor / nurse / admin)", "Patient"],
            key="signup_role_choice",
            horizontal=True,
        )
        _is_patient_signup = signup_role == "Patient"

        with st.form("signup_form"):
            signup_username = st.text_input("Choose a username", key="signup_username")
            signup_display_name = st.text_input(
                "Display name (optional)",
                key="signup_display_name",
                placeholder="e.g. Dr. Mehta" if not _is_patient_signup else "e.g. Rohan Sharma",
            )

            signup_patient_name = ""
            if _is_patient_signup:
                signup_patient_name = st.text_input(
                    "Your full name, exactly as hospital staff have it on file",
                    key="signup_patient_name",
                    placeholder="e.g. Rohan Sharma",
                    help=(
                        "This locks your account to that patient identity - you will only "
                        "ever see records under this exact name, and it can't be changed "
                        "later from this screen."
                    ),
                )

            signup_password = st.text_input(
                "Choose a password",
                type="password",
                key="signup_password",
                help="At least 6 characters.",
            )
            signup_password_confirm = st.text_input(
                "Confirm password",
                type="password",
                key="signup_password_confirm",
            )
            signup_submitted = st.form_submit_button("Create Account", width="stretch")

        if signup_submitted:
            if signup_password != signup_password_confirm:
                st.error("Passwords don't match.")
            elif _is_patient_signup and not signup_patient_name.strip():
                st.error("Please enter your full name as used in your hospital records.")
            else:
                try:
                    create_user(
                        signup_username,
                        signup_password,
                        display_name=signup_display_name.strip() or None,
                        role="patient" if _is_patient_signup else "staff",
                        patient_name=signup_patient_name.strip() if _is_patient_signup else None,
                    )
                    st.success("Account created - you can log in now on the Log In tab.")
                except ValueError as e:
                    st.error(str(e))

    st.stop()


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


# ============================================================
# ROLE-AWARE LANDING SCREEN
# ============================================================
#
# Streamlit's st.tabs() always opens on the first tab and has no way to
# set a different default - so instead of fighting that, this renders a
# role-specific summary ABOVE the tabs, right after login. Staff land on
# an operational snapshot (today's hospital state); a patient lands on
# a summary of their own records. The tabs below are unchanged and still
# work exactly as before for anyone who wants to navigate directly.

def render_landing_stat_cards(items):
    """Render a row of small bordered stat cards (icon, label, value, accent
    color) instead of bare st.metric() calls. st.metric() on its own has no
    background or grouping, so a row of them reads as loose numbers floating
    on the page; wrapping each one in a white card with a colored top border
    ties them together as a single glanceable panel."""
    # IMPORTANT: this HTML must stay on one line with no leading whitespace.
    # st.markdown runs its content through a Markdown parser before handing
    # raw-HTML blocks through - a 4+ space indent or a blank line inside the
    # string gets read as a Markdown "indented code block", which is exactly
    # what happened before: only the first <div> rendered as HTML, and
    # everything after the first blank line printed out as literal text.
    card_template = (
        '<div style="background:#FFFFFF;border-radius:14px;padding:14px 16px;'
        'border-top:4px solid {color};box-shadow:0 3px 10px rgba(0,0,0,.07);'
        'flex:1 1 150px;min-width:150px;">'
        '<div style="font-size:12px;font-weight:700;color:#7F8C8D;letter-spacing:.3px;">{icon} {label}</div>'
        '<div style="margin-top:6px;font-size:21px;font-weight:800;color:#2C3E50;">{value}</div>'
        '</div>'
    )
    cards_html = "".join(
        card_template.format(color=color, icon=icon, label=label.upper(), value=value)
        for icon, label, value, color in items
    )
    st.markdown(
        f'<div style="display:flex;gap:12px;flex-wrap:wrap;margin:10px 0 16px 0;">{cards_html}</div>',
        unsafe_allow_html=True,
    )


_landing_role = st.session_state.get("auth_role")

with st.container(border=True):
    if _landing_role == "patient":
        _own_name = st.session_state.get("auth_patient_name", "")
        st.markdown(f"#### 👋 Welcome back, {_own_name}")

        _own_cases = get_recent_cases(_own_name, limit=1)
        _own_reports = get_recent_lab_reports(_own_name, limit=1)
        _own_profile = get_profile(_own_name)

        if not _own_cases and not _own_reports:
            st.caption(
                "No records on file yet. Start with **Patient Triage** to log your first "
                "symptom check, or **Lab Reports** to upload a report."
            )
        else:
            render_landing_stat_cards([
                (
                    "🩺",
                    "Last symptom check",
                    _own_cases[0]["created_at"].split(" ")[0] if _own_cases else "None yet",
                    "#4C6EF5",
                ),
                (
                    "🧪",
                    "Last lab report",
                    _own_reports[0]["created_at"].split(" ")[0] if _own_reports else "None yet",
                    "#12B886",
                ),
                (
                    "🗂️",
                    "Health profile",
                    "On file" if _own_profile else "Not set up",
                    "#A67C52",
                ),
            ])
            st.caption(
                "Head to **Appointment Prep** to get ready for your next visit, or "
                "**AI Health Chat** to ask about your own records."
            )

    else:
        _staff_name = st.session_state.get("auth_user", "there")
        st.markdown(f"#### 👋 Welcome back, {_staff_name}")

        try:
            with sqlite3.connect(DB_PATH) as conn:
                _landing_df = pd.read_sql_query("SELECT severity, status FROM cases", conn)

            if _landing_df.empty:
                st.caption("No cases on file yet. Head to **Patient Triage** to log the first one.")
            else:
                _total = len(_landing_df)
                _critical_open = len(
                    _landing_df[(_landing_df["severity"] == "Critical") & (_landing_df["status"] != "Resolved")]
                )
                _pending = len(_landing_df[_landing_df["status"] == "Pending"])
                _resolved = len(_landing_df[_landing_df["status"] == "Resolved"])

                render_landing_stat_cards([
                    ("📁", "Total cases", _total, "#4C6EF5"),
                    ("🚨", "Critical (open)", _critical_open, "#E74C3C"),
                    ("⏳", "Pending", _pending, "#F59F00"),
                    ("✅", "Resolved", _resolved, "#37B24D"),
                ])

                if _critical_open > 0:
                    st.caption(f"⚠️ {_critical_open} critical case(s) need attention - see **Doctor Portal**.")
                else:
                    st.caption("No open critical cases right now. See **Dashboard** for full analytics.")
        except Exception:
            st.caption("Use the tabs below to get started.")

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "🩺 Patient Triage", "📋 Case History", "📊 Dashboard", "👨‍⚕️ Doctor Portal", "💊 Drug Checker", "🧪 Lab Reports", "🗓️ Appointment Prep", "💬 AI Health Chat", "🗂️ Health Profile",
])


# ── TAB 1 ─────────────────────────────────────────────────────────
with tab1:
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
                        st.exception(e)
# ── TAB 2 ─────────────────────────────────────────────────────────
with tab2:
    if st.session_state.get("auth_role") == "patient":
        st.warning("This view is for hospital staff only. Patients don't have access to the full case archive.")
    else:
        st.header("📋 Patient Case History")

        # ==========================================================
        # SEARCH + FILTER + SORT CONTROLS
        # ==========================================================

        search_query = st.text_input(
            "🔍 Search Cases",
            placeholder="Search by patient name, symptoms, or department...",
            key="case_history_search"
        )

        filt_col1, filt_col2 = st.columns(2, gap="small")

        with filt_col1:
            severity_filter = st.selectbox(
                "Filter by Severity",
                ["All", "Critical", "Moderate", "Mild"],
                key="case_severity_filter",
            )

        with filt_col2:
            sort_choice = st.selectbox(
                "Sort by",
                ["Newest first", "Oldest first"],
                key="case_sort_choice",
            )

        # ==========================================================
        # CLEAR ALL - two-step confirmation so one misclick can't wipe
        # every case in the database.
        # ==========================================================

        if st.button("🗑 Clear All Cases", key="clear_all_cases"):
            st.session_state["confirm_clear_all_cases"] = True

        if st.session_state.get("confirm_clear_all_cases"):
            st.warning(
                "This permanently deletes **every** case record for **every** "
                "patient. This cannot be undone."
            )
            cc1, cc2 = st.columns(2)
            with cc1:
                if st.button(
                    "✅ Yes, delete everything",
                    key="confirm_clear_all_yes",
                    width="stretch",
                ):
                    clear_all_cases()
                    st.session_state["confirm_clear_all_cases"] = False
                    st.success("All case records deleted successfully!")
                    st.rerun()
            with cc2:
                if st.button(
                    "Cancel",
                    key="confirm_clear_all_no",
                    width="stretch",
                ):
                    st.session_state["confirm_clear_all_cases"] = False
                    st.rerun()

        st.divider()

        # ==========================================================
        # LOAD, FILTER, SORT
        # ==========================================================

        try:
            all_cases = get_all_cases()
        except Exception as e:
            all_cases = []
            st.error(f"Could not load case history: {e}")

        filtered_cases = all_cases

        if search_query.strip():
            q = search_query.strip().lower()
            filtered_cases = [
                c for c in filtered_cases
                if q in (c.get("patient_name") or "").lower()
                or q in (c.get("symptoms") or "").lower()
                or q in (c.get("department") or "").lower()
            ]

        if severity_filter != "All":
            filtered_cases = [
                c for c in filtered_cases
                if (c.get("severity") or "").strip().lower() == severity_filter.lower()
            ]

        # get_all_cases() already returns newest-first (created_at DESC, id DESC)
        if sort_choice == "Oldest first":
            filtered_cases = list(reversed(filtered_cases))

        # ==========================================================
        # SUMMARY METRICS FOR THE CURRENT (FILTERED) VIEW
        # ==========================================================

        if all_cases:
            m1, m2, m3, m4 = st.columns(4, gap="small")
            with m1:
                st.metric("Showing", len(filtered_cases))
            with m2:
                st.metric(
                    "🚨 Critical",
                    sum(1 for c in filtered_cases if (c.get("severity") or "").lower() == "critical"),
                )
            with m3:
                st.metric(
                    "⚠️ Moderate",
                    sum(1 for c in filtered_cases if (c.get("severity") or "").lower() == "moderate"),
                )
            with m4:
                st.metric(
                    "✅ Mild",
                    sum(1 for c in filtered_cases if (c.get("severity") or "").lower() == "mild"),
                )

            st.divider()

        # ==========================================================
        # CASE LIST
        # ==========================================================

        if not all_cases:
            st.info("No patient cases available yet.")
        elif not filtered_cases:
            st.info("No matching cases found.")
        else:
            severity_icon_map = {"critical": "🔴", "moderate": "🟡", "mild": "🟢"}
            status_icon_map = {"Pending": "⏳", "In Progress": "🩺", "Resolved": "✅"}

            for case in filtered_cases:
                case_id = case["id"]
                severity_label = (case.get("severity") or "Unknown").strip()
                status_label = case.get("status") or "Pending"
                sev_icon = severity_icon_map.get(severity_label.lower(), "⚪")
                status_icon = status_icon_map.get(status_label, "⏳")

                severity_colors = {
                    "critical": {"bg": "#FDEDEC", "border": "#E74C3C", "text": "#C0392B"},
                    "moderate": {"bg": "#FBF2E3", "border": "#A67C52", "text": "#8B6A45"},
                    "mild":     {"bg": "#EAF6EE", "border": "#4C9A6A", "text": "#2F7A4D"},
                }
                colors = severity_colors.get(severity_label.lower(), {"bg": "#F1EEE8", "border": "#A6A6A6", "text": "#5C5C5C"})

                with st.container(border=True):
                    st.markdown(
                        f"""<div style="background:{colors['bg']}; border-left:5px solid {colors['border']};
                             border-radius:10px; padding:10px 16px; margin-bottom:6px;">
                            <div style="font-size:16px; font-weight:700; color:#2C3E50;">
                                {sev_icon} {case.get('patient_name', 'Unknown')}
                                <span style="font-weight:400; color:#7F8C8D; font-size:13px;">
                                    &nbsp;&nbsp;·&nbsp;&nbsp;Case #{case_id}&nbsp;&nbsp;·&nbsp;&nbsp;{case.get('created_at', '')}
                                </span>
                            </div>
                            <div style="margin-top:4px; font-size:14px; color:#34495E;">
                                <b>Severity:</b> <span style="color:{colors['text']}; font-weight:700;">{severity_label or 'Unknown'}</span>
                                &nbsp;&nbsp;&nbsp;&nbsp;<b>Department:</b> {case.get('department') or '—'}
                                &nbsp;&nbsp;&nbsp;&nbsp;<b>Status:</b> {status_icon} {status_label}
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                    st.markdown("#### 🩺 Symptoms")

                    st.info(case["symptoms"])

                    if case.get("summary") or case.get("recommendation"):
                        with st.expander(
                            "🧠 AI Assessment",
                            expanded=False,
                        ):
                            if case.get("summary"):
                                st.markdown("### 📝 Clinical Summary")
                                st.info(case["summary"])
                            if case.get("recommendation"):
                                st.markdown("### 💊 Recommended Action")
                                st.info(case["recommendation"])

                    # ------------------------------------------
                    # DELETE - two-step confirmation per case,
                    # scoped by case_id so confirming one case
                    # never accidentally triggers another's delete.
                    # ------------------------------------------

                    confirm_key = f"confirm_delete_case_{case_id}"

                    del_col, _spacer = st.columns([1, 4])
                    with del_col:
                        if not st.session_state.get(confirm_key):
                            if st.button("🗑 Delete", key=f"delete_case_{case_id}"):
                                st.session_state[confirm_key] = True
                                st.rerun()

                    if st.session_state.get(confirm_key):
                        st.warning(f"Delete case #{case_id} for {case.get('patient_name', 'this patient')}? This cannot be undone.")
                        dc1, dc2 = st.columns(2)
                        with dc1:
                            if st.button(
                                "✅ Yes, delete this case",
                                key=f"confirm_delete_yes_{case_id}",
                                width="stretch",
                            ):
                                delete_case(case_id)
                                st.session_state[confirm_key] = False
                                st.success(f"Case #{case_id} deleted.")
                                st.rerun()
                        with dc2:
                            if st.button(
                                "Cancel",
                                key=f"confirm_delete_no_{case_id}",
                                width="stretch",
                            ):
                                st.session_state[confirm_key] = False
                                st.rerun()


# ── TAB 3 ─────────────────────────────────────────────────────────
with tab3:
    if st.session_state.get("auth_role") == "patient":
        st.warning("This view is for hospital staff only. Patients don't have access to hospital-wide analytics.")
    else:
        st.header("📊 Real-Time Hospital Analytics")
        st.caption(
            "Operational overview across all patients and departments - status, overdue "
            "cases, staffing patterns, and case mix. For an individual patient's own doctor "
            "queue, see the Doctor Portal tab."
        )

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

            # ── Operational metrics: status + overdue + today's volume ──
            # These are the numbers that answer "is anything falling through
            # the cracks right now", which severity/department counts alone
            # don't tell you - two hospitals with identical severity mixes can
            # have very different amounts of actually-overdue, unattended work.

            pending_count = len(df[df["status"] == "Pending"]) if "status" in df.columns else 0
            in_progress_count = len(df[df["status"] == "In Progress"]) if "status" in df.columns else 0
            resolved_count = len(df[df["status"] == "Resolved"]) if "status" in df.columns else 0
            resolution_rate = round((resolved_count / total_cases) * 100, 1) if total_cases > 0 else 0

            OVERDUE_THRESHOLD_SECONDS = {
                "critical": 30 * 60,
                "moderate": 3 * 60 * 60,
                "mild": 24 * 60 * 60,
            }

            def _is_overdue(row):
                if row.get("status") == "Resolved":
                    return False
                try:
                    created = pd.to_datetime(row["created_at"])
                    elapsed = (pd.Timestamp.now() - created).total_seconds()
                except Exception:
                    return False
                threshold = OVERDUE_THRESHOLD_SECONDS.get(str(row.get("severity", "")).strip().lower(), 24 * 60 * 60)
                return elapsed > threshold

            overdue_count = int(df.apply(_is_overdue, axis=1).sum()) if not df.empty else 0

            today_count = 0
            if not df.empty:
                _created = pd.to_datetime(df["created_at"], errors="coerce")
                today_count = int((_created.dt.date == pd.Timestamp.now().date()).sum())

            if critical_cases > 10:
                st.error("🚨 Hospital Alert: High Emergency Load")
            elif overdue_count > 0:
                st.warning(f"⏱ {overdue_count} case(s) are overdue and still unresolved.")
            else:
                st.success("✅ Hospital Status Normal")

            render_landing_stat_cards([
                ("📁", "Cases", total_cases, "#4C6EF5"),
                ("🚨", "Critical", critical_cases, "#E74C3C"),
                ("⚠️", "Moderate", moderate_cases, "#A67C52"),
                ("✅", "Mild", mild_cases, "#37B24D"),
                ("📊", "Critical %", f"{critical_percent}%", "#7048E8"),
            ])

            render_landing_stat_cards([
                ("⏳", "Pending", pending_count, "#F59F00"),
                ("🩺", "In Progress", in_progress_count, "#7048E8"),
                ("✅", "Resolved", resolved_count, "#37B24D"),
                ("⏱", "Overdue", overdue_count, "#E74C3C"),
                ("🆕", "Today", today_count, "#4C6EF5"),
            ])


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


            # ── Department Distribution (by severity mix) ───────────────
            # A plain department count tells you volume; breaking each bar
            # down by severity tells you which departments are carrying the
            # riskiest load, not just the busiest one - a department with 10
            # Mild cases is a very different situation from one with 10
            # Critical cases, even though the plain bar chart would look
            # identical for both.

            st.subheader("🏥 Cases by Department")

            if dept_df.empty:

                st.info("No department data available.")

            else:
                dept_sev_df = (
                    df.groupby(["department", "severity"])
                    .size()
                    .reset_index(name="total")
                )

                SEVERITY_COLOR_MAP = {
                    "Critical": "#E74C3C",
                    "Moderate": "#A67C52",
                    "Mild": "#4C9A6A",
                }

                fig = px.bar(
                    dept_sev_df,
                    x="department",
                    y="total",
                    color="severity",
                    barmode="stack",
                    color_discrete_map=SEVERITY_COLOR_MAP,
                    text="total",
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
                    ),

                    legend_title_text="Severity",
                )

                fig.update_traces(textposition="inside")

                st.plotly_chart(
                    fig,
                    width="stretch"
                )


            # ── Cases by Hour of Day (staffing insight) ──────────────────
            # Which hours actually see the most patient volume - useful for
            # deciding when extra staff coverage matters most, something
            # neither the daily timeline nor the department chart shows.

            st.subheader("🕒 Cases by Hour of Day")

            if df.empty:
                st.info("No case data available for hourly analytics.")
            else:
                hour_df = df.copy()
                hour_df["created_at"] = pd.to_datetime(hour_df["created_at"], errors="coerce")
                hour_df = hour_df.dropna(subset=["created_at"])

                if hour_df.empty:
                    st.info("No valid case timestamps available.")
                else:
                    hour_df["Hour"] = hour_df["created_at"].dt.hour
                    hourly_counts = (
                        hour_df.groupby("Hour").size().reindex(range(24), fill_value=0).reset_index(name="Cases")
                    )
                    hourly_counts["Hour"] = hourly_counts["Hour"].apply(lambda h: f"{h:02d}:00")

                    fig_hour = px.bar(
                        hourly_counts,
                        x="Hour",
                        y="Cases",
                    )

                    fig_hour.update_layout(
                        paper_bgcolor="#F5EFE6",
                        plot_bgcolor="#F5EFE6",
                        font=dict(color="#34495E", size=13),
                        height=380,
                        margin=dict(l=50, r=20, t=20, b=70),
                        bargap=0.15,
                        xaxis=dict(
                            title="Hour of Day (IST)",
                            color="#34495E",
                            type="category",
                            # Force every one of the 24 hour labels to render,
                            # angled so they don't overlap - Plotly's default
                            # tick selection was dropping most of them when the
                            # chart was narrow, leaving the axis blank.
                            tickmode="array",
                            tickvals=hourly_counts["Hour"].tolist(),
                            tickangle=-45,
                            tickfont=dict(size=11),
                        ),
                        yaxis=dict(
                            title="Cases",
                            color="#34495E",
                            rangemode="tozero",
                            dtick=1,
                        ),
                    )
                    fig_hour.update_traces(marker_color="#7048E8")

                    st.plotly_chart(fig_hour, width="stretch")


            # ── Case Status Breakdown ─────────────────────────────────────
            # How much of the current caseload is still outstanding vs
            # actually resolved - the workload-progress view that severity
            # and department breakdowns don't answer on their own.

            st.subheader("🔄 Case Status Breakdown")

            if "status" not in df.columns or df.empty:
                st.info("No status data available.")
            else:
                status_df = (
                    df["status"].fillna("Pending").value_counts().reset_index()
                )
                status_df.columns = ["status", "total"]

                STATUS_COLOR_MAP = {
                    "Pending": "#F59F00",
                    "In Progress": "#7048E8",
                    "Resolved": "#37B24D",
                }

                fig_status = px.bar(
                    status_df,
                    x="status",
                    y="total",
                    color="status",
                    color_discrete_map=STATUS_COLOR_MAP,
                    text="total",
                )

                fig_status.update_layout(
                    paper_bgcolor="#F5EFE6",
                    plot_bgcolor="#F5EFE6",
                    font=dict(color="#34495E", size=14),
                    xaxis=dict(title="Status", color="#34495E"),
                    yaxis=dict(title="Cases", color="#34495E"),
                    showlegend=False,
                )
                fig_status.update_traces(textposition="outside")

                st.plotly_chart(fig_status, width="stretch")

                st.caption(f"Resolution rate: **{resolution_rate}%** of all cases on file are marked Resolved.")


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
     

    # ───────────────────────────────────────────────────────────────
    # TAB 4 - DOCTOR PORTAL
    # ───────────────────────────────────────────────────────────────
    with tab4:
        if st.session_state.get("auth_role") == "patient":
            st.warning("This view is for hospital staff only. Patients don't have a doctor work queue.")
        else:

            st.header("👨‍⚕️ Doctor Portal")
            st.caption(
                "Live clinical work queue prioritized by severity and urgency."
            )

            # ---------------------------------------------------------
            # LOAD CASES
            # ---------------------------------------------------------

            with sqlite3.connect(DB_PATH) as conn:
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
                        created_at ASC
                    """,
                    conn,
                )

            if doctor_df.empty:
                st.info("No patient cases available.")
                st.stop()

            # ---------------------------------------------------------
            # DASHBOARD SUMMARY
            # ---------------------------------------------------------

            total_cases = len(doctor_df)

            active_df = doctor_df[
                doctor_df["status"] != "Resolved"
            ]

            active_cases = len(active_df)

            critical_cases = len(
                active_df[
                    active_df["severity"] == "Critical"
                ]
            )

            pending_cases = len(
                active_df[
                    active_df["status"] == "Pending"
                ]
            )

            treating_cases = len(
                active_df[
                    active_df["status"] == "In Progress"
                ]
            )

            resolved_cases = len(
                doctor_df[
                    doctor_df["status"] == "Resolved"
                ]
            )

            st.subheader("📊 Dashboard Overview")

            cards = [
                ("📋", "Active", active_cases, "#3B82F6"),
                ("🔴", "Critical", critical_cases, "#EF4444"),
                ("⏳", "Pending", pending_cases, "#F59E0B"),
                ("🩺", "Treating", treating_cases, "#8B5CF6"),
                ("✅", "Resolved", resolved_cases, "#22C55E"),
            ]

            cols = st.columns(5)

            for col, (icon, title, value, color) in zip(cols, cards):
                with col:
                    st.html(f"""
            <div style="
            background:white;
            border-radius:16px;
            padding:18px;
            border-top:5px solid {color};
            box-shadow:0 4px 10px rgba(0,0,0,.08);
            text-align:center;
            min-height:120px;
            display:flex;
            flex-direction:column;
            justify-content:center;
            ">

            <div style="
            font-size:15px;
            font-weight:600;
            color:#64748B;
            ">
            {icon} {title}
            </div>

            <div style="
            margin-top:12px;
            font-size:28px;
            font-weight:800;
            color:{color};
            ">
            {value}
            </div>

            </div>
            """)

            if critical_cases:

                st.error(
                    f"🚨 {critical_cases} critical patient(s) require immediate attention."
                )

            else:

                st.success(
                    "✅ No critical patients waiting."
                )

            # ---------------------------------------------------------
            # SEARCH + FILTERS
            # ---------------------------------------------------------

            left, right = st.columns([5,1])

            with left:

                search = st.text_input(
                    "🔍 Search Patient",
                    placeholder="Patient name or Case ID..."
                )

            with right:

                show_resolved = st.toggle(
                    "Show Resolved",
                    value=False
                )

            departments = sorted(
                doctor_df["department"]
                .dropna()
                .unique()
                .tolist()
            )

            selected_department = st.selectbox(
                "🏥 Department",
                ["All Departments"] + departments
            )

            # ---------------------------------------------------------
            # FILTER DATA
            # ---------------------------------------------------------

            filtered_df = doctor_df.copy()

            if not show_resolved:

                filtered_df = filtered_df[
                    filtered_df["status"] != "Resolved"
                ]

            if selected_department != "All Departments":

                filtered_df = filtered_df[
                    filtered_df["department"]
                    ==
                    selected_department
                ]

            if search:

                s = search.lower()

                filtered_df = filtered_df[
                    filtered_df["patient_name"]
                        .astype(str)
                        .str.lower()
                        .str.contains(s)

                    |

                    filtered_df["id"]
                        .astype(str)
                        .str.contains(s)
                ]

            # ---------------------------------------------------------
            # ACTIVE QUEUE INFO
            # ---------------------------------------------------------

            workload = (
                filtered_df[filtered_df["status"] != "Resolved"]
                .groupby("department")
                .size()
                .sort_values(ascending=False)
            )

            if not workload.empty:
                chips = "".join(
                    f'<span style="background:#EEF2FF; color:#3B4C8C; font-size:12px; '
                    f'font-weight:600; padding:4px 12px; border-radius:12px; margin-right:6px; '
                    f'display:inline-block;">🏥 {dept}: {count}</span>'
                    for dept, count in workload.items()
                )
                st.markdown(
                    f'<div style="margin:10px 0 4px 0;">{chips}</div>',
                    unsafe_allow_html=True,
                )

            st.divider()

            # ======================================================
            # TIME AGO + OVERDUE HELPERS
            # ======================================================

            # ======================================================
            # PATIENT AVATAR COLOR - deterministic per name (not per
            # severity, which is already used elsewhere on the card), so
            # scanning a queue with several "Moderate" cases in a row still
            # gives an immediate visual anchor for WHO each card belongs to,
            # and the same patient keeps the same color across every card.
            # ======================================================

            AVATAR_PALETTE = [
                "#4C6EF5", "#F76707", "#12B886", "#E64980",
                "#7048E8", "#1098AD", "#F59F00", "#37B24D",
            ]

            def patient_avatar(name):
                name = str(name or "?").strip()
                initial = name[0].upper() if name else "?"
                color = AVATAR_PALETTE[sum(ord(c) for c in name.lower()) % len(AVATAR_PALETTE)]
                return initial, color

            def time_ago(timestamp):
                try:
                    created = pd.to_datetime(timestamp)
                    now = pd.Timestamp.now()
                    seconds = int((now - created).total_seconds())
                    if seconds < 0:
                        seconds = 0
                    if seconds < 60:
                        return "Just now", seconds
                    minutes = seconds // 60
                    if minutes < 60:
                        return f"{minutes} min ago", seconds
                    hours = minutes // 60
                    if hours < 24:
                        return f"{hours} hr ago", seconds
                    days = hours // 24
                    if days < 30:
                        return f"{days} day{'s' if days != 1 else ''} ago", seconds
                    return created.strftime("%d-%m-%Y"), seconds
                except Exception:
                    return str(timestamp), 0

            # Rough SLA thresholds by severity - a case sitting untouched past
            # this gets flagged, so nothing critical silently ages in the queue.
            OVERDUE_THRESHOLD_SECONDS = {
                "critical": 30 * 60,       # 30 min
                "moderate": 3 * 60 * 60,   # 3 hr
                "mild": 24 * 60 * 60,      # 24 hr
            }

            # ======================================================
            # CASE QUEUE
            # ======================================================
            st.divider()

            st.markdown("## 🗂 Patient Queue")

            if filtered_df.empty:
                st.info("No active cases" + ("" if show_resolved else " (resolved cases are hidden - toggle above to include them)") + ".")

            else:
                for _, case in filtered_df.iterrows():

                    case_id = int(case["id"])

                    status = case.get("status", "Pending")
                    if pd.isna(status) or not status:
                        status = "Pending"

                    time_text, elapsed_seconds = time_ago(case["created_at"])

                    severity_key = str(case["severity"]).strip().lower()
                    is_overdue = (
                        status != "Resolved"
                        and elapsed_seconds > OVERDUE_THRESHOLD_SECONDS.get(severity_key, 24 * 60 * 60)
                    )

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

                    severity_colors = {
                        "critical": {"bg": "#FDEDEC", "border": "#E74C3C", "text": "#C0392B"},
                        "moderate": {"bg": "#FBF2E3", "border": "#A67C52", "text": "#8B6A45"},
                        "mild":     {"bg": "#EAF6EE", "border": "#4C9A6A", "text": "#2F7A4D"},
                    }
                    colors = severity_colors.get(
                        severity_key,
                        {"bg": "#F1EEE8", "border": "#A6A6A6", "text": "#5C5C5C"},
                    )

                    overdue_badge = (
                        '<span style="background:#C0392B; color:white; font-size:11px; '
                        'font-weight:700; padding:2px 8px; border-radius:8px; margin-left:8px;">'
                        '⏱ OVERDUE</span>'
                    ) if is_overdue else ""

                    avatar_initial, avatar_color = patient_avatar(case["patient_name"])

                    with st.container(border=True):
                    
                     

                        card_html = dedent(f"""
                        <div style="
                        background:#FFFFFF;
                        border-left:8px solid {colors['border']};
                        border-radius:16px;
                        padding:12px;
                        margin-bottom:6px;
                        box-shadow:0 4px 12px rgba(0,0,0,.08);
                        ">

                        <div style="display:flex;justify-content:space-between;align-items:center;">

                        <div style="display:flex;align-items:center;gap:12px;">

                        <div style="
                        width:44px;height:44px;min-width:44px;border-radius:50%;
                        background:{avatar_color};color:white;font-size:18px;font-weight:800;
                        display:flex;align-items:center;justify-content:center;
                        box-shadow:0 2px 6px rgba(0,0,0,.15);
                        ">
                        {avatar_initial}
                        </div>

                        <div>

                        <div style="
                        font-size:22px;
                        font-weight:800;
                        color:#2C3E50;
                        ">
                        {severity_icon} {case["patient_name"]}
                        </div>

                        <div style="
                        margin-top:5px;
                        font-size:13px;
                        color:#7F8C8D;
                        ">

                        📄 Case #{case_id}
                        &nbsp;&nbsp;•&nbsp;&nbsp;
                        🕒 {time_text}

                        </div>

                        </div>

                        </div>

                        <div>

                        {overdue_badge}

                        </div>

                        </div>

                        <hr style="margin:15px 0;">

                        <div style="
                        display:grid;
                        grid-template-columns:repeat(3,1fr);
                        gap:15px;
                        ">

                        <div>

                        <div style="
                        font-size:12px;
                        color:#7F8C8D;
                        ">
                        SEVERITY
                        </div>

                        <div style="
                        font-size:16px;
                        font-weight:700;
                        color:{colors["text"]};
                        ">
                        {severity_icon} {case["severity"]}
                        </div>

                        </div>

                        <div>

                        <div style="
                        font-size:12px;
                        color:#7F8C8D;
                        ">
                        DEPARTMENT
                        </div>

                        <div style="
                        font-size:16px;
                        font-weight:600;
                        ">
                        🏥 {case["department"]}
                        </div>

                        </div>

                        <div>

                        <div style="
                        font-size:12px;
                        color:#7F8C8D;
                        ">
                        STATUS
                        </div>

                        <div style="
                        font-size:16px;
                        font-weight:600;
                        ">
                        {status_icon} {status}
                        </div>

                        </div>

                        </div>

                        </div>
                        """)

                        st.html(card_html)

                        st.markdown(
                            f"""
                        <div style="
                        background:#F8FAFC;
                        padding:10px 14px;
                        border-radius:10px;
                        margin:6px 0;
                        font-size:15px;
                        ">
                        🩺 <b>Symptoms:</b> {case["symptoms"]}
                        </div>
                        """,
                        unsafe_allow_html=True,
                        )

                        # ------------------------------------------
                        # CLINICAL CONTEXT - pulled from the patient's saved
                        # Health Profile, not from this case row. This is the
                        # thing a doctor actually needs before walking in that
                        # Case History has no reason to show: what this
                        # specific patient is allergic to and already on.
                        # Rendered as one consistent card grid (matching the
                        # main patient card's styling) instead of stacked
                        # st.success/error/warning/info banners, which read as
                        # four different UI components rather than one panel.
                        # ------------------------------------------

                        _profile = get_profile(str(case["patient_name"]))

                        if _profile:
                            profile_fields = []
                            if _profile.get("blood_group") and _profile["blood_group"] != "Unknown":
                                profile_fields.append(("🩸", "Blood Group", _profile["blood_group"], "#2F7A4D", "#EAF6EE"))
                            if _profile.get("allergies"):
                                profile_fields.append(("⚠️", "Allergies", _profile["allergies"], "#C0392B", "#FDEDEC"))
                            if _profile.get("chronic_conditions") and _profile["chronic_conditions"].lower() != "none":
                                profile_fields.append(("🩺", "Chronic Conditions", _profile["chronic_conditions"], "#8B6A45", "#FBF2E3"))
                            if _profile.get("current_medications"):
                                profile_fields.append(("💊", "Medications", _profile["current_medications"], "#34495E", "#EEF2F7"))

                            if profile_fields:
                                st.markdown("##### 📋 Patient Health Profile")
                                card_pieces = []
                                for icon, label, value, txt, bg in profile_fields:
                                    card_pieces.append(
                                        "<div style=\"background:" + bg + "; border-radius:10px; padding:10px 14px; "
                                        "margin-bottom:8px;\">"
                                        "<div style=\"font-size:11px; font-weight:700; color:" + txt + "; letter-spacing:0.4px;\">"
                                        + icon + " " + label.upper() + "</div>"
                                        "<div style=\"font-size:14px; color:#2C3E50; margin-top:2px;\">"
                                        + str(value) + "</div>"
                                        "</div>"
                                    )
                                profile_cards_html = "".join(card_pieces)
                                st.markdown(
                                    "<div style=\"display:grid; grid-template-columns:repeat(2,1fr); gap:10px;\">"
                                    + profile_cards_html + "</div>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.caption("📋 No health profile details saved for this patient yet.")
                        else:
                            st.caption(
                                "📋 No saved Health Profile for this patient - allergies and "
                                "medications won't show here until one is added on the Health Profile tab."
                            )

                        if case.get("summary") or case.get("recommendation"):
                            with st.expander("🧠 AI Assessment", expanded=False):
                                if case.get("summary"):
                                    st.markdown("**Clinical Summary**")
                                    st.write(case["summary"])
                                if case.get("recommendation"):
                                    st.markdown("**Recommended Action**")
                                    st.write(case["recommendation"])

                        # ------------------------------------------
                        # DOCTOR CONSULTATION NOTES - genuinely doctor-only
                        # data that Case History has no equivalent for: a
                        # persisted free-text note the doctor writes during
                        # or after seeing the patient, saved back to this case.
                        # ------------------------------------------

                        with st.expander(
                            "📝 Consultation Notes" + (" (saved)" if case.get("doctor_notes") else ""),
                            expanded=False,
                        ):
                            notes_value = st.text_area(
                                "Notes",
                                value=case.get("doctor_notes") or "",
                                key=f"notes_{case_id}",
                                label_visibility="collapsed",
                                placeholder="e.g. Discussed symptoms, ordered CBC, follow up in 1 week...",
                                height=100,
                            )
                            if st.button("💾 Save Notes", key=f"save_notes_{case_id}"):
                                update_case_notes(case_id, notes_value)
                                st.success("Notes saved.")
                                st.rerun()

                        # ------------------------------------------
                        # QUICK ACTIONS - status transitions plus a jump into
                        # Appointment Prep pre-loaded with this patient, so a
                        # doctor wrapping up a visit can hand off straight into
                        # generating next-visit prep without re-selecting the
                        # patient over on that tab.
                        # ------------------------------------------

                        st.markdown("##### 🔄 Update Status")

                        btn1, btn2, btn3, btn4 = st.columns([1, 1, 1, 1.3])

                        with btn1:
                            if st.button(
                                "⏳ Pending",
                                key=f"pending_{case_id}",
                                disabled=(status == "Pending"),
                                width="stretch",
                            ):
                                update_case_status(case_id, "Pending")
                                st.rerun()

                        with btn2:
                            if st.button(
                                "🩺 In Progress",
                                key=f"progress_{case_id}",
                                disabled=(status == "In Progress"),
                                width="stretch",
                            ):
                                update_case_status(case_id, "In Progress")
                                st.rerun()

                        with btn3:
                            if st.button(
                                "✅ Resolved",
                                key=f"resolved_{case_id}",
                                disabled=(status == "Resolved"),
                                width="stretch",
                            ):
                                update_case_status(case_id, "Resolved")
                                st.rerun()

                        with btn4:
                            _patient_has_history = str(case["patient_name"]) in get_all_patients_with_history()
                            if st.button(
                                "🗓️ Prep next visit",
                                key=f"prep_link_{case_id}",
                                disabled=not _patient_has_history,
                                width="stretch",
                                help="Preloads this patient on the Appointment Prep tab.",
                            ):
                                st.session_state["prep_patient_select"] = str(case["patient_name"])
                                st.toast(f"{case['patient_name']} is loaded on the Appointment Prep tab.", icon="🗓️")

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


# ── Lab Report LLM (cached so it isn't rebuilt on every rerun) ──
@st.cache_resource
def get_lab_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY")
    )


_lab_llm = get_lab_llm()

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
                    
# ── TAB 6 ─────────────────────────────────────────────────────────
with tab6:
    st.header("🧪 Lab Report Explainer")
    st.markdown(
        "Upload a blood/lab report (PDF or photo) to get a plain-English summary, "
        "flagged values, and a trend chart for repeat tests over time."
    )
    st.caption(
        "Regular PDF reports (with selectable text) are read directly. "
        "If your report is a scan or a photo with no selectable text, upload it as a "
        "PNG/JPG image instead of a PDF - it'll be read via OCR."
    )

    st.divider()

    # Apply any pending patient auto-selection from a previous rerun,
    # BEFORE the selectbox below is instantiated. Streamlit forbids writing
    # to a widget's session_state key after that widget has already been
    # drawn in the same run, so this has to happen up here, one run later.
    if "lab_pending_patient_select" in st.session_state:
        st.session_state["lab_patient_select"] = st.session_state.pop("lab_pending_patient_select")

    # Same rule applies to the "new patient" text input - it has to be
    # cleared here, before it's instantiated below, not from inside the
    # button handler after it's already been drawn (that raises
    # StreamlitAPIException: cannot be modified after the widget is instantiated).
    if st.session_state.pop("lab_pending_new_patient_clear", False):
        st.session_state["lab_new_patient_name"] = ""

    existing_patients = get_all_patient_names()
    patient_options = ["+ Add new patient..."] + existing_patients

    _is_patient_session = st.session_state.get("auth_role") == "patient"

    if _is_patient_session:
        # Patient sessions never get a picker over every patient in the
        # system - locked to their own identity from login, same as the
        # Triage tab.
        lab_patient_name = st.session_state.get("auth_patient_name", "")
        st.text_input(
            "Patient",
            value=lab_patient_name,
            disabled=True,
            key="lab_locked_patient_display",
            help="Locked to your account's identity.",
        )
    else:
        selected_patient_option = st.selectbox(
            "Patient",
            options=patient_options,
            key="lab_patient_select",
        )

        if selected_patient_option == "+ Add new patient...":
            lab_patient_name = st.text_input(
                "New patient name",
                key="lab_new_patient_name",
                placeholder="e.g. Rohan Sharma",
            )
        else:
            lab_patient_name = selected_patient_option

    uploaded_file = st.file_uploader(
        "Upload report",
        type=["pdf", "png", "jpg", "jpeg"],
        key="lab_report_upload",
    )

    if st.button("📄 Analyze Report", width="stretch"):
        if not lab_patient_name.strip():
            st.warning("Please enter the patient's name.")
        elif uploaded_file is None:
            st.warning("Please upload a PDF or image of the report.")
        else:
            try:
                with st.spinner("Reading the report..."):
                    file_bytes = uploaded_file.getvalue()
                    raw_text = extract_text_from_file(file_bytes, uploaded_file.name)

                with st.spinner("Extracting values and generating summary..."):
                    parsed = parse_lab_report_with_llm(raw_text, _lab_llm)

                report_id = save_lab_report(
                    patient_name=lab_patient_name.strip(),
                    file_name=uploaded_file.name,
                    raw_text=raw_text,
                    ai_summary=parsed["summary"],
                    parameters=parsed["parameters"],
                )

                # Stash the result in session_state so it's still shown after
                # the rerun below (needed when this was a brand-new patient -
                # otherwise the just-saved report would flash and disappear).
                st.session_state["lab_last_analysis"] = {
                    "patient": lab_patient_name.strip(),
                    "report_id": report_id,
                    "summary": parsed["summary"],
                    "parameters": parsed["parameters"],
                    "raw_text": raw_text,
                }

                if not _is_patient_session and selected_patient_option == "+ Add new patient...":
                    # Auto-select the new patient in the dropdown so their
                    # history/trend appears right away, instead of leaving
                    # "+ Add new patient..." selected after their first report.
                    # (Applied via the pending-key handoff at the top of this
                    # tab, not directly - see the comment up there for why.)
                    st.session_state["lab_pending_patient_select"] = lab_patient_name.strip()
                    st.session_state["lab_pending_new_patient_clear"] = True
                    st.rerun()

            except ExtractionError as e:
                st.error(str(e))
            except Exception as e:
                st.error("Something went wrong while analyzing this report.")
                st.exception(e)

    # Show the most recent analysis, if it belongs to the currently
    # selected patient. Reads from session_state so it survives the
    # rerun triggered above for newly-added patients.
    last_analysis = st.session_state.get("lab_last_analysis")
    if (
        last_analysis
        and lab_patient_name.strip()
        and last_analysis["patient"].lower() == lab_patient_name.strip().lower()
    ):
        st.success(f"Report analyzed and saved (Report #{last_analysis['report_id']}).")

        st.markdown("### 📝 Summary")
        st.info(last_analysis["summary"] or "No summary available.")

        if last_analysis["parameters"]:
            st.markdown("### 📊 Extracted Values")

            flag_colors = {
                "critical": "🔴",
                "high": "🟠",
                "low": "🟡",
                "normal": "🟢",
                "unknown": "⚪",
            }

            df_rows = []
            for p in last_analysis["parameters"]:
                ref_range = (
                    f"{p['ref_low']} - {p['ref_high']}"
                    if p["ref_low"] is not None and p["ref_high"] is not None
                    else "—"
                )
                df_rows.append({
                    "": flag_colors.get(p["flag"], "⚪"),
                    "Parameter": p["parameter"],
                    "Value": p["value"],
                    "Unit": p["unit"],
                    "Reference Range": ref_range,
                    "Flag": p["flag"].capitalize(),
                })

            st.dataframe(
                pd.DataFrame(df_rows),
                width="stretch",
                hide_index=True,
            )
        else:
            st.warning(
                "No structured lab values could be confidently extracted from this report. "
                "The raw text was still saved below."
            )

        with st.expander("View raw extracted text"):
            st.text(last_analysis["raw_text"][:5000])

    st.divider()

    if lab_patient_name.strip():
        st.markdown("### 📈 Trend View")

        known_params = get_known_parameters(lab_patient_name.strip())

        if not known_params:
            st.caption("No past reports for this patient yet - trends will appear once more than one report is analyzed.")
        else:
            selected_param = st.selectbox(
                "Select a parameter to chart",
                options=known_params,
                key="lab_trend_param",
            )

            trend_rows = get_trend_data(lab_patient_name.strip(), selected_param)

            if len(trend_rows) < 2:
                st.caption(f"Only one data point for {selected_param} so far - upload another report to see a trend.")

            trend_df = pd.DataFrame(trend_rows)
            trend_df["created_at"] = pd.to_datetime(trend_df["created_at"])

            fig = px.line(
                trend_df,
                x="created_at",
                y="value",
                markers=True,
                title=f"{selected_param} over time",
                labels={"created_at": "Date", "value": trend_df["unit"].iloc[-1] or "Value"},
            )

            ref_low = trend_df["ref_low"].dropna()
            ref_high = trend_df["ref_high"].dropna()
            if not ref_low.empty and not ref_high.empty:
                fig.add_hrect(
                    y0=ref_low.iloc[-1],
                    y1=ref_high.iloc[-1],
                    fillcolor="green",
                    opacity=0.08,
                    line_width=0,
                    annotation_text="Reference range",
                    annotation_position="top left",
                )

            st.plotly_chart(fig, width="stretch")

        st.markdown("### 🗂️ Past Reports")
        past_reports = get_reports_for_patient(lab_patient_name.strip())

        if not past_reports:
            st.caption("No reports uploaded yet for this patient.")
        else:
            for r in past_reports:
                with st.expander(f"{r['file_name']} — {r['created_at']}"):
                    st.info(r["ai_summary"] or "No summary available.")
                    values = get_values_for_report(r["id"])
                    if values:
                        st.dataframe(pd.DataFrame(values), width="stretch", hide_index=True)
    else:
        st.caption("Enter a patient name above to see their report history and trends.")

# ── TAB 7 ─────────────────────────────────────────────────────────
with tab7:
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

# ── TAB 8 ─────────────────────────────────────────────────────────
with tab8:
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

# ── TAB 9 ─────────────────────────────────────────────────────────
with tab9:
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