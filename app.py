import streamlit as st
import sqlite3
import pandas as pd
from dotenv import load_dotenv

import sys
import os
from tools.appointment_prep_tools import (
    get_recent_cases,
    get_recent_lab_reports,
)
from tools.health_profile_tools import get_profile
from tools.auth_tools import (
    create_user,
    verify_user,
    get_user_count,
    ensure_demo_user,
    ensure_demo_patient_user,
)
load_dotenv()


ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from database.connection import DB_PATH  # noqa: F401 - kept for any code below that still references DB_PATH directly
from database.migrations import run_migrations

# Creates every table if missing and backfills any column an existing
# database doesn't have yet (patient_name on cases, role/patient_name
# on users, updated_at everywhere, etc.) - single entrypoint replacing
# what used to be this function's own inline ALTER TABLE, plus four
# more copies of the same idea scattered across tools/*.py. See
# database/schema.py + database/migrations.py.
run_migrations()

# Tab 1 (Patient Triage) now lives in ui/patient_triage.py, which
# imports run_triage_pipeline itself.
from ui.patient_triage import render_patient_triage_tab
from ui.case_history import render_case_history_tab
from ui.dashboard import render_dashboard_tab
from ui.doctor_portal import render_doctor_portal_tab
from ui.drug_checker import render_drug_checker_tab
from ui.lab_reports import render_lab_reports_tab
from ui.appointment_prep import render_appointment_prep_tab
from ui.health_chat import render_health_chat_tab
from ui.health_profile import render_health_profile_tab

 
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
#
# Extracted to services/report_service.py, now used only by
# ui/patient_triage.py (Tab 1), which imports it directly.


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
#
# render_landing_stat_cards itself now lives in ui/components.py (a
# reusable rendering helper, not landing-screen-specific logic).
from ui.components import render_landing_stat_cards  # noqa: E402


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
    render_patient_triage_tab()
# ── TAB 2 ─────────────────────────────────────────────────────────
with tab2:
    render_case_history_tab()
with tab3:
    render_dashboard_tab()
with tab4:
    render_doctor_portal_tab()


# ── TAB 5 ─────────────────────────────────────────────────────────
with tab5:
    render_drug_checker_tab()
# ── TAB 6 ─────────────────────────────────────────────────────────
with tab6:
    render_lab_reports_tab()
# ── TAB 7 ─────────────────────────────────────────────────────────
with tab7:
    render_appointment_prep_tab()

# ── TAB 8 ─────────────────────────────────────────────────────────
with tab8:
    render_health_chat_tab()

# ── TAB 9 ─────────────────────────────────────────────────────────
with tab9:
    render_health_profile_tab()

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
