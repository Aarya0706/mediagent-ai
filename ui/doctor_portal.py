"""
ui/doctor_portal.py
----------------------
Tab 4 - "Doctor Portal": the staff clinical work queue (severity/
urgency-sorted case cards, status updates, doctor notes) plus the
Agent Observability panel. Extracted verbatim from app.py's
`with tab4:` block.
"""

import sqlite3
from textwrap import dedent

import pandas as pd
import streamlit as st

from database.connection import DB_PATH
from tools.appointment_prep_tools import get_all_patients_with_history
from tools.health_profile_tools import get_profile
from tools.save_case import update_case_notes, update_case_status


def render_doctor_portal_tab():
    if st.session_state.get("auth_role") == "patient":
        st.warning("This view is for hospital staff only. Patients don't have a doctor work queue.")
    else:

        st.header("👨‍⚕️ Doctor Portal")
        st.caption(
            "Live clinical work queue prioritized by severity and urgency."
        )

        # ---------------------------------------------------------
        # AGENT OBSERVABILITY
        # ---------------------------------------------------------
        # Shown before the case queue (and before the st.stop() below
        # for an empty queue) so it's visible even on a fresh
        # install with zero cases yet. See agents/observability.py -
        # this reads call metadata only, never symptom/patient text.
        with st.expander("🔎 Agent Observability (last 24h)", expanded=False):
            try:
                from agents.observability import get_observability_stats
                obs_stats = get_observability_stats(hours=24)

                if obs_stats["total_agent_calls"] == 0:
                    st.caption("No agent runs recorded in this window yet.")
                else:
                    oc1, oc2, oc3, oc4 = st.columns(4)
                    oc1.metric("Triage runs", obs_stats["total_runs"])
                    oc2.metric("Agent calls", obs_stats["total_agent_calls"])
                    oc3.metric(
                        "Success rate",
                        f"{obs_stats['success_rate_pct']}%"
                        if obs_stats["success_rate_pct"] is not None
                        else "—",
                    )
                    oc4.metric(
                        "Avg latency",
                        f"{obs_stats['avg_latency_ms']:.0f} ms"
                        if obs_stats["avg_latency_ms"] is not None
                        else "—",
                    )

                    if obs_stats["by_agent"]:
                        st.caption("By agent")
                        st.dataframe(
                            pd.DataFrame(obs_stats["by_agent"]),
                            width="stretch",
                            hide_index=True,
                        )

                    if obs_stats["recent_failures"]:
                        st.caption("Recent failures")
                        st.dataframe(
                            pd.DataFrame(obs_stats["recent_failures"]),
                            width="stretch",
                            hide_index=True,
                        )
            except Exception as _obs_err:
                st.caption(f"Observability data unavailable: {_obs_err}")

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
        # (total_cases was computed here in the original app.py but
        # never read - dropped as dead code; wrapping this tab in a
        # function is what made ruff able to see it as unused, since
        # F841 only fires for genuinely local/function-scope names.)

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
