"""
ui/case_history.py
---------------------
Tab 2 - "Case History": browse, filter, and manage past triage cases
(staff-only). Extracted verbatim from app.py's `with tab2:` block.
"""

import streamlit as st

from tools.save_case import get_all_cases, delete_case, clear_all_cases
from tools.authorization import can_access_staff_view, STAFF_ONLY_MESSAGE


def render_case_history_tab():
    if not can_access_staff_view():
        st.warning(STAFF_ONLY_MESSAGE)
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
