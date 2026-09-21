"""
ui/dashboard.py
------------------
Tab 3 - "Dashboard": real-time hospital analytics (case volume,
department mix, severity breakdown, hourly patterns) - staff-only.
Extracted verbatim from app.py's `with tab3:` block.
"""

import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

from database.connection import DB_PATH
from ui.components import render_landing_stat_cards


def render_dashboard_tab():
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
