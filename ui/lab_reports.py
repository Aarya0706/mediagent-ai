"""
ui/lab_reports.py
--------------------
Tab 6 - "Lab Report Explainer": upload a lab report (PDF/photo),
extract and parse values with an LLM, flag out-of-range results, and
show trend charts across a patient's reports. Extracted verbatim from
app.py's `with tab6:` block.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from services.llm_clients import get_lab_llm
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

_lab_llm = get_lab_llm()


def render_lab_reports_tab():
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

