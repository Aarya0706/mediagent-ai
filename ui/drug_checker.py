"""
ui/drug_checker.py
---------------------
Tab 5 - "Drug Interaction Checker": OpenFDA-backed drug interaction
lookup with an LLM-generated plain-English explanation. Extracted
verbatim from app.py's `with tab5:` block.
"""

import streamlit as st

from services.drug_service import build_drug_chain, parse_drug_field, query_openfda
from services.llm_clients import get_drug_llm
from ui.components import show_error_details

_drug_chain = build_drug_chain(get_drug_llm())


def render_drug_checker_tab():
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
                fda_result = query_openfda(drug1.strip(), drug2.strip())

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
                    show_error_details(
                        e, "The clinical explanation could not be generated."
                    )
                    llm_output = ""

                if llm_output:
                    severity_label = parse_drug_field(llm_output, "SEVERITY")
                    plain_summary  = parse_drug_field(llm_output, "PLAIN_SUMMARY")
                    mechanism      = parse_drug_field(llm_output, "MECHANISM")
                    patient_advice = parse_drug_field(llm_output, "PATIENT_ADVICE")

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
                        "Clinical explanation generated by Groq (OpenAI GPT-OSS). "
                        "This tool is for informational purposes only - always consult a licensed pharmacist or physician."
                    )
                    
