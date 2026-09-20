"""
tests/test_report_service.py
-------------------------------
Unit tests for services/report_service.py, extracted from app.py.
Pure functions - no Streamlit, no DB, no LLM - so these are fast and
CI-safe like the other tests/test_*.py modules. This coverage wasn't
possible before the extraction (app.py can't be imported standalone -
it calls st.set_page_config() etc. at module scope).

Run:
    pytest tests/test_report_service.py -v
"""

import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.report_service import clean_text_for_pdf, generate_pdf_report


def test_clean_text_for_pdf_handles_empty_and_none():
    assert clean_text_for_pdf("") == ""
    assert clean_text_for_pdf(None) == ""


def test_clean_text_for_pdf_replaces_smart_quotes_and_dashes():
    text = "It\u2019s a \u201ctest\u201d \u2014 really\u2026"
    result = clean_text_for_pdf(text)
    assert "\u2019" not in result
    assert "\u201c" not in result
    assert "'" in result
    assert '"' in result
    assert "..." in result


def test_clean_text_for_pdf_drops_non_latin1_chars():
    result = clean_text_for_pdf("Fever \U0001F912 and cough")
    assert "\U0001F912" not in result
    assert "Fever" in result
    assert "cough" in result


def test_clean_text_for_pdf_force_breaks_long_tokens():
    long_word = "a" * 100
    result = clean_text_for_pdf(long_word, max_word_len=40)
    # Should be split into space-separated chunks of <= 40 chars each
    assert all(len(chunk) <= 40 for chunk in result.split(" "))


def test_generate_pdf_report_returns_nonempty_pdf_bytes():
    result = {
        "summary": "Patient presents with mild headache.",
        "actions": ["Rest", "Hydrate", "Follow up if symptoms persist"],
        "warning": None,
    }
    pdf_bytes = generate_pdf_report(
        patient_name="Test Patient", age=30, gender="Female", phone="0000000000",
        body_part="Head", symptoms_desc="Headache", duration="2 days",
        onset_type="Gradual", severity_slider=4, conditions_str="None",
        severity="Mild", department="General Medicine", urgency=3,
        result=result,
        generated_at=datetime(2026, 1, 1, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    assert pdf_bytes.startswith(b"%PDF")


def test_generate_pdf_report_includes_emergency_warning_section():
    result_with_warning = {
        "summary": "Critical presentation.",
        "actions": ["Call emergency services immediately"],
        "warning": "Possible cardiac event - seek immediate care.",
    }
    result_without_warning = dict(result_with_warning, warning=None)

    ts = datetime(2026, 1, 1, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    pdf_with = generate_pdf_report(
        "Test", 40, "Male", "0000000000", "Chest", "Chest pain", "1 hour",
        "Sudden", 9, "None", "Critical", "Emergency", 10, result_with_warning,
        generated_at=ts,
    )
    pdf_without = generate_pdf_report(
        "Test", 40, "Male", "0000000000", "Chest", "Chest pain", "1 hour",
        "Sudden", 9, "None", "Critical", "Emergency", 10, result_without_warning,
        generated_at=ts,
    )
    # The warning section adds real content, so the PDF with a warning
    # should be meaningfully larger than the one without.
    assert len(pdf_with) > len(pdf_without)


def test_generate_pdf_report_handles_no_actions():
    result = {"summary": "Routine check.", "actions": [], "warning": None}
    pdf_bytes = generate_pdf_report(
        "Test", 25, "Other", "0000000000", "General", "Routine", "N/A",
        "N/A", 1, "None", "Mild", "General Medicine", 1, result,
        generated_at=datetime(2026, 1, 1, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
