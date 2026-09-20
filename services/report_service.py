"""
services/report_service.py
-----------------------------
PDF triage report generation, extracted verbatim from app.py (where it
was two of the file's largest functions, entangled with everything
else purely by sitting in the same 3500-line file - neither function
actually touches Streamlit, session state, or the database).

No behavior changes: clean_text_for_pdf and generate_pdf_report here
are the same logic that lived in app.py, byte-for-byte in the PDF
output. The one difference is generate_pdf_report() now takes an
explicit `generated_at` instead of calling app.py's now_ist() directly
- a service module shouldn't import back from app.py (that's a
circular dependency waiting to happen), and a function that builds a
timestamped report is easier to test when the timestamp is a
parameter instead of "whatever time it happens to run."
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from fpdf import FPDF

IST = ZoneInfo("Asia/Kolkata")


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
    severity, department, urgency, result,
    generated_at=None,
):
    """Builds the patient-facing triage PDF. `generated_at` defaults to
    now (IST) if not given - pass an explicit datetime for reproducible
    output (tests, previews)."""
    generated_at = generated_at or datetime.now(IST)

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
        f"Generated: {generated_at.strftime('%d-%m-%Y %H:%M')} IST"
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
