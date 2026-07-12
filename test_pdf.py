from fpdf import FPDF
from fpdf.enums import XPos, YPos
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

PAGE_MARGIN = 10
CONTENT_WIDTH = 210 - (PAGE_MARGIN * 2)  # 190mm usable width on A4


def now_ist():
    return datetime.now(IST)


def clean_text_for_pdf(text, max_word_len=25):
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
    text = text.encode("latin-1", "ignore").decode("latin-1")
    text = " ".join(text.split())  # collapse odd/repeated whitespace
    words = text.split(" ")
    safe_words = []
    for w in words:
        if len(w) > max_word_len:
            safe_words.append(" ".join(w[i:i + max_word_len] for i in range(0, len(w), max_word_len)))
        else:
            safe_words.append(w)
    return " ".join(safe_words)


def safe_multicell(pdf, text, line_height=7):
    """Always reset X to the left margin before multi_cell and use an
    explicit width, to avoid fpdf2's 'Not enough horizontal space' bug
    that can occur when width=0 and the cursor X isn't exactly at margin."""
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(CONTENT_WIDTH, line_height, text)


def generate_pdf_report(patient_name, age, gender, phone, body_part, symptoms_desc,
                         duration, onset_type, severity_slider, conditions_str,
                         severity, department, urgency, result):
    pdf = FPDF()
    pdf.set_margin(PAGE_MARGIN)
    pdf.add_page()
    pdf.set_fill_color(166, 124, 82)
    pdf.rect(0, 0, 210, 25, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(10, 7)
    pdf.cell(0, 10, "MediAgent AI - Patient Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(10, 30)
    pdf.cell(0, 6, clean_text_for_pdf(f"Generated: {now_ist().strftime('%d-%m-%Y %H:%M')} IST"),
              new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Patient Details", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    safe_multicell(pdf, clean_text_for_pdf(
        f"Name: {patient_name}   |   Age: {age}   |   Gender: {gender}   |   Phone: {phone}"))
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Symptom Intake", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    safe_multicell(pdf, clean_text_for_pdf(
        f"Body Area: {body_part}\n"
        f"Description: {symptoms_desc}\n"
        f"Duration: {duration}   |   Onset: {onset_type}   |   Pain Level: {severity_slider}/10\n"
        f"Known Conditions: {conditions_str}"))
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Triage Result", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    sev_colors = {"Critical": (231, 76, 60), "Moderate": (241, 196, 15), "Mild": (46, 204, 113)}
    r, g, b = sev_colors.get(severity, (100, 100, 100))
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(50, 8, clean_text_for_pdf(f" Severity: {severity} "), fill=True,
              new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 8, clean_text_for_pdf(f"   Department: {department}   |   Urgency: {urgency}/10"),
              new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "AI Assessment", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    safe_multicell(pdf, clean_text_for_pdf(result.get('summary', '')))
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Recommended Actions", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    for i, action in enumerate(result.get("actions", []), 1):
        safe_multicell(pdf, clean_text_for_pdf(f"{i}. {action}"))

    if result.get("warning"):
        pdf.ln(3)
        pdf.set_fill_color(231, 76, 60)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(CONTENT_WIDTH, 8, clean_text_for_pdf(f"EMERGENCY WARNING: {result['warning']}"), fill=True)

    return bytes(pdf.output())


fake_result = {
    "summary": "You're experiencing sudden watering in your eyes, which is causing some discomfort.",
    "actions": [
        "Try to stay calm and avoid rubbing your eyes.",
        "Use artificial tears or over-the-counter eye drops.",
        "If it persists, see an ophthalmologist.",
        "If vision changes occur, seek care immediately."
    ],
    "warning": "If you experience severe vision loss or eye pain, go to the emergency room immediately."
}

pdf_bytes = generate_pdf_report(
    patient_name="Test Patient", age=28, gender="Female", phone="9999999999",
    body_part="Eyes", symptoms_desc="Watering and redness",
    duration="Just started (< 1 hour)", onset_type="Sudden / Abrupt",
    severity_slider=6, conditions_str="None reported",
    severity="Moderate", department="Ophthalmology", urgency=8,
    result=fake_result
)

with open("test_output.pdf", "wb") as f:
    f.write(pdf_bytes)

print("SUCCESS — PDF generated:", len(pdf_bytes), "bytes")