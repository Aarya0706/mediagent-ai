import io
import os
import json
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DB_PATH = os.path.join(
    ROOT,
    "data",
    "hospital.db"
)

os.makedirs(
    os.path.dirname(DB_PATH),
    exist_ok=True
)


# ============================================================
# DATABASE CONNECTION (same pattern as tools/save_case.py)
# ============================================================

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# INITIALIZE / MIGRATE DATABASE
# ============================================================
#
# Two tables, mirroring the MediCore PRD's design decision:
# lab_values is kept SEPARATE from lab_reports (not a JSON blob)
# so trend queries like
#   SELECT value, created_at FROM lab_values
#   WHERE patient_name=? AND parameter=? ORDER BY created_at
# are simple, indexed SQL - no JSON parsing needed for charts.

def initialize_lab_tables():
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS lab_reports
            (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT NOT NULL,
                file_name TEXT NOT NULL,
                raw_text TEXT DEFAULT '',
                ai_summary TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS lab_values
            (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id INTEGER NOT NULL,
                patient_name TEXT NOT NULL,
                parameter TEXT NOT NULL,
                value REAL,
                unit TEXT DEFAULT '',
                ref_low REAL,
                ref_high REAL,
                flag TEXT DEFAULT 'unknown',
                created_at TEXT NOT NULL,
                FOREIGN KEY (report_id) REFERENCES lab_reports (id)
            )
            """
        )

        conn.commit()


# Run initialization automatically whenever imported (same pattern as save_case.py)
initialize_lab_tables()


# ============================================================
# TEXT EXTRACTION
# ============================================================
#
# v1 scope (kept deliberately narrow so it doesn't need extra
# system packages beyond Tesseract):
#   - Text-based PDFs  -> pdfplumber (pure python, no system dep)
#   - Photos / screenshots (png/jpg) -> pytesseract (needs the
#     tesseract-ocr system binary - see packages.txt)
#   - Scanned/image-only PDFs are NOT OCR'd in v1. If no text is
#     found in a PDF, the caller gets a clear message asking the
#     user to upload a photo of the report instead. This can be
#     extended later with pdf2image + poppler if needed.

class ExtractionError(Exception):
    pass


def extract_text_from_pdf(file_bytes: bytes) -> str:
    import pdfplumber

    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)

    text = "\n".join(text_parts).strip()

    if not text:
        raise ExtractionError(
            "No embedded text found in this PDF - it looks like a scanned "
            "image rather than a text PDF. Please upload a clear photo or "
            "screenshot of the report (PNG/JPG) instead."
        )

    return text


def extract_text_from_image(file_bytes: bytes) -> str:
    import pytesseract
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError:
        raise ExtractionError(
            "Tesseract OCR is not installed on this machine. "
            "Install it (e.g. `sudo apt install tesseract-ocr` on Linux, "
            "or see https://github.com/tesseract-ocr/tesseract for other "
            "platforms) and try again."
        )

    text = text.strip()
    if not text:
        raise ExtractionError(
            "No readable text could be extracted from this image. "
            "Try a clearer, well-lit photo of the report."
        )

    return text


def extract_text_from_file(file_bytes: bytes, file_name: str) -> str:
    ext = os.path.splitext(file_name)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in (".png", ".jpg", ".jpeg", ".webp"):
        return extract_text_from_image(file_bytes)
    else:
        raise ExtractionError(
            f"Unsupported file type '{ext}'. Please upload a PDF, PNG, or JPG."
        )


# ============================================================
# LLM PARSING (structured extraction from raw OCR/PDF text)
# ============================================================

LAB_PARSER_SYSTEM_PROMPT = """You are a clinical lab report analyst. You will receive raw text
extracted from a patient's blood/lab report (via PDF text extraction or OCR, so it may contain
minor spacing or line-break artifacts).

Extract every individual test parameter you can confidently identify, and produce ONE plain-English
summary of the whole report.

Respond with ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:

{{
  "summary": "2-4 sentence plain-English summary of what is normal, what is flagged, and what that generally means. No diagnosis, no treatment instructions - just explain the numbers.",
  "parameters": [
    {{
      "parameter": "Hemoglobin",
      "value": 13.5,
      "unit": "g/dL",
      "ref_low": 13.0,
      "ref_high": 17.0,
      "flag": "normal"
    }}
  ]
}}

Rules:
- "flag" must be one of: "low", "normal", "high", "critical". Use "critical" only when the value is
  far outside the reference range in a way that could be clinically urgent.
- If a reference range is not present in the text, set ref_low/ref_high to null and set flag to
  "unknown" - do not invent a normal range.
- If "value" cannot be parsed as a number, omit that parameter entirely rather than guessing.
- Only include rows you can actually find in the text. Do not fabricate parameters.
- Never include any advice to start, stop, or change medication.
"""


def parse_lab_report_with_llm(raw_text: str, llm) -> dict:
    """
    llm: a LangChain-compatible chat model (already configured with the
    Groq API key) is passed in from app.py so this module has no direct
    dependency on how the app wires up its LLM client.
    Returns {"summary": str, "parameters": [dict, ...]}.
    """
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system", LAB_PARSER_SYSTEM_PROMPT),
        ("human", "Raw report text:\n\n{raw_text}"),
    ])

    chain = prompt | llm | StrOutputParser()

    # Lab reports can be long; truncate defensively to keep the prompt
    # within a reasonable size for the LLM context window.
    truncated_text = raw_text[:12000]

    output = chain.invoke({"raw_text": truncated_text})

    cleaned = output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        raise ExtractionError(
            "The AI could not produce a structured reading of this report. "
            "Try re-uploading a clearer copy, or a different report."
        )

    parameters = parsed.get("parameters", [])
    clean_params = []
    for p in parameters:
        try:
            value = float(p.get("value"))
        except (TypeError, ValueError):
            continue

        clean_params.append({
            "parameter": str(p.get("parameter", "")).strip() or "Unknown",
            "value": value,
            "unit": str(p.get("unit", "") or "").strip(),
            "ref_low": _safe_float(p.get("ref_low")),
            "ref_high": _safe_float(p.get("ref_high")),
            "flag": str(p.get("flag", "unknown") or "unknown").strip().lower(),
        })

    return {
        "summary": str(parsed.get("summary", "")).strip(),
        "parameters": clean_params,
    }


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_name(name) -> str:
    # Collapses stray whitespace (e.g. "Rohan   Sharma " -> "Rohan Sharma").
    # Matching against existing patients is done case-insensitively in the
    # query functions below, so this is a safety net - the dropdown in the
    # UI is what actually prevents typos/duplicates going forward.
    cleaned = " ".join(str(name or "").split())
    return cleaned or "Unknown"


# ============================================================
# SAVE
# ============================================================

def save_lab_report(patient_name, file_name, raw_text, ai_summary, parameters):
    patient_name = _normalize_name(patient_name)
    file_name = str(file_name).strip()
    created_at = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO lab_reports (patient_name, file_name, raw_text, ai_summary, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (patient_name, file_name, raw_text, ai_summary, created_at),
        )

        report_id = cursor.lastrowid

        for p in parameters:
            cursor.execute(
                """
                INSERT INTO lab_values
                (report_id, patient_name, parameter, value, unit, ref_low, ref_high, flag, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    patient_name,
                    p["parameter"],
                    p["value"],
                    p["unit"],
                    p["ref_low"],
                    p["ref_high"],
                    p["flag"],
                    created_at,
                ),
            )

        conn.commit()
        return report_id


# ============================================================
# QUERY
# ============================================================

def get_reports_for_patient(patient_name):
    patient_name = _normalize_name(patient_name)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, file_name, ai_summary, created_at
            FROM lab_reports
            WHERE LOWER(patient_name) = LOWER(?)
            ORDER BY created_at DESC
            """,
            (patient_name,),
        )
        return [dict(row) for row in cursor.fetchall()]


def get_all_patient_names():
    """Distinct patient names that already have at least one lab report,
    sorted alphabetically (case-insensitive) - used to populate the
    patient-selection dropdown so names are picked, not retyped."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT patient_name
            FROM lab_reports
            ORDER BY LOWER(patient_name)
            """
        )
        return [row["patient_name"] for row in cursor.fetchall()]


def get_values_for_report(report_id):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT parameter, value, unit, ref_low, ref_high, flag
            FROM lab_values
            WHERE report_id = ?
            ORDER BY parameter
            """,
            (report_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


def get_known_parameters(patient_name):
    patient_name = _normalize_name(patient_name)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT parameter
            FROM lab_values
            WHERE LOWER(patient_name) = LOWER(?)
            ORDER BY parameter
            """,
            (patient_name,),
        )
        return [row["parameter"] for row in cursor.fetchall()]


def get_trend_data(patient_name, parameter):
    patient_name = _normalize_name(patient_name)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT value, unit, ref_low, ref_high, flag, created_at
            FROM lab_values
            WHERE LOWER(patient_name) = LOWER(?) AND parameter = ?
            ORDER BY created_at ASC
            """,
            (patient_name, parameter),
        )
        return [dict(row) for row in cursor.fetchall()]