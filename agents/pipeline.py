"""
agents/pipeline.py
------------------
3-agent LangChain pipeline for MediAgent AI.

Agents:
  1. IntakeAgent    — validates & normalises symptom input
  2. TriageAgent    — scores severity + recommends department
  3. RecommendAgent — generates action plan + summary

Usage:
  from agents.pipeline import run_triage_pipeline
  result = run_triage_pipeline(symptoms, patient_context)
"""

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ─────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)

parser = StrOutputParser()


# ══════════════════════════════════════════════════════════════════
# AGENT 1 — INTAKE AGENT
# Validates and normalises raw symptom input
# ══════════════════════════════════════════════════════════════════

intake_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a medical intake specialist at a hospital.
Your job is to validate and normalise patient symptom input before it reaches the triage doctor.

Rules:
- If input is gibberish, too vague (e.g. "bad", "help"), or non-medical, respond ONLY with:
  INVALID: <reason>
- If input is valid, return a structured intake summary in this EXACT format:

VALID
Primary Symptoms: <list the main symptoms clearly>
Body Systems Affected: <e.g. Cardiovascular, Respiratory, Neurological, Digestive, Musculoskeletal>
Symptom Duration: <extract if mentioned, else "Not specified">
Severity Indicators: <any intensity words like "severe", "mild", "sharp", "sudden">
Additional Context: <age, gender, known conditions if provided>

Be concise. Do not diagnose. Do not recommend treatment."""),
    ("human", """Patient Input: {symptoms}
Patient Context: {patient_context}""")
])

intake_chain = intake_prompt | llm | parser


# ══════════════════════════════════════════════════════════════════
# AGENT 2 — TRIAGE AGENT
# Scores severity and routes to department
# ══════════════════════════════════════════════════════════════════

triage_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior emergency triage physician.
You receive a structured intake summary and must assess severity and route the patient.

Severity Definitions:
- Critical: Life-threatening, requires immediate intervention (chest pain, difficulty breathing, stroke, severe bleeding, loss of consciousness)
- Moderate: Needs prompt attention within hours (high fever, persistent vomiting, moderate injury, severe pain)
- Mild: Can wait for routine appointment (cold, minor headache, mild nausea, small cuts)

Department List:
Emergency, Cardiology, Neurology, Pulmonology, Gastroenterology, Orthopedics,
Dermatology, ENT, Ophthalmology, Pediatrics, Psychiatry, Obstetrics & Gynecology,
Endocrinology, General Medicine

Respond in this EXACT format (no deviations):
SEVERITY: <Critical|Moderate|Mild>
DEPARTMENT: <department name from list above>
TRIAGE_REASONING: <2-3 sentences explaining why this severity and department>
URGENCY_SCORE: <1-10 where 10 is most urgent>"""),
    ("human", """Intake Summary:
{intake_output}""")
])

triage_chain = triage_prompt | llm | parser


# ══════════════════════════════════════════════════════════════════
# AGENT 3 — RECOMMENDATION AGENT
# Generates patient-facing action plan
# ══════════════════════════════════════════════════════════════════

recommend_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a compassionate medical assistant writing clear instructions for a patient.
You receive the triage assessment and must generate a patient-friendly action plan.

Respond in this EXACT format:
SUMMARY: <2-3 sentence plain-English summary for the patient. No jargon.>
ACTIONS:
1. <First recommended action>
2. <Second recommended action>
3. <Third recommended action>
4. <Fourth recommended action>
WARNING: <One sentence about symptoms that should trigger immediate emergency care. Only for Moderate or Critical. Write NONE for Mild.>

Keep language simple, clear, and reassuring. Never diagnose a specific disease."""),
    ("human", """Original Symptoms: {symptoms}
Triage Assessment: {triage_output}""")
])

recommend_chain = recommend_prompt | llm | parser


# ══════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════

def run_triage_pipeline(symptoms: str, patient_context: str = "") -> dict:
    """
    Runs the 3-agent pipeline sequentially.

    Parameters
    ----------
    symptoms        : raw symptom text from the patient form
    patient_context : optional e.g. "Age: 45, Gender: Male"

    Returns
    -------
    dict with keys:
        valid            : bool
        invalid_reason   : str | None
        intake           : str   — structured intake summary
        severity         : str   — Critical | Moderate | Mild
        department       : str   — recommended department
        urgency_score    : int   — 1-10
        triage_reasoning : str
        summary          : str   — patient-facing summary
        actions          : list[str]
        warning          : str | None
        raw_triage       : str
        raw_recommend    : str
    """

    # ── Step 1: Intake ────────────────────────────────────────────
    intake_output = intake_chain.invoke({
        "symptoms": symptoms,
        "patient_context": patient_context
    })

    if intake_output.strip().startswith("INVALID"):
        reason = intake_output.replace("INVALID:", "").strip()
        return {
            "valid": False,
            "invalid_reason": reason,
            "intake": intake_output,
            "severity": "Unknown",
            "department": "General Medicine",
            "urgency_score": 0,
            "triage_reasoning": "",
            "summary": "",
            "actions": [],
            "warning": None,
            "raw_triage": "",
            "raw_recommend": "",
        }

    # ── Step 2: Triage ────────────────────────────────────────────
    triage_output = triage_chain.invoke({
        "intake_output": intake_output
    })

    severity         = _extract_field(triage_output, "SEVERITY")
    department       = _extract_field(triage_output, "DEPARTMENT")
    triage_reasoning = _extract_field(triage_output, "TRIAGE_REASONING")
    urgency_raw      = _extract_field(triage_output, "URGENCY_SCORE")

    try:
        urgency_score = int(urgency_raw.split("/")[0].strip())
    except Exception:
        urgency_score = 5

    sev_lower = severity.lower()
    if "critical" in sev_lower:
        severity = "Critical"
    elif "moderate" in sev_lower:
        severity = "Moderate"
    else:
        severity = "Mild"

    # ── Step 3: Recommendation ────────────────────────────────────
    recommend_output = recommend_chain.invoke({
        "symptoms": symptoms,
        "triage_output": triage_output
    })

    summary     = _extract_field(recommend_output, "SUMMARY")
    warning_raw = _extract_field(recommend_output, "WARNING")
    warning     = None if (not warning_raw or warning_raw.upper() == "NONE") else warning_raw
    actions     = _parse_actions(recommend_output)

    return {
        "valid":            True,
        "invalid_reason":   None,
        "intake":           intake_output,
        "severity":         severity,
        "department":       department or "General Medicine",
        "urgency_score":    urgency_score,
        "triage_reasoning": triage_reasoning,
        "summary":          summary,
        "actions":          actions,
        "warning":          warning,
        "raw_triage":       triage_output,
        "raw_recommend":    recommend_output,
    }


# ── Helpers ───────────────────────────────────────────────────────

def _extract_field(text: str, field: str) -> str:
    for line in text.split("\n"):
        if line.strip().upper().startswith(field.upper() + ":"):
            return line.split(":", 1)[1].strip()
    return ""


def _parse_actions(text: str) -> list:
    actions = []
    in_actions = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.upper().startswith("ACTIONS:"):
            in_actions = True
            continue
        if stripped.upper().startswith("WARNING:"):
            break
        if in_actions and stripped and stripped[0].isdigit():
            action_text = stripped.lstrip("0123456789").lstrip(".").lstrip(")").strip()
            if action_text:
                actions.append(action_text)
    return actions if actions else ["Follow up with a medical professional."]