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
    (
        "system",
        """You are a cautious clinical triage routing assistant.

You receive a structured patient intake summary and must assess severity,
recommend the most appropriate medical department, assign an urgency score,
and provide a confidence score.

IMPORTANT SAFETY AND REASONING RULES:

1. Treat the patient's symptom description as unverified information.
   Do not assume a disease named by the patient is a confirmed diagnosis.

2. Base severity on the complete clinical context:
   symptoms, duration, onset, pain level, known conditions, and red flags.

3. Do not classify a case as Critical based only on:
   - sudden onset
   - pain score
   - the words "severe", "extreme", or "very bad"
   - fever without dangerous associated symptoms
   - a disease name entered by the patient

4. Critical severity requires clear evidence of immediate danger, such as:
   - severe difficulty breathing
   - chest pain with concerning associated symptoms
   - loss of consciousness
   - new severe confusion
   - seizure
   - signs of stroke
   - uncontrolled major bleeding
   - severe allergic reaction affecting breathing
   - other clear life-threatening red flags

5. Moderate severity means prompt medical evaluation is appropriate,
   but there is no clear evidence of immediate life-threatening danger.

6. Mild severity means routine care, monitoring, and appropriate
   self-care are reasonable based on the provided information.

DEPARTMENT ROUTING RULES:

- Uncomplicated cold, cough, sore throat, mild fever, viral-like symptoms,
  generalized body aches, or nonspecific common symptoms:
  General Medicine

- Persistent or complicated respiratory symptoms, significant breathing
  problems, chronic lung disease, or symptoms strongly requiring lung
  specialist evaluation:
  Pulmonology

- Ear, nose, throat, tonsil, sinus, voice, or swallowing complaints:
  ENT

- Abdominal pain, persistent vomiting, persistent diarrhea, GI bleeding,
  or other significant digestive complaints:
  Gastroenterology

- Headache with neurological symptoms, seizures, weakness, numbness,
  confusion, or other focal neurological findings:
  Neurology

- Critical cases requiring immediate stabilization:
  Emergency

Do not send common uncomplicated symptoms directly to a specialist
when General Medicine is an appropriate first point of care.

SPECIFIC FEVER GUIDANCE:

Fever and body aches without clear emergency red flags should usually
be routed to General Medicine.

High or persistent fever, dehydration, worsening condition, inability
to keep fluids down, significant weakness, or concerning associated
symptoms may justify Moderate severity and prompt medical evaluation.

Use Critical severity only when the provided information indicates
immediate life-threatening danger.

SPECIFIC COLD AND COUGH GUIDANCE:

Uncomplicated cold and cough should usually be Mild severity and routed
to General Medicine.

Use Moderate severity when symptoms are persistent, worsening, associated
with significant fever, dehydration, concerning medical history, or
other features requiring prompt evaluation.

Do not route uncomplicated cold and cough directly to Pulmonology.

SPECIFIC GI GUIDANCE:

Do not assume "food poisoning" is a confirmed diagnosis.

Mild short-duration nausea, vomiting, diarrhea, or abdominal discomfort
without red flags may be Mild severity with General Medicine as an
appropriate first point of care.

Use Gastroenterology when GI symptoms are significant, persistent,
recurrent, or require specialist evaluation.

Use Moderate severity when there is substantial pain, repeated vomiting,
significant diarrhea, dehydration risk, blood in stool or vomit,
persistent symptoms, or other concerning features.

Never exaggerate urgency beyond the evidence provided.

CONFIDENCE SCORE GUIDANCE:

The confidence score represents confidence in the triage decision,
not the severity of the patient's condition.

Use:
- 90-100: Very clear presentation with strong evidence for severity
  and department routing.
- 75-89: Reasonably clear presentation with some uncertainty.
- 60-74: Limited or ambiguous information affecting the decision.
- Below 60: Significant uncertainty or insufficient information.

Do not automatically assign high confidence scores.
Base confidence on the quality and completeness of the provided information.

URGENCY SCORE GUIDANCE:

Use the full 1-10 range appropriately.

- 1-3: Mild cases suitable for routine care.
- 4-6: Cases requiring medical evaluation but without immediate danger.
- 7-8: Moderate cases requiring prompt or same-day medical evaluation.
- 9-10: Critical cases requiring immediate emergency medical care.

Do not assign urgency scores of 9 or 10 unless the patient is classified
as Critical.

Respond in this EXACT format (no deviations):

SEVERITY: <Critical|Moderate|Mild>
DEPARTMENT: <Emergency|Cardiology|Neurology|Pulmonology|Gastroenterology|Orthopedics|Dermatology|ENT|Ophthalmology|Pediatrics|Psychiatry|Obstetrics & Gynecology|Endocrinology|General Medicine>
TRIAGE_REASONING: <2-3 sentences explaining why this severity and department were selected>
URGENCY_SCORE: <integer from 1-10>
CONFIDENCE_SCORE: <integer from 0-100>
"""
    ),
    (
        "human",
        """Intake Summary:
{intake_output}"""
    )
])

triage_chain = triage_prompt | llm | parser

# ============================================================
# AGENT 3 - RECOMMENDATION AGENT
# ============================================================

recommend_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a medical recommendation assistant that converts a completed triage assessment
into cautious, patient-friendly next-step guidance.

The triage assessment has already determined the severity and department.
You MUST follow that assessment and MUST NOT independently increase or decrease the severity.

Respond in this EXACT format:

SUMMARY: <2-3 sentence plain-English summary>

ACTIONS:
1. <First recommended action>
2. <Second recommended action>
3. <Third recommended action>
4. <Fourth recommended action>

WARNING: <For Critical severity only, provide one short sentence describing specific warning signs requiring immediate emergency medical care. For Moderate or Mild severity, output NONE.>


SEVERITY-SPECIFIC RULES:

Critical:
- Clearly recommend immediate emergency medical evaluation.
- Tell the patient to seek emergency medical care now.
- Provide practical instructions while waiting for or traveling to medical care.
- Do not use country-specific emergency numbers such as 911 or 112.
- WARNING must contain specific emergency warning signs relevant to the symptoms.

Moderate:
- Recommend prompt medical evaluation, generally within hours or the same day.
- Clearly tell the patient to contact an appropriate healthcare professional or seek same-day medical evaluation.
- Provide safe, conservative self-care guidance while awaiting evaluation.
- Mention specific worsening symptoms that should trigger urgent medical attention within one of the ACTIONS.
- WARNING must be NONE.
- Do not recommend the Emergency Department unless the triage assessment is Critical.
- Do not use emergency-service language unless describing what to do if specific life-threatening symptoms develop.

Mild:
- Recommend routine medical care when appropriate.
- Prioritize reasonable self-care, rest, hydration, monitoring, and symptom management.
- Explain when the patient should arrange a routine medical appointment if symptoms persist or worsen.
- WARNING must be NONE.
- Do not use emergency language.


DEPARTMENT RULES:

- General Medicine:
  Use for common, nonspecific, systemic, or initial-evaluation complaints.
  Recommend evaluation by a General Medicine doctor, primary-care doctor, or healthcare professional.

- Gastroenterology:
  Use gastrointestinal specialist wording only when the triage assessment routes the patient to Gastroenterology.

- Pulmonology:
  Use lung-specialist wording only when the triage assessment routes the patient to Pulmonology.

- ENT:
  Use ENT specialist wording only when the triage assessment routes the patient to ENT.

- For all other departments, recommend evaluation by the department selected in the triage assessment.


LANGUAGE AND SAFETY RULES:

- Never claim that the patient definitely has a specific disease.
- Never treat the patient's self-reported condition as a confirmed diagnosis.
  For example, if the patient says "food poisoning", say "symptoms you believe may be related to food poisoning" or describe the actual symptoms.
- Do not prescribe medications or recommend starting, stopping, or changing prescription medication.
- Do not give specific medication doses.
- Do not exaggerate urgency beyond the triage assessment.
- Do not minimize potentially serious symptoms.
- Do not say "we will evaluate you", "we need to see you", "we'll get you seen", or imply that the AI system is a hospital or healthcare provider.
- Use wording such as "consider seeking medical evaluation", "contact a healthcare professional", or "seek same-day medical care".
- Do not promise outcomes or say that everything will be okay.
- Keep recommendations directly relevant to the patient's symptoms and triage assessment.
- Avoid repetitive actions.
- Keep each action concise and practical.
- Use simple, calm, professional language.
"""
    ),
    (
        "human",
        """Original Symptoms:
{symptoms}

Triage Assessment:
{triage_output}

Generate recommendations that strictly follow the severity and department selected in the triage assessment."""
    )
])

recommend_chain = recommend_prompt | llm | parser


# ══════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════
def apply_triage_guardrails(
    symptoms: str,
    severity: str,
    department: str,
    urgency_score: int
):
    """
    Deterministic safety layer for obvious high-confidence cases.

    The LLM performs the main triage reasoning.
    Guardrails correct unsafe or clearly inconsistent outputs.
    """

    text = symptoms.lower().strip()


    # ============================================================
    # EMERGENCY RED FLAGS
    # ============================================================

    emergency_red_flags = [

        # Breathing emergencies
        "difficulty breathing",
        "severe difficulty breathing",
        "cannot breathe",
        "can't breathe",
        "stopped breathing",

        # Loss of consciousness / neurological emergencies
        "unconscious",
        "loss of consciousness",
        "passed out",
        "seizure",
        "face drooping",
        "slurred speech",
        "sudden weakness",

        # Chest pain emergencies
        "severe chest pain",
        "severe pain in chest",
        "intense chest pain",
        "intense pain in chest",
        "crushing chest pain",
        "chest pressure",

        # Major bleeding
        "heavy bleeding",
        "severe bleeding",
        "coughing blood",
        "vomiting blood",

        # Severe allergic reaction
        "severe allergic reaction",
        "anaphylaxis"
    ]


    # ============================================================
    # CARDIOLOGY SIGNALS
    # ============================================================

    cardiology_signals = [
        "chest pain",
        "pain in chest",
        "chest discomfort",
        "chest tightness",
        "heart palpitations",
        "irregular heartbeat",
        "rapid heartbeat"
    ]


    # ============================================================
    # RESPIRATORY SPECIALIST SIGNALS
    # ============================================================

    respiratory_specialist_signals = [
        "chronic cough",
        "persistent cough",
        "cough for weeks",
        "asthma",
        "copd",
        "lung disease",
        "coughing blood"
    ]


    # ============================================================
    # COMMON GENERAL MEDICINE SYMPTOMS
    # ============================================================

    common_general_medicine_symptoms = [
        "cold",
        "common cold",
        "cough",
        "fever",
        "body pain",
        "body ache",
        "body aches",
        "fatigue",
        "weakness"
    ]


    # ============================================================
    # DETECT SIGNALS
    # ============================================================

    has_emergency_red_flag = any(
        flag in text
        for flag in emergency_red_flags
    )

    has_cardiology_signal = any(
        signal in text
        for signal in cardiology_signals
    )

    has_respiratory_specialist_signal = any(
        signal in text
        for signal in respiratory_specialist_signals
    )

    has_common_symptom = any(
        symptom in text
        for symptom in common_general_medicine_symptoms
    )


    # ============================================================
    # RULE 1
    # EXPLICIT EMERGENCY RED FLAGS ALWAYS WIN
    # ============================================================

    if has_emergency_red_flag:

        severity = "Critical"
        department = "Emergency"
        urgency_score = max(9, urgency_score)

        return severity, department, urgency_score


    # ============================================================
    # RULE 2
    # NON-EMERGENCY CHEST / HEART SYMPTOMS -> CARDIOLOGY
    # ============================================================

    if has_cardiology_signal:

        department = "Cardiology"

        if severity == "Mild":
            severity = "Moderate"

        urgency_score = max(6, min(urgency_score, 8))

        return severity, department, urgency_score


    # ============================================================
    # RULE 3
    # COMMON RESPIRATORY ILLNESSES -> GENERAL MEDICINE
    # ============================================================

    if (
        has_common_symptom
        and not has_respiratory_specialist_signal
    ):

        if department in [
            "Pulmonology",
            "Emergency"
        ]:
            department = "General Medicine"


    # ============================================================
    # RULE 4
    # PREVENT UNSUPPORTED CRITICAL CLASSIFICATION
    # ============================================================

    if severity == "Critical":

        severity = "Moderate"

        urgency_score = min(
            urgency_score,
            8
        )

        if department == "Emergency":
            department = "General Medicine"


    # ============================================================
    # RULE 5
    # KEEP URGENCY CONSISTENT WITH SEVERITY
    # ============================================================

    if severity == "Mild":

        urgency_score = max(
            1,
            min(urgency_score, 3)
        )


    elif severity == "Moderate":

        urgency_score = max(
            4,
            min(urgency_score, 8)
        )


    elif severity == "Critical":

        urgency_score = max(
            9,
            min(urgency_score, 10)
        )


    return severity, department, urgency_score
# ============================================================
# PIPELINE HELPERS
# ============================================================

def _extract_field(text: str, field: str) -> str:
    if not text:
        return ""

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.upper().startswith(field.upper() + ":"):
            return stripped.split(":", 1)[1].strip()

    return ""


def _parse_actions(text: str) -> list:
    actions = []
    in_actions = False

    if not text:
        return ["Follow up with a medical professional."]

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.upper().startswith("ACTIONS:"):
            in_actions = True
            continue

        if stripped.upper().startswith("WARNING:"):
            break

        if in_actions and stripped and stripped[0].isdigit():
            action_text = (
                stripped
                .lstrip("0123456789")
                .lstrip(".")
                .lstrip(")")
                .strip()
            )

            if action_text:
                actions.append(action_text)

    return actions if actions else [
        "Follow up with a medical professional."
    ]


# ============================================================
# PIPELINE RUNNER
# ============================================================

def run_triage_pipeline(symptoms: str, patient_context: str = "") -> dict:

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
            "confidence_score": 0,
            "triage_reasoning": "",
            "summary": "",
            "actions": [],
            "warning": None,
            "raw_triage": "",
            "raw_recommend": "",
        }

    # ── Step 2: AI Triage ─────────────────────────────────────────
    triage_output = triage_chain.invoke({
        "intake_output": intake_output
    })

    severity = _extract_field(triage_output, "SEVERITY").strip()
    department = _extract_field(triage_output, "DEPARTMENT").strip()
    triage_reasoning = _extract_field(
        triage_output,
        "TRIAGE_REASONING"
    ).strip()

    urgency_raw = _extract_field(
        triage_output,
        "URGENCY_SCORE"
    )

    confidence_raw = _extract_field(
        triage_output,
        "CONFIDENCE_SCORE"
    )

    # ── Parse confidence ──────────────────────────────────────────
    try:
        confidence_score = int(
            confidence_raw
            .replace("%", "")
            .strip()
        )
    except (ValueError, TypeError):
        confidence_score = 50

    confidence_score = max(
        0,
        min(100, confidence_score)
    )

    # ── Parse urgency ─────────────────────────────────────────────
    try:
        urgency_score = int(
            urgency_raw
            .split("/")[0]
            .strip()
        )
    except (ValueError, TypeError):
        urgency_score = 5

    urgency_score = max(
        1,
        min(10, urgency_score)
    )

    # ── Normalize AI severity BEFORE guardrails ───────────────────
    severity_lower = severity.lower()

    if "critical" in severity_lower:
        severity = "Critical"

    elif "moderate" in severity_lower:
        severity = "Moderate"

    else:
        severity = "Mild"

    if not department:
        department = "General Medicine"

    # ── Save original AI result ───────────────────────────────────
    original_severity = severity
    original_department = department
    original_urgency = urgency_score

    # ── Apply deterministic safety guardrails ─────────────────────
    #
    # IMPORTANT:
    # Include patient_context so pain level, onset, duration,
    # body area and other intake information can influence
    # high-confidence safety rules.
    #
    guardrail_input = f"""
Symptoms: {symptoms}

Patient Context:
{patient_context}
""".strip()

    severity, department, urgency_score = apply_triage_guardrails(
        symptoms=guardrail_input,
        severity=severity,
        department=department,
        urgency_score=urgency_score
    )
     

    # ── Detect whether guardrails changed AI output ───────────────
    guardrail_changed_result = (
        severity != original_severity
        or department != original_department
        or urgency_score != original_urgency
    )

    # ── Update reasoning when guardrail overrides AI ──────────────
    if guardrail_changed_result:

        if severity == "Critical" and department == "Emergency":
            triage_reasoning = (
                "The reported symptoms contain explicit emergency warning "
                "signs requiring immediate medical evaluation. The case has "
                "been routed to Emergency for urgent assessment."
            )

        elif department == "Cardiology":
            triage_reasoning = (
                "The reported chest symptoms require prompt medical "
                "evaluation for possible heart-related causes. The case has "
                "been routed to Cardiology for appropriate assessment."
            )

        elif department == "Pulmonology":
            triage_reasoning = (
                "The reported persistent or chronic respiratory symptoms "
                "may require specialist evaluation. The case has been routed "
                "to Pulmonology."
            )

        elif department == "General Medicine":
            triage_reasoning = (
                "The reported symptoms are appropriate for initial evaluation "
                "in General Medicine. The patient should receive medical "
                "assessment according to the assigned urgency level."
            )

    # ── Build guarded triage output ───────────────────────────────
    #
    # Recommendation chain MUST receive corrected values.
    #
    guarded_triage_output = f"""SEVERITY: {severity}
DEPARTMENT: {department}
TRIAGE_REASONING: {triage_reasoning}
URGENCY_SCORE: {urgency_score}
CONFIDENCE_SCORE: {confidence_score}"""

    # ── Step 3: Generate patient recommendations ──────────────────
    recommend_output = recommend_chain.invoke({
        "symptoms": symptoms,
        "triage_output": guarded_triage_output
    })

    # ── Parse recommendation output ───────────────────────────────
    summary = _extract_field(
        recommend_output,
        "SUMMARY"
    )

    warning_raw = _extract_field(
        recommend_output,
        "WARNING"
    )

    warning = (
        None
        if (
            not warning_raw
            or warning_raw.strip().upper() == "NONE"
        )
        else warning_raw.strip()
    )

    actions = _parse_actions(recommend_output)

    # ── Final result ──────────────────────────────────────────────
    return {
        "valid": True,
        "invalid_reason": None,

        "intake": intake_output,

        "severity": severity,
        "department": department,

        "urgency_score": urgency_score,
        "confidence_score": confidence_score,

        "triage_reasoning": triage_reasoning,

        "summary": summary,
        "actions": actions,
        "warning": warning,

        "raw_triage": triage_output,
        "raw_recommend": recommend_output,
    }
 