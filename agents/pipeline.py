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

import re
from agents.safety_gate import apply_safety_gate
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ─────────────────────────────────────────────────────────
# GROQ_MODEL lets you override the model per-run (e.g. for evaluation)
# without touching production. Falls back to the oss-20b model used
# in the deployed app if the env var isn't set.
# ── LLM ─────────────────────────────────────────────────────────
# GROQ_MODEL lets you override the model per-run (e.g. for evaluation)
# without touching production — BUT only to a model we've verified is
# actually reachable on this account's tier. Groq periodically moves
# models to Enterprise-only access (this broke llama-3.3-70b-versatile
# and llama-3.1-8b-instant during this project's development) — if that
# happens to whatever GROQ_MODEL is set to (env var, stray secret, etc.),
# falling back automatically here is what keeps production from going
# down instead of 404ing on every request.
_KNOWN_GOOD_MODELS = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
_requested_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_MODEL = _requested_model if _requested_model in _KNOWN_GOOD_MODELS else "openai/gpt-oss-20b"

_llm_kwargs = {
    "model": GROQ_MODEL,
    "temperature": 0.2,
    "api_key": os.getenv("GROQ_API_KEY"),
}

# reasoning_effort is an openai/gpt-oss-specific param — only pass it
# when that family of model is actually selected, otherwise Llama/other
# models on Groq will reject the request.
if GROQ_MODEL.startswith("openai/gpt-oss"):
    _llm_kwargs["model_kwargs"] = {"reasoning_effort": "low"}

llm = ChatGroq(**_llm_kwargs)

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
   - signs of a surgical abdominal emergency (rigid or board-like
     abdomen, severe pain that worsens with movement or palpation,
     suspected appendicitis or bowel perforation)
   - sudden, painless loss of vision in one eye (e.g. described as a
     curtain or shadow coming down) — a time-critical ophthalmological
     or vascular emergency
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
# DETERMINISTIC RED-FLAG PATTERNS
#
# Regex, not plain substring matching. The original literal-phrase list
# (e.g. "loss of consciousness", "coughing blood") silently missed real
# clinical language that used a different inflection or inserted a word
# ("lost consciousness", "coughing up blood") — the phrase was clinically
# present but the exact string wasn't, so Rule 1 never fired and Rule 4
# downgraded a correct LLM "Critical" call to "Moderate".
#
# Each pattern below is still scoped tightly to a single clinical red
# flag (word-boundaried, case-insensitive) so it doesn't start catching
# unrelated mild cases (e.g. "chest" alone must never match — chest pain
# patterns explicitly require the qualifier + "chest" together).
# ══════════════════════════════════════════════════════════════════

EMERGENCY_RED_FLAG_PATTERNS = [
    # Breathing emergencies
    r"\bsevere\s+difficulty\s+breathing\b",
    r"\bdifficulty\s+breathing\b",
    r"\bcan(?:no|')t\s+breathe\b",
    r"\bcannot\s+breathe\b",
    r"\bstopped\s+breathing\b",
    r"\btrouble\s+breathing\b",
    r"\bshort(?:ness)?\s+of\s+breath\b",
    r"\bgasping\s+for\s+(?:air|breath)\b",

    # Loss of consciousness / neurological emergencies
    r"\bloss\s+of\s+consciousness\b",
    r"\blost\s+consciousness\b",
    r"\bunconscious\b",
    r"\bpassed\s+out\b",
    r"\bfainted\b",
    r"\bunresponsive\b",
    r"\bseizure\b",
    r"\bconvuls(?:ing|ion)\b",
    r"\bface\s+drooping\b",
    r"\bfacial\s+droop\b",
    r"\bslurred\s+speech\b",
    r"\bsudden\s+weakness\b",
    r"\bone[-\s]sided\s+weakness\b",
    r"\btrouble\s+finding\s+words\b",

    # Chest pain emergencies — "chest" is always mandatory in the pattern,
    # so this never matches generic "severe pain" for another body part.
    r"\bsevere\s+chest\s+pain\b",
    r"\bsevere\s+pain\s+in\s+(?:the\s+)?chest\b",
    r"\bintense\s+chest\s+pain\b",
    r"\bintense\s+pain\s+in\s+(?:the\s+)?chest\b",
    r"\bcrushing\s+chest\s+pain\b",
    r"\bcrushing\s+(?:pressure|pain)\s+in\s+(?:the\s+)?chest\b",
    r"\bchest\s+pressure\b",

    # Major bleeding
    r"\bheavy\s+bleeding\b",
    r"\bsevere\s+bleeding\b",
    r"\bbleeding\s+heavily\b",
    r"\bcoughing\s+(?:up\s+)?blood\b",
    r"\bvomit(?:ing|ed)\s+blood\b",

    # Severe allergic reaction
    r"\bsevere\s+allergic\s+reaction\b",
    r"\banaphylaxis\b",
    r"\bthroat\s+(?:closing|tightening)\b",
]

_COMPILED_RED_FLAGS = [re.compile(p, re.IGNORECASE) for p in EMERGENCY_RED_FLAG_PATTERNS]

# Words that negate whatever red-flag phrase follows them shortly after.
# Without this, "no shortness of breath" or "no heavy bleeding" would be
# treated identically to an actual positive report of that symptom —
# a real false-positive bug found via CARD-201 and OBGYN-1401, where
# patients explicitly denying a symptom still triggered Critical/Emergency.
_NEGATION_CUES = re.compile(
    r"\b(no|not|without|denies|denied|absent|negative for)\b",
    re.IGNORECASE,
)
_NEGATION_LOOKBACK_CHARS = 20


def _has_emergency_red_flag(text: str) -> bool:
    for pattern in _COMPILED_RED_FLAGS:
        for match in pattern.finditer(text):
            window_start = max(0, match.start() - _NEGATION_LOOKBACK_CHARS)
            preceding_text = text[window_start:match.start()]
            # Restrict to the current clause only. Without this, a negation
            # word from an EARLIER, unrelated clause within the raw character
            # window (e.g. "no nausea" in "no fever, no nausea, but severe
            # chest pain...") would wrongly suppress a real red flag that
            # comes right after it in a new clause.
            last_boundary = max(
                preceding_text.rfind(","),
                preceding_text.rfind(";"),
                preceding_text.rfind("."),
            )
            if last_boundary != -1:
                preceding_text = preceding_text[last_boundary + 1:]
            if _NEGATION_CUES.search(preceding_text):
                # e.g. "no shortness of breath" — the phrase is present
                # but explicitly denied, so it isn't a red flag.
                continue
            return True
    return False


# The confidence score (0-100) below which an LLM Critical call is
# treated as unreliable enough to downgrade. Above this, an LLM Critical
# call is trusted even without a matching red-flag phrase. See Rule 4.
LOW_CONFIDENCE_THRESHOLD = 60


# ══════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════
def apply_triage_guardrails(
    symptoms: str,
    severity: str,
    department: str,
    urgency_score: int,
    confidence_score: int = None,
):
    """
    Deterministic safety layer for obvious high-confidence cases.

    The LLM performs the main triage reasoning.
    Guardrails correct unsafe or clearly inconsistent outputs.

    Returns (severity, department, urgency_score, guardrail_note).
    guardrail_note is a short string describing *why* the guardrail
    acted, or None if it didn't change anything. This makes every
    guardrail decision auditable instead of silent.
    """

    text = symptoms.lower().strip()
    guardrail_note = None


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

    has_emergency_red_flag = _has_emergency_red_flag(text)

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
        guardrail_note = (
            "Matched an explicit emergency red-flag phrase in the "
            "patient-reported symptoms — routed to Emergency regardless "
            "of the LLM's own classification."
        )

        return severity, department, urgency_score, guardrail_note


    # ============================================================
    # RULE 2
    # NON-EMERGENCY CHEST / HEART SYMPTOMS -> CARDIOLOGY
    # ============================================================

    if has_cardiology_signal:

        department = "Cardiology"

        if severity == "Mild":
            severity = "Moderate"

        urgency_score = max(6, min(urgency_score, 8))

        return severity, department, urgency_score, guardrail_note


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
    # RULE 4 — TRUST-BY-DEFAULT (INVERTED)
    #
    # Reaching this point means: no hardcoded red-flag phrase matched
    # (Rule 1 already returned early if one had), but the LLM still
    # classified this case as Critical on its own judgment.
    #
    # OLD BEHAVIOUR (removed): silently downgrade every such case to
    # Moderate. That treats "no phrase matched" as proof the LLM is
    # wrong, which punishes correct judgment on real emergencies that
    # simply weren't phrased using one of ~30 hardcoded strings
    # (e.g. a thunderclap headache, a rigid acute abdomen, early
    # anaphylaxis described without the word "anaphylaxis").
    #
    # NEW BEHAVIOUR: trust the LLM's Critical call by default. Only
    # downgrade when there is a SPECIFIC, NAMED reason to distrust this
    # particular call — currently: the LLM's own confidence score came
    # back below LOW_CONFIDENCE_THRESHOLD. Every downgrade or trust
    # decision is logged in guardrail_note instead of happening silently.
    # ============================================================

    if severity == "Critical":

        if confidence_score is not None and confidence_score < LOW_CONFIDENCE_THRESHOLD:
            severity = "Moderate"
            urgency_score = min(urgency_score, 8)

            if department == "Emergency":
                department = "General Medicine"

            guardrail_note = (
                f"LLM classified this as Critical, but its own confidence "
                f"score ({confidence_score}) was below the "
                f"{LOW_CONFIDENCE_THRESHOLD} reliability threshold and no "
                f"red-flag phrase confirmed it — downgraded to Moderate "
                f"pending clinician review."
            )

        else:
            department = "Emergency"

            guardrail_note = (
                "LLM classified this as Critical with reasonable "
                "confidence, though no hardcoded red-flag phrase matched "
                "the wording used. Trusted and routed to Emergency; "
                "flagged for review since it bypassed the deterministic "
                "phrase list."
            )


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


    return severity, department, urgency_score, guardrail_note
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

        return apply_safety_gate({
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
            "guardrail_note": None,
            "raw_triage": "",
            "raw_recommend": "",
        })

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
    # high-confidence safety rules. confidence_score is passed so
    # Rule 4 can decide whether to trust or downgrade an LLM Critical
    # call instead of downgrading it unconditionally.
    #
    guardrail_input = f"""
Symptoms: {symptoms}

Patient Context:
{patient_context}
""".strip()

    severity, department, urgency_score, guardrail_note = apply_triage_guardrails(
        symptoms=guardrail_input,
        severity=severity,
        department=department,
        urgency_score=urgency_score,
        confidence_score=confidence_score,
    )


    # ── Detect whether guardrails changed AI output ───────────────
    guardrail_changed_result = (
        severity != original_severity
        or department != original_department
        or urgency_score != original_urgency
    )

    # ── Update reasoning when guardrail overrides AI ──────────────
    #
    # guardrail_note (set by Rule 1 or Rule 4) is the most specific,
    # audit-friendly explanation available and takes priority. Rules 2
    # and 3 don't set a note (their behaviour is unambiguous from the
    # department alone), so those fall back to the generic messages
    # below, same as before.
    #
    if guardrail_note:
        triage_reasoning = guardrail_note

    elif guardrail_changed_result:

        if department == "Cardiology":
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
    result = {
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

        # New: exposes *why* the deterministic guardrail acted (or None
        # if it didn't), so this is auditable instead of a black box.
        # Useful for the Doctor Portal / observability work later.
        "guardrail_note": guardrail_note,

        "raw_triage": triage_output,
        "raw_recommend": recommend_output,
    }

    # ── Final structural safety check ───────────────────────────────
    # Runs after every agent, independent of apply_triage_guardrails
    # above. Guarantees an emergency case always carries an explicit
    # `emergency` flag and a non-empty warning, and logs if it had to
    # step in. See agents/safety_gate.py.
    return apply_safety_gate(result)
