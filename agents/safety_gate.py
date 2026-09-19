"""
agents/safety_gate.py
----------------------
The Emergency Safety Gate: a deterministic layer that runs on the
*final* pipeline output, after the Intake, Triage and Recommendation
agents have all run.

This is deliberately separate from apply_triage_guardrails() in
agents/pipeline.py. That earlier layer corrects severity/department
*before* the Recommendation agent sees them - it decides what the
case IS. This layer instead checks what the pipeline is about to SHIP
to the patient, and refuses to let a Critical case leave the pipeline
without an unmistakable warning attached, no matter which agent (or
absence of one) was responsible for that gap.

Two failure modes this specifically closes:

  1. Severity/department say this is an emergency, but the
     Recommendation agent's own LLM call happened not to produce a
     WARNING field (it's free-text generation - it can just forget).
     Without this gate, `warning` would silently be None on a
     Critical case, and the UI's "make emergency results visually
     unmistakable" has nothing to render.

  2. Some later change to the pipeline reorders steps, or a new code
     path builds a result dict without going through
     apply_triage_guardrails at all. The urgency-score invariant
     (Critical => 9 or 10) would then depend entirely on that one
     earlier function being called correctly. This gate re-checks the
     invariant independently, right before the result is returned, so
     a bug upstream degrades safely instead of degrading silently.

This gate only ever escalates (adds a flag, adds a warning, raises an
under-scored urgency number for a Critical case). It never downgrades
severity, department or urgency - that direction of correction stays
in apply_triage_guardrails, which runs earlier and has the fuller
symptom-text context to reason about false positives.

Usage:
    from agents.safety_gate import apply_safety_gate
    result = apply_safety_gate(result, symptoms=symptoms)
"""

import logging
import os

# ── Logging setup ────────────────────────────────────────────────
#
# Every time the gate changes or adds something, that's logged here -
# this is the audit trail called for by the roadmap ("log the reason
# why the safety gate fired"). Deliberately a plain rotating-by-date
# text log rather than the app's SQLite DB: this needs to keep working
# even if the DB layer is the thing that's broken, and it should never
# contain patient-identifying symptom text (see _fire below).

_LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)
os.makedirs(_LOG_DIR, exist_ok=True)

_logger = logging.getLogger("mediagent.safety_gate")
if not _logger.handlers:
    _handler = logging.FileHandler(
        os.path.join(_LOG_DIR, "safety_gate.log"), encoding="utf-8"
    )
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(message)s")
    )
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)


FALLBACK_EMERGENCY_WARNING = (
    "⚠️ This case has been flagged as a potential medical emergency. "
    "Seek immediate in-person medical care or call your local emergency "
    "number - do not wait for a routine appointment."
)

# Mirrors the invariant apply_triage_guardrails is supposed to enforce
# (see agents/pipeline.py:check_internal_consistency /
# evaluation/evaluate_triage.py). Re-checked here independently.
MIN_CRITICAL_URGENCY = 9


def _is_emergency(severity: str, department: str) -> bool:
    return severity == "Critical" or department == "Emergency"


def _fire(reasons, reason: str, severity: str, department: str, urgency_score):
    """Records one thing the gate changed, both in-memory (for the
    caller/UI) and to the audit log. Logs only the structured fields
    already on the result (severity/department/urgency) - never raw
    symptom text - so the log is safe to keep without treating it as
    patient data."""

    reasons.append(reason)
    _logger.info(
        "FIRED reason=%r severity=%s department=%s urgency_score=%s",
        reason,
        severity,
        department,
        urgency_score,
    )


def apply_safety_gate(result: dict) -> dict:
    """Takes a run_triage_pipeline() result dict and returns it with
    the gate applied. Adds:

        result["emergency"]            bool
        result["safety_gate_fired"]    bool
        result["safety_gate_reasons"]  list[str]  (empty if it didn't fire)

    No-ops (besides setting emergency=False and the two fields above)
    on invalid/non-emergency cases.
    """

    if not result.get("valid", True):
        result["emergency"] = False
        result["safety_gate_fired"] = False
        result["safety_gate_reasons"] = []
        return result

    severity = result.get("severity")
    department = result.get("department")
    urgency_score = result.get("urgency_score")
    warning = result.get("warning")

    emergency = _is_emergency(severity, department)
    reasons = []

    if emergency:
        # 1. Never ship a Critical case with no warning attached.
        if not warning or not str(warning).strip():
            result["warning"] = FALLBACK_EMERGENCY_WARNING
            _fire(
                reasons,
                "recommendation agent produced no warning for an emergency "
                "case; applied fallback emergency warning",
                severity,
                department,
                urgency_score,
            )

        # 2. Re-check the urgency-score invariant independently of
        #    apply_triage_guardrails. Never lowers urgency - only
        #    raises it if it's under the emergency floor.
        if not isinstance(urgency_score, int) or urgency_score < MIN_CRITICAL_URGENCY:
            _fire(
                reasons,
                f"urgency_score {urgency_score!r} below the emergency floor "
                f"of {MIN_CRITICAL_URGENCY} for a Critical/Emergency case; "
                f"raised to {MIN_CRITICAL_URGENCY}",
                severity,
                department,
                urgency_score,
            )
            result["urgency_score"] = MIN_CRITICAL_URGENCY

    result["emergency"] = emergency
    result["safety_gate_fired"] = bool(reasons)
    result["safety_gate_reasons"] = reasons

    return result
