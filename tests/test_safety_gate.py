"""
tests/test_safety_gate.py
--------------------------
Unit tests for agents.safety_gate. apply_safety_gate() is a pure
function of a result dict - no LLM calls, no DB, no Streamlit - so
these run instantly and don't need GROQ_API_KEY.

Run:
    pytest tests/test_safety_gate.py -v
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.safety_gate import apply_safety_gate, FALLBACK_EMERGENCY_WARNING


def base_result(**overrides):
    result = {
        "valid": True,
        "invalid_reason": None,
        "intake": "some intake text",
        "severity": "Mild",
        "department": "General Medicine",
        "urgency_score": 2,
        "confidence_score": 80,
        "triage_reasoning": "reasoning",
        "summary": "summary",
        "actions": ["rest", "hydrate"],
        "warning": None,
        "guardrail_note": None,
        "raw_triage": "",
        "raw_recommend": "",
    }
    result.update(overrides)
    return result


# ── Non-emergency cases: gate is a no-op ────────────────────────────


def test_mild_case_not_flagged_emergency():
    result = apply_safety_gate(base_result())
    assert result["emergency"] is False
    assert result["safety_gate_fired"] is False
    assert result["safety_gate_reasons"] == []
    assert result["warning"] is None  # untouched


def test_moderate_case_not_flagged_emergency():
    result = apply_safety_gate(
        base_result(severity="Moderate", department="Cardiology", urgency_score=6)
    )
    assert result["emergency"] is False
    assert result["safety_gate_fired"] is False


def test_invalid_case_is_not_flagged_emergency_and_does_not_crash():
    result = apply_safety_gate(
        base_result(valid=False, severity="Unknown", department="General Medicine", urgency_score=0)
    )
    assert result["emergency"] is False
    assert result["safety_gate_fired"] is False
    assert result["safety_gate_reasons"] == []


# ── Emergency cases: gate must act ──────────────────────────────────


def test_critical_case_flagged_emergency():
    result = apply_safety_gate(
        base_result(severity="Critical", department="Emergency", urgency_score=10, warning="Go to the ER now.")
    )
    assert result["emergency"] is True


def test_department_emergency_alone_counts_as_emergency():
    # Severity could theoretically be inconsistent with department;
    # department == "Emergency" alone is enough to flag it.
    result = apply_safety_gate(
        base_result(severity="Moderate", department="Emergency", urgency_score=9, warning="Seek care now.")
    )
    assert result["emergency"] is True


def test_critical_with_no_warning_gets_fallback_and_gate_fires():
    result = apply_safety_gate(
        base_result(severity="Critical", department="Emergency", urgency_score=10, warning=None)
    )
    assert result["warning"] == FALLBACK_EMERGENCY_WARNING
    assert result["safety_gate_fired"] is True
    assert any("no warning" in r for r in result["safety_gate_reasons"])


def test_critical_with_blank_warning_gets_fallback():
    result = apply_safety_gate(
        base_result(severity="Critical", department="Emergency", urgency_score=10, warning="   ")
    )
    assert result["warning"] == FALLBACK_EMERGENCY_WARNING
    assert result["safety_gate_fired"] is True


def test_critical_with_real_warning_is_not_overwritten():
    original = "Go to the nearest emergency room immediately."
    result = apply_safety_gate(
        base_result(severity="Critical", department="Emergency", urgency_score=10, warning=original)
    )
    assert result["warning"] == original
    # gate still didn't need to act on the warning
    assert not any("warning" in r for r in result["safety_gate_reasons"])


def test_critical_with_low_urgency_is_raised_to_floor():
    result = apply_safety_gate(
        base_result(severity="Critical", department="Emergency", urgency_score=4, warning="Go now.")
    )
    assert result["urgency_score"] == 9
    assert result["safety_gate_fired"] is True
    assert any("urgency_score" in r for r in result["safety_gate_reasons"])


def test_critical_urgency_already_at_floor_is_untouched():
    result = apply_safety_gate(
        base_result(severity="Critical", department="Emergency", urgency_score=9, warning="Go now.")
    )
    assert result["urgency_score"] == 9
    assert not any("urgency_score" in r for r in result["safety_gate_reasons"])


def test_gate_never_downgrades_urgency():
    # Sanity check on the "only ever escalates" contract: a
    # higher-than-floor urgency must never be lowered by the gate.
    result = apply_safety_gate(
        base_result(severity="Critical", department="Emergency", urgency_score=10, warning="Go now.")
    )
    assert result["urgency_score"] == 10
