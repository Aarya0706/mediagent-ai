"""
evaluation/evaluate_triage.py
------------------------------
Runs the curated case set in evaluation/triage_cases.json through the real
run_triage_pipeline() and reports:

  1. Invalid-input handling      -- does the Intake Agent correctly reject
                                     gibberish/non-medical input?
  2. Emergency detection          -- recall / precision / F1 for
                                     severity == "Critical" vs. the
                                     clinically expected label.
  3. Severity accuracy            -- 3-class exact-match rate (Critical /
                                     Moderate / Mild), among valid cases.
  4. Department accuracy          -- exact-match rate against the primary
                                     expected department, and a looser
                                     "acceptable set" match rate.
  5. Urgency-band accuracy        -- does the returned urgency_score fall
                                     inside the clinically expected range
                                     for that case?
  6. Internal consistency         -- does the pipeline's own output obey
                                     its own stated invariants regardless
                                     of clinical correctness? i.e. Critical
                                     => department == "Emergency" and
                                     urgency in [9,10]; Moderate => urgency
                                     in [4,8]; Mild => urgency in [1,3].
                                     This catches guardrail logic bugs
                                     (see agents/pipeline.py:apply_triage_guardrails)
                                     independently of whether the *clinical*
                                     call was correct.
  7. Latency                      -- per-case wall-clock time (3 LLM calls
                                     per case: intake, triage, recommend).

Usage:
    python evaluation/evaluate_triage.py
    python evaluation/evaluate_triage.py --limit 10          # quick smoke test
    python evaluation/evaluate_triage.py --sleep 1.5          # slower, gentler on rate limits
    python evaluation/evaluate_triage.py --category guardrail_probe
    python evaluation/evaluate_triage.py --dry-run            # validate case file only, no LLM calls

Requires a working GROQ_API_KEY in the environment / .env (same as the app).
Each case costs 3 LLM calls, so the full 60-case set makes ~180 calls --
budget for that against your Groq rate limits (use --sleep and/or --limit).

Outputs (written to evaluation/results/ by default):
    run_<timestamp>.json   -- full per-case results + aggregate metrics
    report_<timestamp>.md  -- human-readable summary + failure examples
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ── Build the exact multi-line symptoms string app.py builds ──────────────
def build_symptoms_string(case: dict) -> str:
    conditions = case.get("conditions") or "None reported"
    return (
        f"Body Part Affected: {case['body_part']}\n"
        f"Symptom Description: {case['description'].strip()}\n"
        f"Duration: {case['duration']}\n"
        f"Onset: {case['onset']}\n"
        f"Pain/Discomfort Level: {case['pain_level']}/10\n"
        f"Known Conditions: {conditions}\n"
        f"Current Medications: {case.get('medications') or 'None reported'}\n"
        f"Allergies: {case.get('allergies') or 'None reported'}"
    ).strip()


def build_patient_context(case: dict) -> str:
    return f"Age: {case['age']}, Gender: {case['gender']}"


# ── Internal-consistency invariant (independent of clinical labels) ───────
def check_internal_consistency(severity: str, department: str, urgency_score) -> bool:
    if severity == "Critical":
        return department == "Emergency" and urgency_score in (9, 10)
    if severity == "Moderate":
        return isinstance(urgency_score, int) and 4 <= urgency_score <= 8
    if severity == "Mild":
        return isinstance(urgency_score, int) and 1 <= urgency_score <= 3
    return False


def run_case(case: dict, sleep_between_calls: float):
    """Runs one case through the real pipeline. Returns a result dict."""
    try:
        from agents.pipeline import run_triage_pipeline  # imported lazily so --dry-run needs no API key
    except ImportError as exc:
        raise SystemExit(
            f"Could not import agents.pipeline ({exc}). "
            "Is your virtual environment activated? Look for '(.venv)' in your prompt."
        )

    symptoms = build_symptoms_string(case)
    patient_context = build_patient_context(case)

    start = time.time()
    error = None
    result = None
    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            result = run_triage_pipeline(symptoms, patient_context)
            error = None
            break
        except Exception as exc:  # noqa: BLE001 - record ANY failure as a case failure, never crash the run
            try:
                error = f"{type(exc).__name__}: {exc}"
            except Exception:
                error = f"{type(exc).__name__} (unprintable error)"
            if "429" in error and attempt < max_retries:
                backoff = 20 * (attempt + 1)
                print(f"\n  rate-limited, backing off {backoff}s (attempt {attempt + 1}/{max_retries})...", end=" ")
                time.sleep(backoff)
                continue
            break
    elapsed = time.time() - start

    if sleep_between_calls:
        time.sleep(sleep_between_calls)

    record = {
        "id": case["id"],
        "category": case["category"],
        "expected_severity": case["expected_severity"],
        "expected_departments": case.get("expected_departments", []),
        "expected_urgency_min": case.get("expected_urgency_min"),
        "expected_urgency_max": case.get("expected_urgency_max"),
        "notes": case.get("notes", ""),
        "latency_sec": round(elapsed, 2),
        "error": error,
    }

    if error is not None:
        record.update({
            "actual_valid": None,
            "actual_severity": None,
            "actual_department": None,
            "actual_urgency": None,
            "actual_confidence": None,
            "internal_consistency": None,
        })
        return record

    record["actual_valid"] = result["valid"]

    if not result["valid"]:
        record.update({
            "actual_severity": "Invalid",
            "actual_department": None,
            "actual_urgency": None,
            "actual_confidence": None,
            "internal_consistency": None,
            "invalid_reason": result.get("invalid_reason"),
        })
        return record

    severity = result["severity"]
    department = result["department"]
    urgency = result["urgency_score"]
    confidence = result.get("confidence_score")

    record.update({
        "actual_severity": severity,
        "actual_department": department,
        "actual_urgency": urgency,
        "actual_confidence": confidence,
        "triage_reasoning": result.get("triage_reasoning", ""),
        "guardrail_note": result.get("guardrail_note"),
        "internal_consistency": check_internal_consistency(severity, department, urgency),
    })
    return record


def score_case(r: dict) -> dict:
    """Adds pass/fail flags to a case record. Assumes no transport error."""
    expected_invalid = r["expected_severity"] == "Invalid"

    scores = {}

    if expected_invalid:
        scores["invalid_handling_correct"] = (r["actual_valid"] is False)
        return {**r, **scores}

    scores["invalid_handling_correct"] = None

    if r["actual_valid"] is False:
        # The pipeline incorrectly rejected a valid clinical case as invalid.
        scores.update({
            "severity_exact_match": False,
            "department_exact_match": False,
            "department_acceptable_match": False,
            "urgency_in_expected_range": False,
            "expected_emergency": r["expected_severity"] == "Critical",
            "predicted_emergency": False,
        })
        return {**r, **scores}

    expected_emergency = r["expected_severity"] == "Critical"
    predicted_emergency = r["actual_severity"] == "Critical"

    dept_exact = r["actual_department"] == (r["expected_departments"][0] if r["expected_departments"] else None)
    dept_acceptable = r["actual_department"] in r["expected_departments"]

    urgency_ok = None
    if r["expected_urgency_min"] is not None and isinstance(r["actual_urgency"], int):
        urgency_ok = r["expected_urgency_min"] <= r["actual_urgency"] <= r["expected_urgency_max"]

    scores.update({
        "severity_exact_match": r["actual_severity"] == r["expected_severity"],
        "department_exact_match": dept_exact,
        "department_acceptable_match": dept_acceptable,
        "urgency_in_expected_range": urgency_ok,
        "expected_emergency": expected_emergency,
        "predicted_emergency": predicted_emergency,
    })
    return {**r, **scores}


def aggregate(records: list) -> dict:
    valid_clinical = [r for r in records if r["expected_severity"] != "Invalid" and r["error"] is None]
    invalid_cases = [r for r in records if r["expected_severity"] == "Invalid" and r["error"] is None]
    errored = [r for r in records if r["error"] is not None]

    def rate(items, key):
        vals = [i[key] for i in items if i.get(key) is not None]
        return round(100 * sum(1 for v in vals if v) / len(vals), 1) if vals else None

    tp = sum(1 for r in valid_clinical if r.get("expected_emergency") and r.get("predicted_emergency"))
    fn = sum(1 for r in valid_clinical if r.get("expected_emergency") and not r.get("predicted_emergency"))
    fp = sum(1 for r in valid_clinical if not r.get("expected_emergency") and r.get("predicted_emergency"))
    precision = round(100 * tp / (tp + fp), 1) if (tp + fp) else None
    recall = round(100 * tp / (tp + fn), 1) if (tp + fn) else None
    f1 = round(2 * precision * recall / (precision + recall), 1) if precision and recall and (precision + recall) else None

    latencies = [r["latency_sec"] for r in records if r.get("latency_sec") is not None]

    return {
        "total_cases": len(records),
        "errored_cases": len(errored),
        "clinical_cases": len(valid_clinical),
        "invalid_input_cases": len(invalid_cases),
        "invalid_handling_accuracy_pct": rate(invalid_cases, "invalid_handling_correct"),
        "severity_exact_match_pct": rate(valid_clinical, "severity_exact_match"),
        "department_exact_match_pct": rate(valid_clinical, "department_exact_match"),
        "department_acceptable_match_pct": rate(valid_clinical, "department_acceptable_match"),
        "urgency_in_expected_range_pct": rate(valid_clinical, "urgency_in_expected_range"),
        "internal_consistency_pct": rate(valid_clinical, "internal_consistency"),
        "emergency_detection_precision_pct": precision,
        "emergency_detection_recall_pct": recall,
        "emergency_detection_f1_pct": f1,
        "avg_latency_sec": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "max_latency_sec": round(max(latencies), 2) if latencies else None,
    }


def write_report(records: list, metrics: dict, out_dir: str, timestamp: str):
    os.makedirs(out_dir, exist_ok=True)

    # NOTE: encoding="utf-8" is required on both writes below. Without it,
    # Python on Windows opens text files using the system's default codepage
    # (commonly cp1252), which cannot represent characters the LLM sometimes
    # emits (e.g. U+2011 NON-BREAKING HYPHEN, smart quotes, em dashes) inside
    # triage_reasoning / summary / actions text. That mismatch is what caused
    # the UnicodeEncodeError crash during report writing.
    json_path = os.path.join(out_dir, f"run_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": timestamp, "metrics": metrics, "cases": records}, f, indent=2)

    failures = [
        r for r in records
        if r["error"] is not None
        or r.get("severity_exact_match") is False
        or r.get("department_acceptable_match") is False
        or r.get("internal_consistency") is False
        or r.get("invalid_handling_correct") is False
    ]

    md_path = os.path.join(out_dir, f"report_{timestamp}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# MediAgent AI - Triage Evaluation Report\n\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write("## Summary\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        for k, v in metrics.items():
            f.write(f"| {k} | {v} |\n")

        f.write("\n## Known limitation this eval is designed to surface\n\n")
        f.write(
            "`apply_triage_guardrails()` in `agents/pipeline.py` downgrades ANY "
            "LLM-flagged Critical case to Moderate unless the raw symptom text "
            "contains one of ~20 hardcoded phrases (e.g. `difficulty breathing`, "
            "`seizure`, `loss of consciousness`). It also has a Rule-2/Rule-4 "
            "ordering issue: chest-pain cases that don't match a Rule-1 phrase "
            "get routed to Cardiology with urgency clamped to 6-8 even if the "
            "LLM correctly said Critical, leaving severity and department/urgency "
            "inconsistent. The `internal_consistency` metric above and the "
            "`guardrail_probe` category failures below are testing specifically "
            "for this.\n"
        )

        f.write(f"\n## Failures ({len(failures)})\n\n")
        if not failures:
            f.write("No failures.\n")
        for r in failures:
            f.write(f"### {r['id']} ({r['category']})\n\n")
            if r["error"]:
                f.write(f"- **Transport error:** {r['error']}\n\n")
                continue
            f.write(f"- Expected severity: `{r['expected_severity']}` | Actual: `{r['actual_severity']}`\n")
            f.write(f"- Expected department(s): `{r['expected_departments']}` | Actual: `{r.get('actual_department')}`\n")
            f.write(f"- Expected urgency range: `[{r['expected_urgency_min']}, {r['expected_urgency_max']}]` | Actual: `{r.get('actual_urgency')}`\n")
            f.write(f"- Internal consistency: `{r.get('internal_consistency')}`\n")
            if r.get("triage_reasoning"):
                f.write(f"- Model's stated reasoning: {r['triage_reasoning']}\n")
            if r.get("guardrail_note"):
                f.write(f"- Guardrail action: {r['guardrail_note']}\n")
            f.write(f"- Case notes: {r['notes']}\n\n")

    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate MediAgent AI's triage pipeline.")
    parser.add_argument("--cases", default=os.path.join(os.path.dirname(__file__), "triage_cases.json"))
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N cases (smoke test).")
    parser.add_argument("--category", default=None, help="Only run cases with this category.")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between cases (rate-limit friendly).")
    parser.add_argument("--dry-run", action="store_true", help="Validate the case file and exit, no LLM calls.")
    args = parser.parse_args()

    with open(args.cases, encoding="utf-8") as f:
        data = json.load(f)
    cases = data["cases"]

    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.limit:
        cases = cases[: args.limit]

    if args.dry_run:
        required = {"id", "category", "body_part", "description", "duration", "onset",
                    "pain_level", "age", "gender", "expected_severity"}
        bad = [c["id"] for c in cases if not required.issubset(c.keys())]
        print(f"Validated {len(cases)} cases.")
        if bad:
            print(f"Missing required fields in: {bad}")
            sys.exit(1)
        print("All cases OK. (No LLM calls made.)")
        return

    if not os.getenv("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY is not set in the environment. The pipeline will likely fail.")

    print(f"Running {len(cases)} case(s) through run_triage_pipeline "
          f"(3 LLM calls each, ~{args.sleep}s sleep between cases)...\n")

    records = []
    try:
        for i, case in enumerate(cases, 1):
            print(f"[{i}/{len(cases)}] {case['id']} ({case['category']})...", end=" ", flush=True)
            record = run_case(case, args.sleep)
            record = score_case(record)
            records.append(record)
            if record["error"]:
                print(f"ERROR: {record['error']}")
            else:
                print(f"severity={record.get('actual_severity')} dept={record.get('actual_department')} "
                      f"urgency={record.get('actual_urgency')} ({record['latency_sec']}s)")
    finally:
        # Always write whatever we have, even on a crash or Ctrl-C, so progress is never lost.
        if records:
            metrics = aggregate(records)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            json_path, md_path = write_report(records, metrics, args.output, timestamp)
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            for k, v in metrics.items():
                print(f"{k:40s}: {v}")
            print(f"\nFull results: {json_path}")
            print(f"Report:       {md_path}")


if __name__ == "__main__":
    main()
