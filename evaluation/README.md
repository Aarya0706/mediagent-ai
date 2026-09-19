# MediAgent AI — Triage Evaluation

A repeatable, quantitative check on the 3-agent triage pipeline's accuracy and safety
behavior, so claims about AI quality are backed by numbers instead of anecdotes.

## What's here

- `triage_cases.json` — 60 curated clinical scenarios spanning explicit emergencies,
  moderate/routine presentations across all 14 routable departments, deliberate
  "guardrail probe" cases, and invalid/non-medical inputs. Each case records the
  *clinically correct* expected severity, department(s), and urgency range.
- `evaluate_triage.py` — runs every case through the real `run_triage_pipeline()`
  and reports accuracy + a safety-specific "internal consistency" check.
- `results/` — generated on each run (gitignored contents are fine to keep, or add
  `evaluation/results/*.json` to `.gitignore` if you don't want raw runs in git and
  just want to commit the markdown reports).

## Running it

```bash
export GROQ_API_KEY="your-key-here"   # or rely on your .env
python evaluation/evaluate_triage.py
```

Useful flags:

```bash
python evaluation/evaluate_triage.py --dry-run          # validate the case file, no API calls
python evaluation/evaluate_triage.py --limit 10          # quick smoke test (10 cases, 30 calls)
python evaluation/evaluate_triage.py --sleep 2.0         # slower pace if you hit Groq rate limits
python evaluation/evaluate_triage.py --category guardrail_probe   # just the probe cases
```

Every case costs 3 LLM calls (Intake, Triage, Recommendation agents), so the full
60-case run makes ~180 calls. Budget a few minutes and mind your Groq rate limit.

## What it measures

| Metric | What it tells you |
|---|---|
| `invalid_handling_accuracy_pct` | Does the Intake Agent correctly reject gibberish/non-medical input? |
| `severity_exact_match_pct` | 3-class (Critical/Moderate/Mild) accuracy vs. the clinically expected label |
| `department_exact_match_pct` / `department_acceptable_match_pct` | Routing accuracy — exact vs. "reasonable" match |
| `urgency_in_expected_range_pct` | Does the numeric urgency score land where it clinically should? |
| `emergency_detection_precision/recall/f1_pct` | How well the pipeline catches true emergencies without over-flagging routine cases |
| `internal_consistency_pct` | **Independent of clinical correctness** — does the pipeline's own output obey its own stated rules (Critical ⇒ Emergency dept + urgency 9–10, etc.)? |

## Known limitation this harness is built to catch

`apply_triage_guardrails()` in `agents/pipeline.py` only recognizes ~20 hardcoded
phrases (e.g. `"difficulty breathing"`, `"seizure"`, `"loss of consciousness"`) as
emergency red flags. Anything the LLM correctly identifies as Critical using
*different* wording (e.g. "gasping for air", "sudden vision loss", a rigid acute
abdomen) gets forcibly downgraded to Moderate by Rule 4. Separately, chest-pain
cases that mention the word "chest" but not an exact red-flag phrase can end up
with `severity="Critical"` but `department="Cardiology"` and urgency clamped to
6–8 — an internally inconsistent result.

The `guardrail_probe` category cases (`GR-101` … `GR-105`, `GI-506`, `NEURO-603`,
`OPHTH-1002`) and the `internal_consistency_pct` metric are specifically designed
to surface this class of bug on every run, so a fix can be verified by re-running
the eval rather than by manual spot-checking.

## RAG evaluation

This harness only covers the triage pipeline. A separate `rag_cases.json` +
evaluation script (measuring retrieval relevance and answer grounding) should be
added once the RAG/citation feature from the roadmap (Section 7) is built —
there's currently no retrieval system in the codebase to evaluate yet.
