"""
evaluation/evaluate_rag.py
----------------------------
Runs the curated case set in evaluation/rag_cases.json through the real
retrieve_relevant_chunks() (tools/chat_rag_tools.py) and reports:

  1. Retrieval recall     -- for cases with an expected relevant source,
                              was at least one of them actually retrieved?
  2. Retrieval precision   -- of what got retrieved, how much was in the
                              expected set (only scored for cases that
                              declare a non-empty expected set)?
  3. Confidence accuracy   -- does the retriever's own `confident` flag
                              (used upstream to decide whether to also
                              pull PubMed) match what each case expects?

This is a *retrieval* evaluation - no LLM calls, no GROQ_API_KEY needed -
same "instant, CI-safe, quantitative" spirit as evaluate_triage.py, just
one layer earlier in the pipeline (retrieval, not generation).

Optional --with-llm additionally runs a handful of cases through the real
generate_chat_answer() (costs Groq API calls) as a lightweight grounding
spot-check: does the answer actually reference a "Based on your..." /
"Per PubMed" style citation when personal context was retrieved? This is
NOT a full automated grounding judge (that would need a second LLM to
grade the first one's output) - it's a cheap signal for manual review,
same caveat evaluate_triage.py gives its own numbers.

Usage:
    python evaluation/evaluate_rag.py
    python evaluation/evaluate_rag.py --category synonym_expansion
    python evaluation/evaluate_rag.py --with-llm         # + grounding spot-check, costs API calls
    python evaluation/evaluate_rag.py --dry-run          # validate the case file only

Outputs (written to evaluation/results/ by default):
    rag_run_<timestamp>.json   -- full per-case results + aggregate metrics
    rag_report_<timestamp>.md  -- human-readable summary + failures
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def run_case(case):
    from tools.chat_rag_tools import retrieve_relevant_chunks

    retrieved, confident = retrieve_relevant_chunks(case["question"], case["chunks"])
    retrieved_sources = [c["source"] for c in retrieved]

    expected = set(case.get("expected_relevant_sources", []))

    if expected:
        hit = bool(expected & set(retrieved_sources))
        # Precision only over chunks pulled from this case's own pool -
        # every retrieved source should trace back to something we gave it.
        overlap = len(expected & set(retrieved_sources))
        precision = overlap / len(retrieved_sources) if retrieved_sources else 0.0
    else:
        # A case that expects nothing relevant (e.g. no_confident_match,
        # empty_records) "passes" recall trivially - there's nothing to find.
        hit = True
        precision = None

    confidence_correct = confident == case["expect_confident"]

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expected_confident": case["expect_confident"],
        "actual_confident": confident,
        "confidence_correct": confidence_correct,
        "expected_relevant_sources": sorted(expected),
        "retrieved_sources": retrieved_sources,
        "recall_hit": hit,
        "precision": precision,
    }


def run_llm_spot_check(case, llm):
    from tools.chat_rag_tools import retrieve_relevant_chunks, generate_chat_answer

    retrieved, confident = retrieve_relevant_chunks(case["question"], case["chunks"])
    result = generate_chat_answer(case["question"], retrieved, mode="patient", llm=llm)

    cites_source = any(
        marker in result["answer"] for marker in ["Based on", "Per PubMed", "records show", "records don't show"]
    )

    return {
        "id": case["id"],
        "question": case["question"],
        "answer": result["answer"],
        "sources_returned": result["sources"],
        "appears_to_cite": cites_source,
    }


def aggregate(records):
    total = len(records)
    recall_hits = sum(1 for r in records if r["recall_hit"])
    confidence_correct = sum(1 for r in records if r["confidence_correct"])

    precisions = [r["precision"] for r in records if r["precision"] is not None]
    avg_precision = sum(precisions) / len(precisions) if precisions else None

    return {
        "total_cases": total,
        "retrieval_recall_pct": round(100 * recall_hits / total, 1) if total else 0.0,
        "confidence_accuracy_pct": round(100 * confidence_correct / total, 1) if total else 0.0,
        "avg_precision_pct": round(100 * avg_precision, 1) if avg_precision is not None else None,
    }


def write_report(records, metrics, output_dir, timestamp, llm_records=None):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"rag_run_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"metrics": metrics, "records": records, "llm_spot_check": llm_records or []},
            f, indent=2,
        )

    md_path = os.path.join(output_dir, f"rag_report_{timestamp}.md")
    lines = [
        f"# RAG Retrieval Evaluation — {timestamp}",
        "",
        "## Aggregate metrics",
        "",
    ]
    for k, v in metrics.items():
        lines.append(f"- **{k}**: {v}")

    failures = [r for r in records if not r["recall_hit"] or not r["confidence_correct"]]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            lines.append(f"### {r['id']} ({r['category']})")
            lines.append(f"- question: {r['question']}")
            lines.append(f"- expected_confident={r['expected_confident']} actual_confident={r['actual_confident']}")
            lines.append(f"- expected_sources={r['expected_relevant_sources']}")
            lines.append(f"- retrieved_sources={r['retrieved_sources']}")
            lines.append("")

    if llm_records:
        lines += ["", "## LLM grounding spot-check (manual review)", ""]
        for r in llm_records:
            lines.append(f"### {r['id']}")
            lines.append(f"- question: {r['question']}")
            lines.append(f"- appears_to_cite: {r['appears_to_cite']}")
            lines.append(f"- answer: {r['answer']}")
            lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate MediAgent AI's RAG retrieval quality.")
    parser.add_argument("--cases", default=os.path.join(ROOT, "evaluation", "rag_cases.json"))
    parser.add_argument("--output", default=os.path.join(ROOT, "evaluation", "results"))
    parser.add_argument("--category", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate the case file and exit, no retrieval calls.")
    parser.add_argument("--with-llm", action="store_true", help="Also run a small grounding spot-check via the real LLM (costs Groq API calls).")
    args = parser.parse_args()

    with open(args.cases, encoding="utf-8") as f:
        data = json.load(f)
    cases = data["cases"]

    if args.category:
        cases = [c for c in cases if c["category"] == args.category]

    if args.dry_run:
        required = {"id", "category", "question", "chunks", "expect_confident", "expected_relevant_sources"}
        bad = [c["id"] for c in cases if not required.issubset(c.keys())]
        print(f"Validated {len(cases)} cases.")
        if bad:
            print(f"Missing required fields in: {bad}")
            sys.exit(1)
        print("All cases OK. (No retrieval calls made.)")
        return

    print(f"Running {len(cases)} case(s) through retrieve_relevant_chunks (no LLM calls)...\n")

    records = [run_case(c) for c in cases]
    metrics = aggregate(records)

    for r in records:
        status = "OK" if (r["recall_hit"] and r["confidence_correct"]) else "MISS"
        print(f"[{status}] {r['id']} ({r['category']}) confident={r['actual_confident']} "
              f"retrieved={len(r['retrieved_sources'])}")

    llm_records = None
    if args.with_llm:
        if not os.getenv("GROQ_API_KEY"):
            print("\nWARNING: --with-llm requested but GROQ_API_KEY is not set. Skipping grounding spot-check.")
        else:
            from langchain_groq import ChatGroq
            llm = ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"), temperature=0.2)
            spot_check_cases = [c for c in cases if c.get("expected_relevant_sources")][:5]
            print(f"\nRunning LLM grounding spot-check on {len(spot_check_cases)} case(s)...")
            llm_records = [run_llm_spot_check(c, llm) for c in spot_check_cases]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path, md_path = write_report(records, metrics, args.output, timestamp, llm_records)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k:30s}: {v}")
    print(f"\nFull results: {json_path}")
    print(f"Report:       {md_path}")


if __name__ == "__main__":
    main()
