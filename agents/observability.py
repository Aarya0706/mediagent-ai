"""
agents/observability.py
------------------------
Lightweight observability for the MediAgent AI multi-agent pipeline.

Multi-agent systems are hard to debug blind: when a triage run comes
back wrong (or slow, or fails outright), you need to know *which*
agent did it, how long each step took, and whether it's an isolated
blip or a pattern. This module gives every pipeline run a request ID
and records, per agent call:

  - which agent ran, on which model
  - start/end time and latency
  - success / error, and the error type if it failed
  - token usage, when the provider exposes it

Deliberately NOT logged: symptom text, patient context, or any other
patient content. The table below only ever stores metadata about the
*call*, never the clinical payload - see the schema.

Usage:
    from agents.observability import new_run_id, traced_agent_call

    run_id = new_run_id()
    output = traced_agent_call(
        run_id, "intake_agent", GROQ_MODEL, intake_chain,
        {"symptoms": symptoms, "patient_context": patient_context},
    )

    # elsewhere (e.g. a Doctor Portal / admin view):
    from agents.observability import get_observability_stats
    stats = get_observability_stats(hours=24)
"""

import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:  # pragma: no cover - fallback for older langchain
    from langchain.callbacks.base import BaseCallbackHandler

# ── Storage location ────────────────────────────────────────────────
# Same data/ directory app.py already uses for hospital.db, so this
# ships with zero extra configuration. Kept in its own table
# (agent_runs) rather than the old unused agent_logs scaffold in
# database/schema.py, since that table's shape (case_id + free-text
# "action") doesn't fit per-agent-call metrics.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "hospital.db")

_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS agent_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    model TEXT,
    started_at TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    status TEXT NOT NULL,          -- 'success' | 'error'
    error_type TEXT,               -- exception class name, if any
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    case_id INTEGER                -- linked once the case is saved, if ever
);
"""

_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_agent_runs_started_at
    ON agent_runs (started_at);
"""


def _ensure_table():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(_TABLE_SQL)
        conn.execute(_INDEX_SQL)
        conn.commit()


def new_run_id() -> str:
    """One ID per triage-pipeline execution, shared by all its agent calls."""
    return uuid.uuid4().hex[:12]


class _TokenUsageCallback(BaseCallbackHandler):
    """Captures token usage from the LLM response, when the provider sends it.

    Groq (via langchain_groq) reports usage in the same
    `llm_output["token_usage"]` shape the OpenAI-compatible callbacks
    expect. If a future model/provider doesn't send it, these just stay
    None - never fail the pipeline over missing telemetry.
    """

    def __init__(self):
        self.prompt_tokens = None
        self.completion_tokens = None
        self.total_tokens = None

    def on_llm_end(self, response, **kwargs):
        try:
            usage = (response.llm_output or {}).get("token_usage", {})
            self.prompt_tokens = usage.get("prompt_tokens")
            self.completion_tokens = usage.get("completion_tokens")
            self.total_tokens = usage.get("total_tokens")
        except Exception:
            # Telemetry must never break the actual agent call.
            pass


def _log_row(
    run_id,
    agent_name,
    model,
    started_at,
    latency_ms,
    status,
    error_type=None,
    prompt_tokens=None,
    completion_tokens=None,
    total_tokens=None,
    case_id=None,
):
    try:
        _ensure_table()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO agent_runs (
                    run_id, agent_name, model, started_at, latency_ms,
                    status, error_type,
                    prompt_tokens, completion_tokens, total_tokens, case_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, agent_name, model, started_at, latency_ms,
                    status, error_type,
                    prompt_tokens, completion_tokens, total_tokens, case_id,
                ),
            )
            conn.commit()
    except Exception:
        # Observability is best-effort - a logging failure must never
        # take down a triage run.
        pass


@contextmanager
def _timed_call(run_id, agent_name, model):
    """Times a block, logs it on the way out (success or exception),
    and re-raises. Yields the callback so callers can pass it into
    chain.invoke(config={"callbacks": [...]}) to capture token usage."""
    callback = _TokenUsageCallback()
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    try:
        yield callback
    except Exception as exc:
        latency_ms = int((time.monotonic() - start) * 1000)
        _log_row(
            run_id, agent_name, model, started_at, latency_ms,
            status="error", error_type=type(exc).__name__,
        )
        raise
    else:
        latency_ms = int((time.monotonic() - start) * 1000)
        _log_row(
            run_id, agent_name, model, started_at, latency_ms,
            status="success",
            prompt_tokens=callback.prompt_tokens,
            completion_tokens=callback.completion_tokens,
            total_tokens=callback.total_tokens,
        )


def traced_agent_call(run_id, agent_name, model, chain, invoke_kwargs):
    """Runs `chain.invoke(invoke_kwargs)`, logging latency/status/tokens
    against `run_id` and `agent_name`. Returns whatever the chain returns;
    exceptions are logged then re-raised unchanged."""
    with _timed_call(run_id, agent_name, model) as callback:
        return chain.invoke(
            invoke_kwargs,
            config={"callbacks": [callback]},
        )


def log_pipeline_total(run_id, model, started_at, status, error_type=None):
    """Optional roll-up row for the whole pipeline run (agent_name =
    'pipeline_total'), so stats can report end-to-end latency alongside
    per-agent latency."""
    latency_ms = int(
        (datetime.now(timezone.utc) - started_at).total_seconds() * 1000
    )
    _log_row(
        run_id, "pipeline_total", model,
        started_at.isoformat(), latency_ms,
        status=status, error_type=error_type,
    )


def get_observability_stats(hours: int = 24, failure_limit: int = 10) -> dict:
    """Summary for a dev/admin view: total runs, success rate, average
    latency, and the most recent failures (agent + error type + when -
    never symptom/patient content, since that's never stored here)."""
    _ensure_table()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        totals = conn.execute(
            """
            SELECT
                COUNT(*) AS total_calls,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS successes,
                AVG(latency_ms) AS avg_latency_ms
            FROM agent_runs
            WHERE started_at >= ? AND agent_name != 'pipeline_total'
            """,
            (cutoff,),
        ).fetchone()

        runs = conn.execute(
            """
            SELECT COUNT(DISTINCT run_id) AS run_count
            FROM agent_runs
            WHERE started_at >= ?
            """,
            (cutoff,),
        ).fetchone()

        by_agent = conn.execute(
            """
            SELECT
                agent_name,
                COUNT(*) AS calls,
                AVG(latency_ms) AS avg_latency_ms,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors
            FROM agent_runs
            WHERE started_at >= ?
            GROUP BY agent_name
            ORDER BY agent_name
            """,
            (cutoff,),
        ).fetchall()

        failures = conn.execute(
            """
            SELECT run_id, agent_name, model, started_at, error_type
            FROM agent_runs
            WHERE started_at >= ? AND status = 'error'
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (cutoff, failure_limit),
        ).fetchall()

    total_calls = totals["total_calls"] or 0
    successes = totals["successes"] or 0
    success_rate = (successes / total_calls * 100) if total_calls else None

    return {
        "window_hours": hours,
        "total_runs": runs["run_count"] or 0,
        "total_agent_calls": total_calls,
        "success_rate_pct": round(success_rate, 1) if success_rate is not None else None,
        "avg_latency_ms": (
            round(totals["avg_latency_ms"], 0) if totals["avg_latency_ms"] is not None else None
        ),
        "by_agent": [dict(row) for row in by_agent],
        "recent_failures": [dict(row) for row in failures],
    }
