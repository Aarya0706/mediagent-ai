"""
tests/test_observability.py
-----------------------------
Unit tests for agents.observability. Everything here is pure
SQLite + Python - no LLM/network calls - so it's safe to run in CI
without GROQ_API_KEY, same as test_safety_gate.py and
test_authorization.py.

Each test points the module at a fresh temp DB file (via monkeypatch)
so runs don't collide with each other or with a real data/hospital.db.

Run:
    pytest tests/test_observability.py -v
"""

import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import agents.observability as observability


def _use_temp_db(monkeypatch, tmp_path):
    db_path = str(tmp_path / "test_observability.db")
    monkeypatch.setattr(observability, "DB_PATH", db_path)
    return db_path


def test_ensure_table_is_idempotent(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)

    observability._ensure_table()
    observability._ensure_table()  # must not raise on second call


def test_new_run_id_is_unique_and_short():
    ids = {observability.new_run_id() for _ in range(50)}

    assert len(ids) == 50
    assert all(len(run_id) == 12 for run_id in ids)


def test_traced_agent_call_logs_success_and_returns_chain_output(
    monkeypatch, tmp_path
):
    _use_temp_db(monkeypatch, tmp_path)

    class FakeChain:
        def invoke(self, kwargs, config=None):
            return "chain output"

    run_id = observability.new_run_id()
    output = observability.traced_agent_call(
        run_id, "intake_agent", "test-model", FakeChain(), {"symptoms": "x"}
    )

    assert output == "chain output"

    stats = observability.get_observability_stats(hours=24)
    assert stats["total_agent_calls"] == 1
    assert stats["success_rate_pct"] == 100.0
    assert stats["recent_failures"] == []


def test_traced_agent_call_logs_failure_and_reraises(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)

    class FailingChain:
        def invoke(self, kwargs, config=None):
            raise ValueError("boom")

    run_id = observability.new_run_id()

    try:
        observability.traced_agent_call(
            run_id, "triage_agent", "test-model", FailingChain(), {}
        )
        assert False, "expected ValueError to propagate"
    except ValueError:
        pass

    stats = observability.get_observability_stats(hours=24)
    assert stats["total_agent_calls"] == 1
    assert stats["success_rate_pct"] == 0.0
    assert len(stats["recent_failures"]) == 1
    assert stats["recent_failures"][0]["agent_name"] == "triage_agent"
    assert stats["recent_failures"][0]["error_type"] == "ValueError"


def test_stats_respect_time_window(monkeypatch, tmp_path):
    db_path = _use_temp_db(monkeypatch, tmp_path)
    observability._ensure_table()

    import sqlite3

    old_timestamp = "2000-01-01T00:00:00+00:00"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO agent_runs (
                run_id, agent_name, model, started_at, latency_ms, status
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("old-run", "intake_agent", "test-model", old_timestamp, 100, "success"),
        )
        conn.commit()

    stats = observability.get_observability_stats(hours=24)

    assert stats["total_agent_calls"] == 0
    assert stats["total_runs"] == 0


def test_logged_rows_never_contain_patient_content(monkeypatch, tmp_path):
    """The whole point of this table is call metadata only. Guard against
    a future change accidentally adding a symptoms/patient_context column."""
    _use_temp_db(monkeypatch, tmp_path)
    observability._ensure_table()

    import sqlite3

    with sqlite3.connect(observability.DB_PATH) as conn:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(agent_runs)")
        }

    forbidden = {"symptoms", "patient_context", "intake", "summary", "notes"}
    assert not (columns & forbidden)


def test_log_pipeline_total_records_a_pipeline_total_row(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)

    run_id = observability.new_run_id()
    started = datetime.now(timezone.utc)
    observability.log_pipeline_total(run_id, "test-model", started, "success")

    stats = observability.get_observability_stats(hours=24)
    agent_names = {row["agent_name"] for row in stats["by_agent"]}
    assert "pipeline_total" in agent_names
