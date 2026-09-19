"""
tests/test_rag_retrieval.py
-----------------------------
Unit tests for tools.chat_rag_tools.retrieve_relevant_chunks() - pure
TF-IDF/sklearn logic over in-memory chunk lists, no DB/LLM/network calls,
so this runs in CI without GROQ_API_KEY (same as test_safety_gate.py and
test_authorization.py).

This is the fast, always-on companion to evaluation/evaluate_rag.py,
which runs the full labeled case set (evaluation/rag_cases.json) and
produces a report; these tests just guard the core behaviors so a
regression fails the build immediately.

Run:
    pytest tests/test_rag_retrieval.py -v
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.chat_rag_tools import retrieve_relevant_chunks


LAB_CHUNK = {
    "text": "Extracted lab values - Hemoglobin: 10.2 g/dL (flag: Low, reference range: 13.0-17.0)",
    "source": "Lab report from 2026-03-01",
}
UNRELATED_CASE_CHUNK = {
    "text": "Symptoms: mild ankle sprain. Severity: Mild. Department: Orthopedics.",
    "source": "Symptom checker case from 2026-01-10",
}


def test_empty_chunks_returns_not_confident():
    retrieved, confident = retrieve_relevant_chunks("what were my results?", [])
    assert retrieved == []
    assert confident is False


def test_direct_keyword_match_is_confident():
    retrieved, confident = retrieve_relevant_chunks(
        "What was my hemoglobin level?", [LAB_CHUNK, UNRELATED_CASE_CHUNK]
    )
    assert confident is True
    assert retrieved[0]["source"] == "Lab report from 2026-03-01"


def test_diabetes_synonym_expansion_matches_blood_sugar_chunk():
    sugar_chunk = {
        "text": "Extracted lab values - Fasting Blood Sugar: 138 mg/dL (flag: High)",
        "source": "Lab report from 2026-04-12",
    }
    retrieved, confident = retrieve_relevant_chunks(
        "Do I have diabetes?", [sugar_chunk, UNRELATED_CASE_CHUNK]
    )
    assert confident is True
    assert any(c["source"] == "Lab report from 2026-04-12" for c in retrieved)


def test_bruising_synonym_matches_platelet_chunk():
    platelet_chunk = {
        "text": "Extracted lab values - Platelets: 95000 /uL (flag: Low)",
        "source": "Lab report from 2026-07-01",
    }
    retrieved, confident = retrieve_relevant_chunks(
        "I've been bruising easily, is that linked to my bloodwork?",
        [platelet_chunk, UNRELATED_CASE_CHUNK],
    )
    assert confident is True
    assert any(c["source"] == "Lab report from 2026-07-01" for c in retrieved)


def test_liver_synonym_matches_alt_ast_chunk_even_with_distractors():
    liver_chunk = {
        "text": "Extracted lab values - ALT: 78 U/L (flag: High); AST: 65 U/L (flag: High)",
        "source": "Lab report from 2026-09-01",
    }
    distractors = [
        {"text": f"Symptoms: minor issue {i}. Severity: Mild. Department: General Medicine.",
         "source": f"Symptom checker case from 2026-0{i}-05"}
        for i in range(1, 5)
    ]
    retrieved, confident = retrieve_relevant_chunks(
        "What did my liver enzymes show?", [liver_chunk] + distractors
    )
    assert confident is True
    assert retrieved[0]["source"] == "Lab report from 2026-09-01"


def test_unrelated_general_question_is_not_confident():
    retrieved, confident = retrieve_relevant_chunks(
        "What foods are generally good for heart health?",
        [UNRELATED_CASE_CHUNK],
    )
    assert confident is False


def test_case_insensitive_and_punctuation_tolerant():
    retrieved, confident = retrieve_relevant_chunks(
        "WHAT WAS MY HEMOGLOBIN???", [LAB_CHUNK, UNRELATED_CASE_CHUNK]
    )
    assert confident is True
    assert retrieved[0]["source"] == "Lab report from 2026-03-01"


def test_top_k_is_respected():
    chunks = [
        {"text": f"Extracted lab values - Hemoglobin: {10 + i} g/dL (flag: Low)",
         "source": f"Lab report {i}"}
        for i in range(10)
    ]
    retrieved, confident = retrieve_relevant_chunks(
        "What was my hemoglobin?", chunks, top_k=4
    )
    assert len(retrieved) <= 4
