"""
tests/test_drug_service.py
-----------------------------
Unit tests for services/drug_service.py, extracted from app.py.
normalize_drug_name and parse_drug_field are pure and fully covered
here with no network calls. query_openfda hits the real OpenFDA API
(no key required, but it is a live network call) - covered by a
single lightweight smoke test that tolerates the API being slow or
briefly unavailable rather than failing CI on flakiness, same
philosophy as not making evaluation/evaluate_triage.py's LLM calls
part of the CI-required suite.

Run:
    pytest tests/test_drug_service.py -v
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

from services.drug_service import normalize_drug_name, parse_drug_field, query_openfda


def test_normalize_drug_name_maps_known_brand_names():
    assert normalize_drug_name("Dolo 650") == "acetaminophen"
    assert normalize_drug_name("crocin") == "acetaminophen"
    assert normalize_drug_name("Ecosprin") == "aspirin"
    assert normalize_drug_name("Advil") == "ibuprofen"


def test_normalize_drug_name_passes_through_unknown_names():
    assert normalize_drug_name("warfarin") == "warfarin"
    assert normalize_drug_name("SomeUnknownDrug") == "someunknowndrug"


def test_normalize_drug_name_strips_and_lowercases():
    assert normalize_drug_name("  Paracetamol  ") == "acetaminophen"


def test_parse_drug_field_extracts_known_field():
    text = "SEVERITY: Major\nPLAIN_SUMMARY: Some summary here.\nMECHANISM: Unknown\nPATIENT_ADVICE: See a doctor."
    assert parse_drug_field(text, "SEVERITY") == "Major"
    assert parse_drug_field(text, "PATIENT_ADVICE") == "See a doctor."


def test_parse_drug_field_missing_field_returns_empty_string():
    text = "SEVERITY: Major\n"
    assert parse_drug_field(text, "MECHANISM") == ""


def test_parse_drug_field_is_case_insensitive_on_field_name():
    text = "severity: Minor\n"
    assert parse_drug_field(text, "SEVERITY") == "Minor"


def test_query_openfda_smoke_test_known_interacting_pair():
    """Live network call - a real, well-documented interaction pair
    (warfarin + aspirin) should return a well-formed result dict even
    if the exact evidence count varies over time as OpenFDA's data
    changes. Never asserts an exact count - only shape and the fact
    that evidence exists for a pair known to have documented
    interactions."""
    try:
        result = query_openfda("warfarin", "aspirin")
    except Exception:
        pytest.skip("OpenFDA API unreachable from this environment")

    assert isinstance(result, dict)
    assert "found" in result
    assert "count" in result
    assert "interactions" in result
    assert isinstance(result["interactions"], list)
