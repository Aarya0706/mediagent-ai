"""
tests/test_authorization.py
----------------------------
Unit tests for tools.authorization. These don't touch Streamlit's
session_state at all - every check takes an explicit `session` dict,
so the access rules can be verified without a running app.

Run:
    pytest tests/test_authorization.py -v
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.authorization import can_access_staff_view, can_access_patient_record


def staff_session():
    return {"role": "staff", "patient_name": None, "user": "dr_patel"}


def patient_session(name="Jane Doe"):
    return {"role": "patient", "patient_name": name, "user": "jane_login"}


def signed_out_session():
    return {"role": None, "patient_name": None, "user": None}


# ── can_access_staff_view ──────────────────────────────────────────


def test_staff_can_access_staff_view():
    assert can_access_staff_view(staff_session()) is True


def test_patient_cannot_access_staff_view():
    assert can_access_staff_view(patient_session()) is False


def test_signed_out_cannot_access_staff_view():
    assert can_access_staff_view(signed_out_session()) is False


# ── can_access_patient_record ──────────────────────────────────────


def test_staff_can_access_any_patient_record():
    assert can_access_patient_record("Anyone At All", staff_session()) is True


def test_patient_can_access_own_record():
    session = patient_session("Jane Doe")
    assert can_access_patient_record("Jane Doe", session) is True


def test_patient_access_is_case_insensitive():
    session = patient_session("Jane Doe")
    assert can_access_patient_record("jane doe", session) is True


def test_patient_cannot_access_someone_elses_record():
    session = patient_session("Jane Doe")
    assert can_access_patient_record("John Smith", session) is False


def test_patient_with_no_linked_name_denied():
    session = patient_session(name=None)
    assert can_access_patient_record("Jane Doe", session) is False


def test_empty_requested_name_denied_for_patient():
    session = patient_session("Jane Doe")
    assert can_access_patient_record("", session) is False


def test_signed_out_cannot_access_any_patient_record():
    assert can_access_patient_record("Jane Doe", signed_out_session()) is False


def test_unknown_role_denied():
    session = {"role": "guest", "patient_name": "Jane Doe", "user": "someone"}
    assert can_access_patient_record("Jane Doe", session) is False
    assert can_access_staff_view(session) is False