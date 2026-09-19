"""
tools/authorization.py
-----------------------
Centralized authorization for MediAgent AI.

Before this module, every screen in app.py re-implemented its own
version of "is this a patient session, and are they only allowed to
see their own name" inline - one check per tab, written slightly
differently each time (some checked auth_role, some checked
auth_patient_name, some did both). That meant the actual access rule
lived nowhere in particular, and there was nothing stopping a new
screen from forgetting the check, or a future edit from quietly
dropping it. This module makes the rule a single reusable function
so it can be reasoned about, reused, and unit-tested in one place.

Two checks:

  1. can_access_staff_view()
     True only for a signed-in "staff" session. Use to gate any
     screen/tab that shows data across all patients (the case
     archive, analytics, doctor portal).

  2. can_access_patient_record(requested_patient_name)
     True when the current session is allowed to see records for
     requested_patient_name: staff can see anyone; a patient-role
     session only its own locked identity; anyone else, never.

Streamlit note: app.py renders every tab inside one script run, so a
gate here must NOT call st.stop() (that would kill every other tab
too) - it just returns a bool, and the caller shows a message and
skips rendering the protected section, exactly like the pre-existing
per-tab checks did. What changes is that the rule itself now lives in
one audited place instead of being copy-pasted.

Pair this with a DB-level scoped query (e.g.
tools.save_case.get_cases_for_patient()) wherever the protected data
is patient-specific, so access is enforced twice: once here in the UI
layer, and again as a WHERE clause at the database layer. A screen
that shows the full archive across all patients only needs the
staff-only check below, since there's no per-patient WHERE clause to
add.
"""

import streamlit as st


def current_session():
    """Returns {"role": ..., "patient_name": ..., "user": ...} for the
    logged-in session, pulling straight from st.session_state so there
    is exactly one place that knows the session-state key names."""

    return {
        "role": st.session_state.get("auth_role"),
        "patient_name": st.session_state.get("auth_patient_name"),
        "user": st.session_state.get("auth_user"),
    }


def can_access_staff_view(session=None):
    """True only when the current session is a signed-in staff user."""

    session = session or current_session()
    return session["user"] is not None and session["role"] == "staff"


def can_access_patient_record(requested_patient_name, session=None):
    """True when the current session may view records belonging to
    requested_patient_name.

    - staff: always True (staff can look up any patient).
    - patient: True only when requested_patient_name matches the
      identity this login was locked to at signup (case-insensitive).
    - anonymous / any other role: always False.
    """

    session = session or current_session()
    requested = (requested_patient_name or "").strip()

    if session["user"] is None:
        return False

    if session["role"] == "staff":
        return True

    if session["role"] == "patient":
        own_name = (session["patient_name"] or "").strip()
        return bool(own_name) and bool(requested) and own_name.lower() == requested.lower()

    return False


STAFF_ONLY_MESSAGE = (
    "This view is for hospital staff only. Patients don't have access "
    "to the full case archive."
)

PATIENT_MISMATCH_MESSAGE = "You can only view your own records."
