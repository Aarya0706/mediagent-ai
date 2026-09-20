import hashlib
import secrets
from datetime import datetime
from zoneinfo import ZoneInfo

from database.connection import get_connection, DB_PATH  # noqa: F401 - DB_PATH kept for callers that import it from here

IST = ZoneInfo("Asia/Kolkata")


# ============================================================
# SCHEMA
# ============================================================
#
# Streamlit-appropriate reinterpretation of the PRD's JWT/refresh-token
# auth: this app has one long-lived server-side session per browser tab
# (Streamlit's own session_state) rather than a stateless API consumed by
# a separate frontend, so there's no token to issue, refresh, or expire -
# the login just gates st.session_state["auth_user"] for the session's
# lifetime. Passwords are hashed with PBKDF2-HMAC-SHA256 (stdlib hashlib,
# no bcrypt dependency needed) with a random per-user salt.
#
# The users table itself, and its role/patient_name columns, are now
# defined once in database/schema.py and created/migrated by
# database/migrations.py (see ensure_users_table() below) rather than
# each living here.

def ensure_users_table():
    """Kept as a function (rather than inlining the call below) so
    existing callers/imports of ensure_users_table() elsewhere keep
    working unchanged."""
    from database.migrations import run_migrations
    run_migrations()


ensure_users_table()


# ============================================================
# PASSWORD HASHING
# ============================================================

_PBKDF2_ITERATIONS = 260_000


def _hash_password(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        _PBKDF2_ITERATIONS,
    ).hex()


# ============================================================
# REGISTER / VERIFY
# ============================================================

def username_exists(username):
    username = (username or "").strip()
    if not username:
        return False
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM users WHERE LOWER(username) = LOWER(?)",
            (username,),
        )
        return cursor.fetchone() is not None


def create_user(username, password, display_name=None, role="staff", patient_name=None):
    """Registers a new login. Raises ValueError on bad input or a
    username that's already taken (checked case-insensitively).

    role: "staff" (can look up any patient) or "patient" (locked to
    exactly one patient identity, set here and never changed elsewhere).
    For role="patient", patient_name is required and becomes the fixed
    identity this login is allowed to see - the app must not let a
    patient-role session pick a different patient_name from a dropdown
    or free-text field anywhere else.
    """
    username = (username or "").strip()
    password = password or ""

    if not username:
        raise ValueError("Username is required.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    if username_exists(username):
        raise ValueError("That username is already taken.")
    if role not in ("staff", "patient"):
        raise ValueError("Invalid role.")
    if role == "patient" and not (patient_name or "").strip():
        raise ValueError("Patient accounts must have a linked patient name.")

    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO users (username, display_name, salt, password_hash, created_at, role, patient_name)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                username,
                (display_name or username).strip(),
                salt,
                password_hash,
                datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                role,
                (patient_name or "").strip() or None,
            ),
        )
        conn.commit()


def verify_user(username, password):
    """Returns a dict {"display_name", "role", "patient_name"} on
    success, or None if the username doesn't exist or the password is
    wrong. Deliberately returns the same None for both cases so a login
    form can't be used to enumerate which usernames exist."""
    username = (username or "").strip()
    password = password or ""
    if not username or not password:
        return None

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT display_name, salt, password_hash, role, patient_name FROM users WHERE LOWER(username) = LOWER(?)",
            (username,),
        )
        row = cursor.fetchone()

    if row is None:
        return None

    candidate_hash = _hash_password(password, row["salt"])
    if not secrets.compare_digest(candidate_hash, row["password_hash"]):
        return None

    return {
        "display_name": row["display_name"] or username,
        "role": row["role"] or "staff",
        "patient_name": row["patient_name"],
    }


def get_user_count():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        return cursor.fetchone()[0]


# ============================================================
# DEMO ACCOUNT
# ============================================================
#
# One-click "Try Demo" access for anyone evaluating the app (recruiters,
# interviewers) who shouldn't have to register a real account just to
# look around. Idempotent - safe to call on every "Try Demo" click,
# creates the account once and reuses it after that.

DEMO_USERNAME = "demo"
DEMO_PASSWORD = "demo12345"
DEMO_DISPLAY_NAME = "Demo User"

DEMO_PATIENT_USERNAME = "demo_patient"
DEMO_PATIENT_PASSWORD = "demopatient123"
DEMO_PATIENT_DISPLAY_NAME = "Demo Patient"
DEMO_PATIENT_NAME = "Demo Patient"


def ensure_demo_user():
    """Creates the shared staff demo account if it doesn't exist yet, and
    returns its login info either way (same shape as verify_user)."""
    if not username_exists(DEMO_USERNAME):
        try:
            create_user(DEMO_USERNAME, DEMO_PASSWORD, display_name=DEMO_DISPLAY_NAME, role="staff")
        except ValueError:
            pass  # created by a concurrent request between the check and here
    return {"display_name": DEMO_DISPLAY_NAME, "role": "staff", "patient_name": None}


def ensure_demo_patient_user():
    """Creates the shared patient-role demo account if it doesn't exist
    yet, and returns its login info either way. Lets anyone evaluating
    the app see the restricted patient view without registering."""
    if not username_exists(DEMO_PATIENT_USERNAME):
        try:
            create_user(
                DEMO_PATIENT_USERNAME,
                DEMO_PATIENT_PASSWORD,
                display_name=DEMO_PATIENT_DISPLAY_NAME,
                role="patient",
                patient_name=DEMO_PATIENT_NAME,
            )
        except ValueError:
            pass
    return {"display_name": DEMO_PATIENT_DISPLAY_NAME, "role": "patient", "patient_name": DEMO_PATIENT_NAME}