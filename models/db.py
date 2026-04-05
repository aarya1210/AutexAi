"""
models/db.py  —  AutexAI
=========================
FIXES vs original:
  1. postgres:// URL → postgresql:// normalisation  (psycopg2 requires the longer form)
  2. connect_timeout=10 added so gunicorn never hangs at cold-start
  3. Retry loop (3 attempts, 2 s apart) handles brief DNS-propagation
     window that produced "Name or service not known" on Render
  4. init_db() wrapped in try/except — a DB hiccup at startup no longer
     crashes the process; schema creation retries on the first request
  5. RealDictCursor attached so psycopg2 rows support row['col'] access
     exactly like sqlite3.Row
"""

import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATABASE, DATABASE_URL

log = logging.getLogger(__name__)


def _normalise_pg_url(url: str) -> str:
    """Render / Supabase sometimes give postgres:// — psycopg2 needs postgresql://"""
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


def get_db(retries: int = 3, delay: float = 2.0):
    """
    Return an open database connection.

    PostgreSQL (Render + Supabase):
      - Normalises URL scheme
      - Sets connect_timeout=10
      - Retries up to `retries` times with `delay` s between attempts

    SQLite (local dev):
      - No retries needed — local file.
    """
    url = DATABASE_URL.strip() if DATABASE_URL else ""

    if url and (url.startswith("postgres://") or url.startswith("postgresql://")):
        url = _normalise_pg_url(url)
        import psycopg2
        import psycopg2.extras

        last_err = None
        for attempt in range(1, retries + 1):
            try:
                conn = psycopg2.connect(
                    url,
                    sslmode="require",
                    connect_timeout=10,
                    cursor_factory=psycopg2.extras.RealDictCursor,
                )
                conn.autocommit = False
                return conn
            except psycopg2.OperationalError as exc:
                last_err = exc
                if attempt < retries:
                    log.warning(
                        "DB connection attempt %d/%d failed (%s). Retrying in %.0f s…",
                        attempt, retries, exc, delay,
                    )
                    time.sleep(delay)
        raise last_err

    import sqlite3
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def _is_pg(conn) -> bool:
    try:
        import psycopg2
        return isinstance(conn, psycopg2.extensions.connection)
    except ImportError:
        return False


def init_db():
    """
    Create all tables if they do not already exist.
    Wrapped in try/except — a transient DB failure at startup does NOT
    crash gunicorn. Schema creation retries on the first real request.
    """
    try:
        conn = get_db()
    except Exception as exc:
        log.error("init_db: could not connect to database: %s", exc)
        log.error("App will start anyway; DB will be initialised on first request.")
        return

    c = conn.cursor()
    try:
        if _is_pg(conn):
            c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id                 SERIAL PRIMARY KEY,
                    username           TEXT UNIQUE NOT NULL,
                    email              TEXT UNIQUE NOT NULL,
                    password_hash      TEXT NOT NULL,
                    role               TEXT NOT NULL DEFAULT 'patient',
                    hipaa_consent      INTEGER NOT NULL DEFAULT 0,
                    hipaa_consent_date TEXT,
                    created_at         TEXT DEFAULT (to_char(now(), 'YYYY-MM-DD HH24:MI:SS'))
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id               SERIAL PRIMARY KEY,
                    user_id          INTEGER NOT NULL,
                    username         TEXT NOT NULL,
                    responses        TEXT NOT NULL,
                    asd_probability  REAL NOT NULL,
                    asd_threshold    REAL NOT NULL,
                    prediction_label TEXT NOT NULL,
                    confidence       TEXT NOT NULL,
                    top_features     TEXT,
                    shap_plot        TEXT,
                    report_path      TEXT,
                    created_at       TEXT DEFAULT (to_char(now(), 'YYYY-MM-DD HH24:MI:SS')),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS appointments (
                    id               SERIAL PRIMARY KEY,
                    patient_id       INTEGER NOT NULL,
                    patient_name     TEXT NOT NULL,
                    doctor_name      TEXT NOT NULL,
                    doctor_email     TEXT,
                    appt_date        TEXT NOT NULL,
                    appt_time        TEXT NOT NULL,
                    appt_type        TEXT DEFAULT 'in_clinic',
                    notes            TEXT,
                    status           TEXT DEFAULT 'pending',
                    created_at       TEXT DEFAULT (to_char(now(), 'YYYY-MM-DD HH24:MI:SS')),
                    FOREIGN KEY (patient_id) REFERENCES users(id)
                )
            """)
        else:
            c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                    username           TEXT UNIQUE NOT NULL,
                    email              TEXT UNIQUE NOT NULL,
                    password_hash      TEXT NOT NULL,
                    role               TEXT NOT NULL DEFAULT 'patient',
                    hipaa_consent      INTEGER NOT NULL DEFAULT 0,
                    hipaa_consent_date TEXT,
                    created_at         TEXT DEFAULT (datetime('now'))
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id          INTEGER NOT NULL,
                    username         TEXT NOT NULL,
                    responses        TEXT NOT NULL,
                    asd_probability  REAL NOT NULL,
                    asd_threshold    REAL NOT NULL,
                    prediction_label TEXT NOT NULL,
                    confidence       TEXT NOT NULL,
                    top_features     TEXT,
                    shap_plot        TEXT,
                    report_path      TEXT,
                    created_at       TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS appointments (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id   INTEGER NOT NULL,
                    patient_name TEXT NOT NULL,
                    doctor_name  TEXT NOT NULL,
                    doctor_email TEXT,
                    appt_date    TEXT NOT NULL,
                    appt_time    TEXT NOT NULL,
                    appt_type    TEXT DEFAULT 'in_clinic',
                    notes        TEXT,
                    status       TEXT DEFAULT 'pending',
                    created_at   TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (patient_id) REFERENCES users(id)
                )
            """)

        conn.commit()
        log.info("init_db: schema ready.")

    except Exception as exc:
        log.error("init_db: schema creation failed: %s", exc)
        conn.rollback()
    finally:
        conn.close()
