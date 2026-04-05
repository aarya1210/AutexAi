import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATABASE, DATABASE_URL

log = logging.getLogger(__name__)


def _fix_url(url):
    # psycopg2 needs postgresql:// — Render / Supabase often give postgres://
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


def get_db(retries=3, delay=2.0):
    url = (DATABASE_URL or "").strip()

    if url and (url.startswith("postgres") or url.startswith("postgresql")):
        url = _fix_url(url)
        import psycopg2, psycopg2.extras
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
                    log.warning("DB attempt %d/%d failed: %s  retrying in %ss",
                                attempt, retries, exc, delay)
                    time.sleep(delay)
        raise last_err

    import sqlite3
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def _is_pg(conn):
    try:
        import psycopg2
        return isinstance(conn, psycopg2.extensions.connection)
    except ImportError:
        return False


def init_db():
    try:
        conn = get_db()
    except Exception as exc:
        log.error("init_db: DB unreachable: %s — will retry on first request.", exc)
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
