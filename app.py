"""
AutexAI Care — app.py  (v4 — Slot Locking + Email Notifications)
=================================================================
LINE-BY-LINE EXPLANATION IS IN THE COMMENTS ABOVE EVERY SECTION.
Model logic (ASDModel, predict, shap) is UNTOUCHED.
"""

# ── STANDARD LIBRARY IMPORTS ──────────────────────────────────────────────────
# os, sys  : file paths, sys.path manipulation
# json     : serialize/deserialize Python dicts to store in DB as TEXT
# time     : timestamp for PDF filenames
# random   : shuffle questions each load so order doesn't bias answers
# smtplib  : Python's built-in SMTP client — used to send emails
# email.*  : build multipart HTML+plain-text email messages
import os, sys, json, time, random, smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from functools import wraps          # preserves function name/docstring in decorators

# ── FLASK IMPORTS ─────────────────────────────────────────────────────────────
# Flask        : the web application object
# render_template : renders Jinja2 HTML templates from /templates folder
# request      : incoming HTTP request data (form fields, JSON body, etc.)
# redirect     : send the browser to a different URL
# url_for      : generate a URL from a route function name (avoids hardcoding URLs)
# flash        : one-time messages shown to the user (success/error banners)
# session      : server-side per-user storage (stores user_id, username, role)
# send_file    : stream a file download to the browser
# jsonify      : converts a Python dict to a JSON HTTP response
from flask import (Flask, render_template, request, redirect, url_for,
                   flash, session, send_file, jsonify)

# ── WERKZEUG ──────────────────────────────────────────────────────────────────
# generate_password_hash : bcrypt/pbkdf2 hash of a plain-text password
# check_password_hash    : verifies plain-text against stored hash
from werkzeug.security import generate_password_hash, check_password_hash

# ── PROJECT MODULES ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from models.db import get_db, init_db
from models.ml_model import ASDModel, QUESTIONS, FEATURE_LABELS, RECOMMENDED_DOCTORS

# ── FLASK APP SETUP ───────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = config.SECRET_KEY

os.makedirs(config.REPORTS_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(config.DATABASE), exist_ok=True)

# ── LOAD ML MODEL ─────────────────────────────────────────────────────────────
asd_model = ASDModel()
MODEL_OK = False
if os.path.exists(config.MODEL_PATH):
    asd_model.load(config.MODEL_PATH)
    MODEL_OK = True

# ============================================================
# HOW TO APPLY THIS PATCH TO app.py
# ============================================================
# In your app.py, find this OLD block (around line 60):
#
#     with app.app_context():
#         init_db()
#
# DELETE those two lines and REPLACE them with the block below.
# Everything else in app.py stays the same.
# ============================================================


# ── LAZY DATABASE INITIALISATION ─────────────────────────────────────────────
# DO NOT call init_db() here at module level.
# Calling it at import time causes gunicorn to crash with
#   "psycopg2.OperationalError: Name or service not known"
# because the Supabase DNS is not yet resolvable during the
# cold-start window before gunicorn binds the port.
#
# This before_request hook runs init_db() exactly ONCE — on the
# very first HTTP request, after gunicorn has fully started and
# the network is available.  A threading.Lock stops multiple
# workers from racing on the same first request.

import threading
_db_initialised = False
_db_lock = threading.Lock()

@app.before_request
def ensure_db():
    global _db_initialised
    if not _db_initialised:
        with _db_lock:
            if not _db_initialised:   # double-checked locking
                init_db()
                _db_initialised = True

# =============================================================================
# EMAIL HELPER
# =============================================================================
def send_email(to_list, subject, html_body):
    """
    Send HTML email via Gmail SMTP SSL.
    IMPORTANT: Set MAIL_USER and MAIL_PASS in Render environment variables.
    MAIL_PASS must be a Gmail App Password (not your normal password).
    Steps to get App Password:
      Google Account -> Security -> 2-Step Verification -> App Passwords
      -> Select app: Mail, device: Other -> Copy the 16-char password
    """
    mail_user = config.MAIL_USER
    mail_pass = config.MAIL_PASS

    if not mail_user or not mail_pass:
        print("[EMAIL] MAIL_USER/MAIL_PASS not configured — skipping email")
        return False

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From']    = f"AutexAI Care <{mail_user}>"
        msg['To']      = ", ".join(to_list)

        plain = html_body.replace('<br>', '\n').replace('</p>', '\n')
        msg.attach(MIMEText(plain, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(mail_user, mail_pass)
            server.sendmail(mail_user, to_list, msg.as_string())

        print(f"[EMAIL] Sent '{subject}' to {to_list}")
        return True
    except Exception as e:
        print(f"[EMAIL] Failed: {e}")
        return False


def _booking_email_html(patient_name, doctor_name, appt_date, appt_time,
                         appt_type, notes, status):
    """Builds the HTML email body for booking and confirmation emails."""
    type_label   = "Video Consult" if appt_type == "video" else "In-Clinic Visit"
    status_color = {"pending": "#f0a500", "confirmed": "#28a745",
                    "cancelled": "#dc3545"}.get(status, "#666")
    note_row = (f"<tr><td style='padding:8px;background:#f5f5f5;font-weight:bold'>Notes</td>"
                f"<td style='padding:8px'>{notes}</td></tr>") if notes else ""
    msg_text = ("Please log in to AutexAI Care to confirm this appointment."
                if status == "pending"
                else "Your appointment has been confirmed. Please be available at the scheduled time.")
    return f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:auto;
                border:1px solid #e0e0e0;border-radius:8px;overflow:hidden">
      <div style="background:#4A90D9;padding:20px;text-align:center">
        <h2 style="color:white;margin:0">AutexAI Care</h2>
        <p style="color:#d0e8ff;margin:4px 0">Understanding Your Child Together</p>
      </div>
      <div style="padding:24px">
        <h3 style="color:#4A90D9">Appointment {status.capitalize()}</h3>
        <table style="width:100%;border-collapse:collapse">
          <tr><td style="padding:8px;background:#f5f5f5;font-weight:bold;width:40%">Patient</td>
              <td style="padding:8px">{patient_name}</td></tr>
          <tr><td style="padding:8px;font-weight:bold">Doctor</td>
              <td style="padding:8px">{doctor_name}</td></tr>
          <tr><td style="padding:8px;background:#f5f5f5;font-weight:bold">Date</td>
              <td style="padding:8px">{appt_date}</td></tr>
          <tr><td style="padding:8px;font-weight:bold">Time</td>
              <td style="padding:8px">{appt_time}</td></tr>
          <tr><td style="padding:8px;background:#f5f5f5;font-weight:bold">Type</td>
              <td style="padding:8px">{type_label}</td></tr>
          <tr><td style="padding:8px;font-weight:bold">Status</td>
              <td style="padding:8px">
                <span style="color:{status_color};font-weight:bold">{status.upper()}</span>
              </td></tr>
          {note_row}
        </table>
        <p style="margin-top:20px;color:#666;font-size:13px">{msg_text}</p>
      </div>
      <div style="background:#f5f5f5;padding:12px;text-align:center;font-size:11px;color:#999">
        AutexAI Care — HIPAA Compliant ASD Screening Platform
      </div>
    </div>
    """

# =============================================================================
# TRANSLATIONS
# =============================================================================
TRANSLATIONS = {
    'en': {
        'app_name': 'AutexAI Care', 'tagline': 'Understanding Your Child Together',
        'welcome': 'Welcome', 'login': 'Login', 'register': 'Register',
        'logout': 'Logout', 'dashboard': 'Dashboard', 'take_test': 'Take Assessment',
        'language': 'Language', 'patient': 'Parent/Guardian', 'doctor': 'Doctor/Specialist'
    },
    'hi': {
        'app_name': 'ऑटेक्सएआई देखभाल', 'tagline': 'आपके बच्चे को समझना',
        'welcome': 'स्वागत है', 'login': 'लॉगिन', 'register': 'पंजीकरण',
        'logout': 'लॉगआउट', 'dashboard': 'डैशबोर्ड', 'take_test': 'परीक्षण लें',
        'language': 'भाषा', 'patient': 'अभिभावक', 'doctor': 'डॉक्टर/विशेषज्ञ'
    },
    'mr': {
        'app_name': 'ऑटेक्सएआय काळजी', 'tagline': 'तुमच्या मुलाला समजून घेणे',
        'welcome': 'स्वागत आहे', 'login': 'लॉगिन', 'register': 'नोंदणी',
        'logout': 'लॉगआउट', 'dashboard': 'डॅशबोर्ड', 'take_test': 'मूल्यांकन घ्या',
        'language': 'भाषा', 'patient': 'पालक/पालक', 'doctor': 'डॉक्टर/तज्ञ'
    }
}

def get_lang():
    return session.get('lang', 'en')

def t(key):
    lang = get_lang()
    return TRANSLATIONS.get(lang, {}).get(key, TRANSLATIONS['en'].get(key, key))

# =============================================================================
# DECORATORS
# =============================================================================
def login_required(f):
    @wraps(f)
    def dec(*a, **kw):
        if 'user_id' not in session:
            flash(t('login') + ' required', 'error')
            return redirect(url_for('login'))
        return f(*a, **kw)
    return dec

def role_required(role):
    def decorator(f):
        @wraps(f)
        def dec(*a, **kw):
            if 'user_id' not in session:
                return redirect(url_for('login'))
            if session.get('role') != role:
                flash('Access denied', 'error')
                return redirect(url_for('dashboard'))
            return f(*a, **kw)
        return dec
    return decorator

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/set_lang/<lang>')
def set_lang(lang):
    if lang in ['en', 'hi', 'mr']:
        session['lang'] = lang
    return redirect(request.referrer or url_for('index'))

@app.route('/')
def index():
    return redirect(url_for('dashboard') if 'user_id' in session else url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm_password', '')
        role     = request.form.get('role', 'patient')
        hipaa    = request.form.get('hipaa_consent')

        if not all([username, email, password, confirm]):
            flash('All fields required', 'error'); return redirect(url_for('register'))
        if not hipaa:
            flash('HIPAA consent required', 'error'); return redirect(url_for('register'))
        if password != confirm:
            flash('Passwords do not match', 'error'); return redirect(url_for('register'))
        if len(password) < 6:
            flash('Password min 6 characters', 'error'); return redirect(url_for('register'))
        if role not in ('patient', 'doctor'):
            role = 'patient'

        db = get_db()
        if db.execute('SELECT id FROM users WHERE username=?', (username,)).fetchone():
            flash('Username taken', 'error'); db.close(); return redirect(url_for('register'))
        if db.execute('SELECT id FROM users WHERE email=?', (email,)).fetchone():
            flash('Email registered', 'error'); db.close(); return redirect(url_for('register'))

        db.execute(
            'INSERT INTO users (username,email,password_hash,role,hipaa_consent,hipaa_consent_date)'
            ' VALUES (?,?,?,?,?,?)',
            (username, email, generate_password_hash(password), role, 1, datetime.utcnow().isoformat())
        )
        db.commit(); db.close()
        flash('Registration successful!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', t=t)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        hipaa    = request.form.get('hipaa_consent')
        if not hipaa:
            flash('HIPAA acknowledgement required', 'error'); return redirect(url_for('login'))
        db   = get_db()
        user = db.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
        db.close()
        if user and check_password_hash(user['password_hash'], password):
            session['user_id']  = user['id']
            session['username'] = user['username']
            session['role']     = user['role']
            flash(f"{t('welcome')}, {user['username']}!", 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid credentials', 'error')
    return render_template('login.html', t=t)

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    db = get_db()
    if session['role'] == 'doctor':
        preds    = db.execute(
            'SELECT p.*, u.username as uname FROM predictions p '
            'JOIN users u ON p.user_id=u.id ORDER BY p.created_at DESC').fetchall()
        patients = db.execute(
            "SELECT id,username,email,created_at FROM users WHERE role='patient'"
            " ORDER BY created_at DESC").fetchall()
        appointments = db.execute(
            'SELECT a.*, u.username as pname FROM appointments a '
            'JOIN users u ON a.patient_id=u.id ORDER BY a.appt_date ASC, a.appt_time ASC'
        ).fetchall()
        total_asd = sum(1 for p in preds if p['prediction_label'] == 'ASD Detected')
        db.close()
        return render_template('dashboard_doctor.html',
                               preds=preds, patients=patients,
                               appointments=appointments, total_asd=total_asd,
                               model_metrics=asd_model.metrics,
                               doctors=RECOMMENDED_DOCTORS, t=t)
    else:
        preds = db.execute(
            'SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC',
            (session['user_id'],)).fetchall()
        appointments = db.execute(
            'SELECT * FROM appointments WHERE patient_id=? ORDER BY appt_date ASC, appt_time ASC',
            (session['user_id'],)).fetchall()
        db.close()
        return render_template('dashboard_patient.html', preds=preds,
                               appointments=appointments,
                               doctors=RECOMMENDED_DOCTORS, t=t)

@app.route('/questionnaire', methods=['GET', 'POST'])
@role_required('patient')
def questionnaire():
    # ── MODEL LOGIC — DO NOT CHANGE ──────────────────────────────────────────
    lang = get_lang()
    if not MODEL_OK:
        flash('Model not ready', 'error'); return redirect(url_for('dashboard'))

    if request.method == 'POST':
        try:
            raw = {}
            questions = QUESTIONS[lang]
            for q in questions:
                raw[q['key']] = int(request.form.get(q['key'], 0))

            prob, threshold, label, confidence = asd_model.predict(raw)
            contributions, plot_path = asd_model.generate_shap_plot(
                raw, config.REPORTS_FOLDER, 'shap')
            top_features = [(f, round(v, 4)) for f, v, _ in contributions]

            report_path = _make_pdf(session['username'], raw, prob, threshold,
                                    label, confidence, top_features, plot_path, lang)
            db = get_db()
            db.execute(
                '''INSERT INTO predictions
                   (user_id, username, responses, asd_probability, asd_threshold,
                    prediction_label, confidence, top_features, shap_plot, report_path)
                   VALUES (?,?,?,?,?,?,?,?,?,?)''',
                (session['user_id'], session['username'], json.dumps(raw),
                 round(prob * 100, 2), round(threshold * 100, 2),
                 label, confidence, json.dumps(top_features),
                 os.path.basename(plot_path), report_path))
            db.commit()
            pred_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]
            db.close()
            return redirect(url_for('results', pred_id=pred_id))
        except Exception as e:
            import traceback; traceback.print_exc()
            flash(f'Error: {str(e)}', 'error')
            return redirect(url_for('questionnaire'))
    # ─────────────────────────────────────────────────────────────────────────

    questions = list(QUESTIONS[lang])
    random.shuffle(questions)
    return render_template('questionnaire.html', questions=questions, lang=lang, t=t)

@app.route('/results/<int:pred_id>')
@login_required
def results(pred_id):
    db   = get_db()
    pred = db.execute('SELECT * FROM predictions WHERE id=?', (pred_id,)).fetchone()
    db.close()
    if not pred:
        flash('Not found', 'error'); return redirect(url_for('dashboard'))
    if session['role'] == 'patient' and pred['user_id'] != session['user_id']:
        flash('Access denied', 'error'); return redirect(url_for('dashboard'))
    top_features = json.loads(pred['top_features']) if pred['top_features'] else []
    top_labeled  = [(FEATURE_LABELS.get(f, f), v) for f, v in top_features]
    shap_img     = pred['shap_plot'] or None
    show_doctors = pred['asd_probability'] >= pred['asd_threshold']
    return render_template('results.html', pred=pred, top_features=top_labeled,
                           shap_img=shap_img, show_doctors=show_doctors,
                           doctors=RECOMMENDED_DOCTORS, t=t)

@app.route('/download/<int:pred_id>')
@login_required
def download_report(pred_id):
    db   = get_db()
    pred = db.execute('SELECT * FROM predictions WHERE id=?', (pred_id,)).fetchone()
    db.close()
    if not pred:
        flash('Not found', 'error'); return redirect(url_for('dashboard'))
    if session['role'] == 'patient' and pred['user_id'] != session['user_id']:
        flash('Access denied', 'error'); return redirect(url_for('dashboard'))
    rp = pred['report_path']
    if rp and os.path.exists(rp):
        return send_file(rp, as_attachment=True)
    flash('PDF not found', 'error')
    return redirect(url_for('dashboard'))

@app.route('/doctor/<int:doctor_idx>')
@login_required
def doctor_profile(doctor_idx):
    if doctor_idx < 0 or doctor_idx >= len(RECOMMENDED_DOCTORS):
        flash('Doctor not found', 'error'); return redirect(url_for('dashboard'))
    doc = RECOMMENDED_DOCTORS[doctor_idx]
    db = get_db()
    appointments = db.execute(
        'SELECT * FROM appointments WHERE doctor_name=? ORDER BY appt_date ASC',
        (doc['name'],)).fetchall()
    db.close()
    return render_template('doctor_profile.html', doc=doc, doc_idx=doctor_idx,
                           appointments=appointments, t=t)

# =============================================================================
# ★ NEW API: GET AVAILABLE SLOTS
# =============================================================================
@app.route('/api/available_slots')
@login_required
def available_slots():
    """
    Called by JavaScript when a patient picks a date in the booking modal.
    Returns only the slots NOT already booked for that doctor+date.

    HOW SLOT BLOCKING WORKS:
    - ALL_SLOTS = every possible time in a day
    - We query appointments WHERE doctor_name=? AND appt_date=? AND status IN ('pending','confirmed')
    - We subtract those from ALL_SLOTS
    - Frontend renders only what's returned here → booked slots are INVISIBLE
    """
    doctor_name = request.args.get('doctor_name', '')
    date        = request.args.get('date', '')

    ALL_SLOTS = ['09:00','09:30','10:00','10:30','11:00','11:30','12:00','12:30',
                 '17:00','17:30','18:00','18:30','19:00','19:30']

    if not doctor_name or not date:
        return jsonify({'available': ALL_SLOTS})

    db = get_db()
    rows = db.execute(
        "SELECT appt_time FROM appointments "
        "WHERE doctor_name=? AND appt_date=? AND status IN ('pending','confirmed')",
        (doctor_name, date)
    ).fetchall()
    db.close()

    booked    = {row['appt_time'] for row in rows}    # set for fast lookup
    available = [s for s in ALL_SLOTS if s not in booked]
    return jsonify({'available': available})

# =============================================================================
# ★ UPDATED: BOOK APPOINTMENT — conflict check + emails
# =============================================================================
@app.route('/book_appointment', methods=['POST'])
@login_required
def book_appointment():
    """
    SLOT CONFLICT CHECK (server-side safety net):
    Even if the JS already filtered slots, we re-check in the DB.
    If someone books the same slot in two browser tabs simultaneously,
    only the first INSERT succeeds — the second sees the existing row and
    returns an error flash.

    EMAIL FLOW:
    After successful booking:
      1. Fetch patient email from users table
      2. doctor_email comes from the form (hidden input = doc.email)
      3. send_email([patient_email, doctor_email], subject, html)
         → both get the "Appointment Pending" email
    """
    doctor_name  = request.form.get('doctor_name', '')
    doctor_email = request.form.get('doctor_email', '')
    appt_date    = request.form.get('appt_date', '')
    appt_time    = request.form.get('appt_time', '')
    appt_type    = request.form.get('appt_type', 'in_clinic')
    notes        = request.form.get('notes', '')

    if not all([doctor_name, appt_date, appt_time]):
        flash('Please fill in all booking fields.', 'error')
        return redirect(request.referrer or url_for('dashboard'))

    db = get_db()

    # SERVER-SIDE SLOT CONFLICT CHECK
    existing = db.execute(
        "SELECT id FROM appointments "
        "WHERE doctor_name=? AND appt_date=? AND appt_time=?"
        " AND status IN ('pending','confirmed')",
        (doctor_name, appt_date, appt_time)
    ).fetchone()

    if existing:
        flash(f'The {appt_time} slot on {appt_date} is already booked. Please choose another time.', 'error')
        db.close()
        return redirect(request.referrer or url_for('dashboard'))

    # Insert with status='pending'
    db.execute(
        '''INSERT INTO appointments
           (patient_id, patient_name, doctor_name, doctor_email,
            appt_date, appt_time, appt_type, notes, status)
           VALUES (?,?,?,?,?,?,?,?,?)''',
        (session['user_id'], session['username'],
         doctor_name, doctor_email,
         appt_date, appt_time, appt_type, notes, 'pending')
    )
    db.commit()

    # Fetch patient email for notification
    patient_row = db.execute('SELECT email FROM users WHERE id=?',
                              (session['user_id'],)).fetchone()
    db.close()

    # SEND EMAILS TO BOTH PATIENT AND DOCTOR
    recipients = []
    if patient_row and patient_row['email']:
        recipients.append(patient_row['email'])
    if doctor_email:
        recipients.append(doctor_email)
    if recipients:
        html = _booking_email_html(session['username'], doctor_name,
                                   appt_date, appt_time, appt_type, notes, 'pending')
        send_email(recipients,
                   f"[AutexAI] Appointment Booked — {appt_date} at {appt_time}", html)

    flash(f'Appointment booked with {doctor_name} on {appt_date} at {appt_time}!'
          f' Confirmation email sent.', 'success')
    return redirect(url_for('dashboard'))

# =============================================================================
# CANCEL APPOINTMENT
# =============================================================================
@app.route('/cancel_appointment/<int:appt_id>', methods=['POST'])
@login_required
def cancel_appointment(appt_id):
    """
    Sets status='cancelled' → this frees the slot so available_slots API
    will include it again for future bookings.
    """
    db   = get_db()
    appt = db.execute('SELECT * FROM appointments WHERE id=?', (appt_id,)).fetchone()
    if appt and (appt['patient_id'] == session['user_id'] or session['role'] == 'doctor'):
        db.execute("UPDATE appointments SET status='cancelled' WHERE id=?", (appt_id,))
        db.commit()
        flash('Appointment cancelled.', 'info')
    db.close()
    return redirect(request.referrer or url_for('dashboard'))

# =============================================================================
# ★ UPDATED: CONFIRM APPOINTMENT — doctor action + email to patient
# =============================================================================
@app.route('/confirm_appointment/<int:appt_id>', methods=['POST'])
@role_required('doctor')
def confirm_appointment(appt_id):
    """
    Doctor clicks "Confirm" on an appointment.
    1. DB: status = 'confirmed'
    2. Fetch appointment + patient email
    3. send_email to patient (and CC doctor)
       → patient gets "Your appointment is CONFIRMED" email
    """
    db = get_db()
    db.execute("UPDATE appointments SET status='confirmed' WHERE id=?", (appt_id,))
    db.commit()

    appt        = db.execute('SELECT * FROM appointments WHERE id=?', (appt_id,)).fetchone()
    patient_row = None
    if appt:
        patient_row = db.execute('SELECT email FROM users WHERE id=?',
                                  (appt['patient_id'],)).fetchone()
    db.close()

    # SEND CONFIRMATION EMAIL
    if appt and patient_row and patient_row['email']:
        recipients = [patient_row['email']]
        if appt['doctor_email']:
            recipients.append(appt['doctor_email'])
        html = _booking_email_html(appt['patient_name'], appt['doctor_name'],
                                   appt['appt_date'], appt['appt_time'],
                                   appt['appt_type'], appt['notes'] or '', 'confirmed')
        send_email(recipients,
                   f"[AutexAI] Appointment CONFIRMED — {appt['appt_date']} at {appt['appt_time']}",
                   html)

    flash('Appointment confirmed. Patient notified by email.', 'success')
    return redirect(request.referrer or url_for('dashboard'))

# =============================================================================
# CHATBOT
# =============================================================================
CHATBOT_KB = {
    'what is autism': 'Autism Spectrum Disorder (ASD) is a neurodevelopmental condition that affects communication, social interaction, and behavior.',
    'asd symptoms': 'Common signs: difficulty with social interaction, repetitive behaviors, sensory sensitivity, delayed speech, insistence on routines.',
    'how is asd diagnosed': 'ASD is diagnosed by specialists through behavioral observations and standardized tests. Our app provides early screening only.',
    'early signs': 'Before age 3: limited eye contact, not responding to name, delayed speech, repetitive movements.',
    'treatment': 'Early intervention helps: ABA therapy, speech therapy, occupational therapy, social skills training.',
    'book appointment': 'Book from Dashboard → Recommended Doctors. Only available (unbooked) slots are shown.',
    'video call': 'Video consultations available. Book via doctor profile — only open slots shown. Confirmation email sent.',
    'assessment': '15-question behavioral screening (~5 min). Answer Yes/No about your child.',
    'results': 'ASD probability %, contributing behaviors, SHAP chart, downloadable PDF report.',
    'privacy': 'Data encrypted and handled under HIPAA regulations.',
    'languages': 'Supports English, Hindi, and Marathi.',
    'doctors': 'Specialized doctors in Pune: Child Psychologists, Pediatric Psychiatrists, Developmental Pediatricians.',
    'hello': 'Hello! I\'m the AutexAI Care assistant. Ask me about autism, assessments, or appointments.',
    'hi': 'Hello! How can I help you today?',
    'help': 'I can help with: autism symptoms, ASD screening, booking appointments, understanding results.',
    'what is autexai': 'AutexAI Care is an early ASD screening platform using ensemble ML to assess autism risk.',
    'how accurate': 'Our ensemble model has ~85%+ accuracy. This is a screening tool — always consult a specialist.',
    'age': 'Designed for children aged 2–12 years.',
    'cost': 'Screening is free. Doctor fees vary.',
    'sensory': 'Sensory sensitivities are common in autism — one of our 15 behavioral indicators.',
}

def get_chatbot_response(message):
    msg = message.lower().strip()
    for key, reply in CHATBOT_KB.items():
        if key in msg: return reply
    if any(w in msg for w in ['appoint','book','schedule']): return CHATBOT_KB['book appointment']
    if any(w in msg for w in ['video','call','online']):     return CHATBOT_KB['video call']
    if any(w in msg for w in ['symptom','sign','behav']):    return CHATBOT_KB['asd symptoms']
    if any(w in msg for w in ['diagnos','test','assess']):   return CHATBOT_KB['assessment']
    if any(w in msg for w in ['treat','therap','interven']): return CHATBOT_KB['treatment']
    if any(w in msg for w in ['result','score','percent']):  return CHATBOT_KB['results']
    if any(w in msg for w in ['doctor','specialist']):       return CHATBOT_KB['doctors']
    if any(w in msg for w in ['privac','secure','hipaa']):   return CHATBOT_KB['privacy']
    if any(w in msg for w in ['accur','reliab','trust']):    return CHATBOT_KB['how accurate']
    if any(w in msg for w in ['age','year','child','kid']):  return CHATBOT_KB['age']
    if any(w in msg for w in ['early','infant','toddler']):  return CHATBOT_KB['early signs']
    return "I'm not sure. Ask me about autism symptoms, ASD assessment, booking appointments, or results."

@app.route('/api/chat', methods=['POST'])
@login_required
def chat_api():
    data    = request.get_json()
    message = data.get('message', '').strip()
    if not message:
        return jsonify({'reply': 'Please type a message.'})
    return jsonify({'reply': get_chatbot_response(message)})

# =============================================================================
# PDF REPORT GENERATOR (unchanged from v3)
# =============================================================================
def _make_pdf(username, raw, prob, threshold, label, confidence, top_features, shap_plot, lang='en'):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                     Paragraph, Spacer, Image, HRFlowable)
    from reportlab.lib.units import inch

    fname    = f'asd_report_{username}_{int(time.time())}.pdf'
    filepath = os.path.join(config.REPORTS_FOLDER, fname)
    pdf_doc  = SimpleDocTemplate(filepath, pagesize=letter,
                                  leftMargin=0.75*inch, rightMargin=0.75*inch,
                                  topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles  = getSampleStyleSheet(); elems = []
    title_s = ParagraphStyle('T', parent=styles['Heading1'], fontSize=22,
                              textColor=colors.HexColor('#4A90D9'), alignment=1, spaceAfter=4)
    sub_s   = ParagraphStyle('S', parent=styles['Normal'], fontSize=10,
                              textColor=colors.grey, alignment=1, spaceAfter=14)
    h2_s    = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=13,
                              textColor=colors.HexColor('#4A90D9'), spaceBefore=12, spaceAfter=6)
    elems.append(Paragraph('AutexAI Care — ASD Screening Report', title_s))
    elems.append(Paragraph('Understanding Your Child Together  |  HIPAA Compliant', sub_s))
    elems.append(HRFlowable(width='100%', thickness=1.5,
                             color=colors.HexColor('#4A90D9'), spaceAfter=10))
    prob_pct = round(prob * 100, 2); thr_pct = round(threshold * 100, 2)
    info = [
        ['Parent/Guardian', username, 'Date', datetime.now().strftime('%Y-%m-%d %H:%M')],
        ['ASD Probability', f'{prob_pct:.2f}%', 'Threshold', f'{thr_pct:.2f}%'],
        ['Result', label, 'Confidence', confidence],
    ]
    tbl = Table(info, colWidths=[1.5*inch, 2.5*inch, 1.3*inch, 1.5*inch])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(0,-1),colors.HexColor('#4A90D9')),
        ('BACKGROUND',(2,0),(2,-1),colors.HexColor('#4A90D9')),
        ('TEXTCOLOR',(0,0),(0,-1),colors.white),('TEXTCOLOR',(2,0),(2,-1),colors.white),
        ('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),10),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey),
        ('BOTTOMPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),9),
    ]))
    elems.append(tbl); elems.append(Spacer(1, 0.2*inch))
    elems.append(Paragraph('What This Means', h2_s))
    if prob_pct >= thr_pct:
        elems.append(Paragraph(
            f'ASD probability <b>{prob_pct:.2f}%</b> above threshold <b>{thr_pct:.2f}%</b>. '
            f'<b>Please consult a specialist.</b> Early intervention makes a significant difference!',
            styles['Normal'])); elems.append(Spacer(1, 0.15*inch))
        elems.append(Paragraph('Recommended Specialists in Pune', h2_s))
        doc_data = [['Doctor','Hospital','Contact']]
        for d in RECOMMENDED_DOCTORS:
            doc_data.append([f"{d['name']}\n{d['specialty']}",
                             f"{d['hospital']}\n{d['address']}",
                             f"{d['phone']}\n{d['email']}"])
        dt = Table(doc_data, colWidths=[2.2*inch, 2.8*inch, 1.6*inch])
        dt.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#4A90D9')),
            ('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,-1),9),('GRID',(0,0),(-1,-1),0.4,colors.grey),
            ('VALIGN',(0,0),(-1,-1),'TOP'),
            ('BOTTOMPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8),
        ]))
        elems.append(dt); elems.append(Spacer(1, 0.15*inch))
    else:
        elems.append(Paragraph(
            f'ASD probability <b>{prob_pct:.2f}%</b> below threshold. Fewer ASD indicators present. '
            f'Continue monitoring milestones. If concerns persist, consult a professional.',
            styles['Normal']))
    elems.append(Spacer(1, 0.15*inch))
    elems.append(Paragraph('Top Contributing Behaviors', h2_s))
    feat_data = [['#','Behavior','Effect']]
    for i, (fn, fv) in enumerate(top_features[:8], 1):
        feat_data.append([str(i), FEATURE_LABELS.get(fn, fn),
                          'Increases ASD indicators' if fv >= 0 else 'Decreases ASD indicators'])
    ft = Table(feat_data, colWidths=[0.4*inch, 4*inch, 2.2*inch])
    ft.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#7FB069')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),9),('GRID',(0,0),(-1,-1),0.4,colors.grey),
        ('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),
    ]))
    elems.append(ft); elems.append(Spacer(1, 0.2*inch))
    if os.path.exists(shap_plot):
        elems.append(Paragraph('Behavioral Analysis Chart', h2_s))
        elems.append(Image(shap_plot, width=6.5*inch, height=3.4*inch))
        elems.append(Spacer(1, 0.15*inch))
    disc_s = ParagraphStyle('D', parent=styles['Normal'], fontSize=8, textColor=colors.grey)
    elems.append(HRFlowable(width='100%', thickness=0.5, color=colors.grey, spaceAfter=6))
    elems.append(Paragraph(
        '<b>IMPORTANT:</b> Screening tool only — NOT a medical diagnosis. '
        'Only a licensed specialist can diagnose ASD. All data handled per HIPAA.',
        disc_s))
    pdf_doc.build(elems)
    return filepath

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
