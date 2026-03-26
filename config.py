import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

SECRET_KEY     = os.environ.get('SECRET_KEY', 'autexai-hipaa-secret-2025')
REPORTS_FOLDER = os.path.join(BASE_DIR, 'static', 'reports')
MODEL_PATH     = os.path.join(BASE_DIR, 'models', 'autism_model.pkl')
DATASET_PATH   = os.path.join(BASE_DIR, 'Mental_disorder_symptoms.csv')

# --- Database ---
# Set DATABASE_URL env var to postgres://... for Supabase/Render
# Falls back to local SQLite for dev
DATABASE_URL = os.environ.get('DATABASE_URL', '')
DATABASE     = os.path.join(BASE_DIR, 'instance', 'autexai.db')

# Supabase (optional — only needed if you use Supabase REST API directly)
SUPABASE_URL    = os.environ.get('SUPABASE_URL', '')
SUPABASE_ANON_KEY = os.environ.get('SUPABASE_ANON_KEY', '')

# --- Email (Gmail SMTP) ---
# Set these in Render Environment Variables
# MAIL_USER = your Gmail address (e.g. autexai@gmail.com)
# MAIL_PASS = Gmail App Password (16 chars) from Google Account > Security > App Passwords
MAIL_USER = os.environ.get('MAIL_USER', '')
MAIL_PASS = os.environ.get('MAIL_PASS', '')
