# AutexAI Care — Deployment Guide

## Overview
AutexAI is an ASD (Autism Spectrum Disorder) early screening tool for parents and doctors.  
- **15 behavioral questions** (4 new ones added; model logic unchanged)
- **AI-powered** ensemble ML model (RandomForest, GradientBoosting, ExtraTrees, AdaBoost)
- **Multi-language**: English, Hindi, Marathi
- **HIPAA-compliant** data handling
- **Professional minimalistic UI**

---

## Deploying to Render

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/autexai.git
git push -u origin main
```

### 2. Create Render Web Service
1. Go to [render.com](https://render.com) → **New Web Service**
2. Connect your GitHub repo
3. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT`
   - **Environment:** Python 3

### 3. Set Environment Variables on Render
| Variable | Value |
|---|---|
| `SECRET_KEY` | (generate a random string) |
| `DATABASE_URL` | Your Supabase or Render PostgreSQL URL |
| `SUPABASE_URL` | Your Supabase project URL (optional) |
| `SUPABASE_ANON_KEY` | Your Supabase anon key (optional) |

---

## Supabase Setup

### Option A: Use Supabase PostgreSQL directly (recommended)
1. Go to [supabase.com](https://supabase.com) → New Project
2. **Settings → Database → Connection string** → copy the `URI`
3. Paste it as `DATABASE_URL` in your Render environment variables
4. The app auto-creates tables on first run

### Option B: Render PostgreSQL
1. Create a new **PostgreSQL** database on Render
2. Copy the **External Database URL**
3. Set it as `DATABASE_URL` in your web service

---

## Local Development
```bash
pip install -r requirements.txt
python app.py
```
Open http://localhost:5000

---

## Questions: 15 Total
| # | Key | Maps to Model Feature |
|---|---|---|
| 1–11 | Direct model features | Direct |
| 12 | anger_q2 | anger |
| 13 | repetitive_q2 | repetitive.behaviour |
| 14 | social_q2 | introvert |
| 15 | sensory_q | avoids.people.or.activities |

The ML model (11 features, VotingClassifier ensemble) is **unchanged**.  
Extra questions use OR logic — if Q12 = Yes, `anger = 1`.
