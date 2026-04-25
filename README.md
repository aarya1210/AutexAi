# AutexAI — Explainable AI System for Autism Spectrum Detection

> An AI-powered early screening tool for parents and clinicians — predicting autism spectrum risk from behavioral symptom inputs, with full explainability via SHAP.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=flat-square&logo=scikit-learn)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-purple?style=flat-square)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Supabase-green?style=flat-square&logo=postgresql)
![Deployed](https://img.shields.io/badge/Deployed-Render-46E3B7?style=flat-square&logo=render)

---

## What it does

AutexAI is a clinical screening web application that:

- Walks parents or caregivers through **15 behavioral questions** across 3 languages (English, Hindi, Marathi)
- Runs responses through an **ensemble ML model** to predict ASD (Autism Spectrum Disorder) risk probability
- Generates a **SHAP explanation chart** showing which specific behavioral indicators contributed to the prediction
- Provides separate **dashboards for patients and doctors**, with role-based access control
- Stores assessment history in **PostgreSQL** and supports PDF report export

---

## ML Architecture

### Model
A `VotingClassifier` ensemble of 4 models trained on behavioral symptom data:

| Model | Role |
|---|---|
| Random Forest (200 trees) | Primary classifier, class-balanced |
| Gradient Boosting (150 trees) | Captures sequential patterns |
| ExtraTrees (200 trees) | Reduces overfitting via randomization |
| AdaBoost + DecisionTree | Handles misclassified edge cases |

Prediction threshold is **optimized for F1 score** (sweep from 0.30–0.75) on the test split rather than defaulting to 0.5 — improving recall for clinical use.

### Feature Selection Pipeline
Features are **not manually hardcoded** — a 3-method consensus pipeline runs at training time:

```
ANOVA F-test  ──┐
Mutual Info   ──┼──► Consensus Vote (≥2/3) ──► 11 selected features
RFE + RF      ──┘
```

26 candidate behavioral features → 11 selected → saved into `.pkl` so training and inference always use the same feature set.

### Explainability
`shap.TreeExplainer` runs on each prediction to produce a horizontal bar chart of the top 8 behavioral indicators, showing the **direction and magnitude** of each feature's contribution to the ASD probability score.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask 3.0 |
| ML | Scikit-learn, SHAP, NumPy, Pandas |
| Frontend | HTML, CSS, JavaScript (Jinja2 templates) |
| Database | PostgreSQL (Supabase) / SQLite (local dev) |
| Auth | Werkzeug password hashing, Flask sessions |
| Deployment | Render (gunicorn), Procfile |
| Reports | ReportLab (PDF generation) |

---

## Features

- **Role-based access** — separate patient and doctor dashboards
- **Multilingual** — English, Hindi, Marathi questionnaire
- **SHAP charts** — per-prediction behavioral indicator breakdown
- **Auto-train fallback** — retrains from CSV if `.pkl` is missing or corrupted (zero-downtime cold starts)
- **Email notifications** — Gmail SMTP integration
- **PDF reports** — exportable assessment results via ReportLab

---

## Project Structure

```
AutexAI/
├── app.py                    # Flask app, routes, auth, sessions
├── train_model.py            # Training script with feature selection report
├── config.py                 # Env-based configuration
├── Mental_disorder_symptoms.csv
├── models/
│   ├── ml_model.py           # ASDModel class, feature selection, SHAP
│   ├── db.py                 # PostgreSQL / SQLite abstraction
│   └── autism_model.pkl      # Saved trained model
├── templates/
│   ├── questionnaire.html
│   ├── results.html          # SHAP chart + prediction output
│   ├── dashboard_doctor.html
│   └── dashboard_patient.html
└── static/
    ├── css/style.css
    └── js/main.js
```

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/aarya1210/autexai.git
cd autexai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python train_model.py

# 4. Run the app
python app.py
```

Open `http://localhost:5000`

---

## Deployment (Render + Supabase)

### Environment Variables

| Variable | Description |
|---|---|
| `SECRET_KEY` | Flask session secret |
| `DATABASE_URL` | Supabase / Render PostgreSQL URI |
| `MAIL_USER` | Gmail address for notifications |
| `MAIL_PASS` | Gmail App Password (16 chars) |

### Render Settings
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT`

Tables are auto-created on first request — no manual migration needed.

---

## How the Prediction Works

```
User answers 15 questions
        │
        ▼
preprocess_input()  ←  OR-logic for Q12–Q15 (mapped to base features)
        │
        ▼
StandardScaler.transform()
        │
        ▼
VotingClassifier.predict_proba()  →  ASD probability score
        │
        ├──  score ≥ threshold  →  "ASD Detected"   (High / Moderate confidence)
        └──  score < threshold  →  "ASD Not Detected"
        │
        ▼
shap.TreeExplainer  →  SHAP bar chart (top 8 contributing features)
```

---

## Disclaimer

AutexAI is a **screening aid only** — not a clinical diagnostic tool. Results should always be reviewed by a qualified medical professional. The app recommends local specialists and encourages professional follow-up for all positive screenings.

---

## Author

**Aarya Ingawale**  
BE Computer Engineering — PES Modern College of Engineering, Pune (2026)  
[LinkedIn](https://www.linkedin.com/in/aarya-ingawale-8080bb220) · [GitHub](https://github.com/aarya1210) · aaryaingawale12@gmail.com
