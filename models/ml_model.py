"""
models/ml_model.py  — AutexAI v6
=================================
KEY CHANGE FROM v4:
  Features are NO LONGER manually hardcoded.
  A three-method feature selection pipeline runs during training:
    1. ANOVA F-test        (measures variance between ASD vs non-ASD classes)
    2. Mutual Information  (measures non-linear statistical dependency)
    3. RFE with RandomForest (recursive elimination using tree importance)
  A feature is selected if at least 2 out of 3 methods agree (consensus vote).
  The selected feature list is SAVED into .pkl so predict() always uses
  exactly the same features the model was trained on.

EVERYTHING ELSE IDENTICAL TO v4 — ensemble, scaler, threshold, preprocess,
predict(), generate_shap_plot() are ALL unchanged.
"""

import numpy as np
import pandas as pd
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

# ── NEW imports for feature selection ─────────────────────────────────────────
from sklearn.feature_selection import f_classif, mutual_info_classif, RFE

import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CANDIDATE FEATURES  (all numeric columns from the CSV that are clinically relevant)
# =============================================================================
# Feature selection will automatically pick the best subset from this list.
# 'no_close_friend' is engineered as (1 - close.friend) in _load_and_engineer().
CANDIDATE_FEATURES = [
    'feeling.nervous',
    'panic',
    'breathing.rapidly',
    'sweating',
    'trouble.in.concentration',
    'having.trouble.in.sleeping',
    'having.trouble.with.work',
    'hopelessness',
    'anger',
    'over.react',
    'change.in.eating',
    'suicidal.thought',
    'feeling.tired',
    'social.media.addiction',
    'weight.gain',
    'introvert',
    'popping.up.stressful.memory',
    'having.nightmares',
    'avoids.people.or.activities',
    'feeling.negative',
    'trouble.concentrating',
    'blamming.yourself',
    'hallucinations',
    'repetitive.behaviour',
    'increased.energy',
    'no_close_friend',
]

# How many top features each method nominates before consensus voting
N_SELECT = 11

# MODEL_FEATURES is populated automatically during training / loading.
# DO NOT edit this list manually — it is set by select_features().
MODEL_FEATURES = []

# =============================================================================
# FEATURE LABELS  (human-readable display names — shown in results + PDF)
# =============================================================================
FEATURE_LABELS = {
    'introvert':                    'Strong Introversion / Avoids Socializing',
    'avoids.people.or.activities':  'Avoids People or Activities',
    'no_close_friend':              'Difficulty Making Close Friends',
    'repetitive.behaviour':         'Repetitive Behaviours or Rituals',
    'anger':                        'Intense / Uncontrollable Anger',
    'having.nightmares':            'Frequent Nightmares',
    'social.media.addiction':       'Social Media Addiction',
    'over.react':                   'Overreacting to Situations',
    'panic':                        'Panic Attacks',
    'change.in.eating':             'Significant Change in Eating Habits',
    'breathing.rapidly':            'Rapid Breathing / Hyperventilation',
    'feeling.nervous':              'Persistent Nervousness / Anxiety',
    'sweating':                     'Excessive Sweating / Physical Anxiety',
    'trouble.in.concentration':     'Difficulty Concentrating',
    'having.trouble.in.sleeping':   'Sleep Difficulties',
    'having.trouble.with.work':     'Difficulty with Daily Tasks',
    'hopelessness':                 'Feelings of Hopelessness',
    'suicidal.thought':             'Intrusive or Distressing Thoughts',
    'feeling.tired':                'Persistent Fatigue / Low Energy',
    'weight.gain':                  'Unexplained Weight Change',
    'popping.up.stressful.memory':  'Intrusive Stressful Memories',
    'feeling.negative':             'Persistent Negative Thinking',
    'trouble.concentrating':        'Sustained Concentration Difficulty',
    'blamming.yourself':            'Excessive Self-Blame',
    'hallucinations':               'Perceptual Disturbances',
    'increased.energy':             'Unusually Increased Energy',
}

# =============================================================================
# QUESTIONS  (unchanged from v4 — all 15 questions in EN / HI / MR)
# =============================================================================
QUESTIONS = {
    'en': [
        {"key": "introvert",                   "text": "Does your child prefer to be alone rather than play with other children?",            "quote": "Every child is unique and develops at their own pace",                  "image": "child_alone.png"},
        {"key": "avoids.people.or.activities", "text": "Does your child avoid people or activities they used to enjoy?",                     "quote": "Understanding leads to better support",                               "image": "child_activities.png"},
        {"key": "close_friend_q",              "text": "Does your child find it hard to make or keep close friends?",                        "quote": "Friendship skills can be learned and nurtured",                       "image": "children_playing.png"},
        {"key": "repetitive.behaviour",        "text": "Does your child repeat the same actions, movements, or words over and over?",         "quote": "Patterns are how children make sense of the world",                   "image": "child_routine.jpg"},
        {"key": "anger",                       "text": "Does your child get very angry suddenly without a clear reason?",                    "quote": "Emotions are messages — we just need to understand them",              "image": "child_emotions.png"},
        {"key": "having.nightmares",           "text": "Does your child often have bad dreams or wake up scared at night?",                  "quote": "Sleep is important for development and well-being",                   "image": "child_sleeping.png"},
        {"key": "social.media.addiction",      "text": "Does your child spend too much time on phone, tablet, or computer?",                 "quote": "Balance is key in the digital age",                                  "image": "child_tablet.png"},
        {"key": "over.react",                  "text": "Does your child react very strongly to small changes or problems?",                  "quote": "Sensitivity can be a strength when understood",                      "image": "child_change.png"},
        {"key": "panic",                       "text": "Does your child suddenly feel very scared or have panic attacks?",                   "quote": "Your awareness is the first step to helping",                        "image": "child_comfort.png"},
        {"key": "change.in.eating",            "text": "Has your child's eating habits changed a lot recently?",                            "quote": "Nutrition affects mood and behavior",                                 "image": "child_eating.png"},
        {"key": "breathing.rapidly",           "text": "Does your child breathe very fast or feel breathless when worried?",                 "quote": "You're doing great by seeking answers",                              "image": "child_calm.png"},
        {"key": "anger_q2",                    "text": "Does your child have difficulty calming down after becoming upset?",                 "quote": "Emotional regulation is a skill that can be developed",               "image": "child_emotions.png",   "maps_to": "anger"},
        {"key": "repetitive_q2",               "text": "Does your child insist on strict routines and get upset when they change?",          "quote": "Structure and predictability help children feel safe",                "image": "child_routine.jpg",    "maps_to": "repetitive.behaviour"},
        {"key": "social_q2",                   "text": "Does your child struggle to understand other people's feelings or expressions?",     "quote": "Empathy grows with gentle guidance and understanding",                "image": "children_playing.png", "maps_to": "introvert"},
        {"key": "sensory_q",                   "text": "Is your child unusually sensitive to sounds, lights, textures, or touch?",           "quote": "Sensory experiences shape how children engage with the world",        "image": "child_comfort.png",    "maps_to": "avoids.people.or.activities"},
    ],
    'hi': [
        {"key": "introvert",                   "text": "क्या आपका बच्चा दूसरे बच्चों के साथ खेलने की बजाय अकेले रहना पसंद करता है?",       "quote": "हर बच्चा अनोखा है और अपनी गति से विकसित होता है",                    "image": "child_alone.png"},
        {"key": "avoids.people.or.activities", "text": "क्या आपका बच्चा उन लोगों या गतिविधियों से बचता है जो पहले उसे पसंद थीं?",           "quote": "समझ बेहतर सहायता की ओर ले जाती है",                                  "image": "child_activities.png"},
        {"key": "close_friend_q",              "text": "क्या आपके बच्चे को करीबी दोस्त बनाने या रखने में कठिनाई होती है?",                 "quote": "दोस्ती के कौशल सीखे और विकसित किए जा सकते हैं",                     "image": "children_playing.png"},
        {"key": "repetitive.behaviour",        "text": "क्या आपका बच्चा एक ही हरकत, गतिविधि, या शब्द को बार-बार दोहराता है?",              "quote": "पैटर्न वह तरीका है जिससे बच्चे दुनिया को समझते हैं",                 "image": "child_routine.jpg"},
        {"key": "anger",                       "text": "क्या आपका बच्चा अचानक बिना किसी स्पष्ट कारण के बहुत गुस्सा हो जाता है?",           "quote": "भावनाएं संदेश हैं — हमें बस उन्हें समझने की जरूरत है",               "image": "child_emotions.png"},
        {"key": "having.nightmares",           "text": "क्या आपके बच्चे को अक्सर बुरे सपने आते हैं या रात में डर कर उठता है?",             "quote": "नींद विकास और कल्याण के लिए महत्वपूर्ण है",                         "image": "child_sleeping.png"},
        {"key": "social.media.addiction",      "text": "क्या आपका बच्चा फोन, टैबलेट, या कंप्यूटर पर बहुत अधिक समय बिताता है?",            "quote": "डिजिटल युग में संतुलन महत्वपूर्ण है",                               "image": "child_tablet.png"},
        {"key": "over.react",                  "text": "क्या आपका बच्चा छोटे बदलाव या समस्याओं पर अत्यधिक प्रतिक्रिया देता है?",           "quote": "समझने पर संवेदनशीलता एक ताकत हो सकती है",                           "image": "child_change.png"},
        {"key": "panic",                       "text": "क्या आपके बच्चे को अचानक बहुत डर लगता है या घबराहट के दौरे आते हैं?",              "quote": "आपकी जागरूकता मदद की पहली कदम है",                                  "image": "child_comfort.png"},
        {"key": "change.in.eating",            "text": "क्या आपके बच्चे की खाने की आदतें हाल ही में बहुत बदल गई हैं?",                    "quote": "पोषण मनोदशा और व्यवहार को प्रभावित करता है",                         "image": "child_eating.png"},
        {"key": "breathing.rapidly",           "text": "क्या आपका बच्चा चिंतित होने पर बहुत तेजी से सांस लेता है?",                       "quote": "आप उत्तर खोजकर बहुत अच्छा कर रहे हैं",                             "image": "child_calm.png"},
        {"key": "anger_q2",                    "text": "क्या आपके बच्चे को परेशान होने के बाद शांत होने में कठिनाई होती है?",               "quote": "भावनात्मक नियंत्रण एक कौशल है जो विकसित किया जा सकता है",           "image": "child_emotions.png",   "maps_to": "anger"},
        {"key": "repetitive_q2",               "text": "क्या आपका बच्चा सख्त दिनचर्या पर जोर देता है और बदलाव पर परेशान हो जाता है?",     "quote": "संरचना और पूर्वानुमान बच्चों को सुरक्षित महसूस कराता है",           "image": "child_routine.jpg",    "maps_to": "repetitive.behaviour"},
        {"key": "social_q2",                   "text": "क्या आपके बच्चे को दूसरों की भावनाओं या अभिव्यक्तियों को समझने में कठिनाई होती है?","quote": "सहानुभूति कोमल मार्गदर्शन के साथ बढ़ती है",                          "image": "children_playing.png", "maps_to": "introvert"},
        {"key": "sensory_q",                   "text": "क्या आपका बच्चा आवाज़, रोशनी, या स्पर्श के प्रति असामान्य रूप से संवेदनशील है?",  "quote": "संवेदी अनुभव बच्चों की दुनिया को आकार देते हैं",                    "image": "child_comfort.png",    "maps_to": "avoids.people.or.activities"},
    ],
    'mr': [
        {"key": "introvert",                   "text": "तुमचं मूल इतर मुलांसोबत खेळण्यापेक्षा एकटं राहणं पसंत करतं का?",                  "quote": "प्रत्येक मूल अद्वितीय आहे आणि स्वतःच्या गतीने विकसित होते",         "image": "child_alone.png"},
        {"key": "avoids.people.or.activities", "text": "तुमचं मूल ज्या लोकांना किंवा गोष्टी आधी आवडायच्या त्यांना टाळतं का?",             "quote": "समजून घेणे चांगल्या समर्थनाकडे नेते",                               "image": "child_activities.png"},
        {"key": "close_friend_q",              "text": "तुमच्या मुलाला जवळचे मित्र करण्यात किंवा ठेवण्यात अडचण येते का?",                "quote": "मैत्रीचे कौशल्य शिकले आणि विकसित केले जाऊ शकते",                   "image": "children_playing.png"},
        {"key": "repetitive.behaviour",        "text": "तुमचं मूल एकच हालचाल, क्रिया किंवा शब्द पुन्हा पुन्हा करतं का?",                 "quote": "पॅटर्न हा मार्ग आहे ज्याद्वारे मुले जगाला समजतात",                  "image": "child_routine.jpg"},
        {"key": "anger",                       "text": "तुमचं मूल अचानक कोणत्याही स्पष्ट कारणाशिवाय खूप रागावतं का?",                   "quote": "भावना संदेश आहेत — आम्हाला फक्त त्यांना समजण्याची गरज आहे",         "image": "child_emotions.png"},
        {"key": "having.nightmares",           "text": "तुमच्या मुलाला वारंवार वाईट स्वप्न पडतात किंवा रात्री घाबरून उठतं का?",          "quote": "झोप विकास आणि कल्याणासाठी महत्त्वाची आहे",                         "image": "child_sleeping.png"},
        {"key": "social.media.addiction",      "text": "तुमचं मूल फोन, टॅब्लेट किंवा कॉम्प्युटरवर खूप जास्त वेळ घालवतं का?",           "quote": "डिजिटल युगात संतुलन महत्वाचे आहे",                                 "image": "child_tablet.png"},
        {"key": "over.react",                  "text": "तुमचं मूल लहान बदल किंवा समस्यांवर खूप जास्त प्रतिक्रिया देतं का?",              "quote": "समजल्यावर संवेदनशीलता शक्ती असू शकते",                              "image": "child_change.png"},
        {"key": "panic",                       "text": "तुमच्या मुलाला अचानक खूप भीती वाटते किंवा घाबरण्याचे झटके येतात का?",           "quote": "तुमची जागरूकता मदतीची पहिली पायरी आहे",                             "image": "child_comfort.png"},
        {"key": "change.in.eating",            "text": "तुमच्या मुलाच्या खाण्याच्या सवयी अलीकडे खूप बदलल्या आहेत का?",                 "quote": "पोषण मनःस्थिती आणि वर्तनावर परिणाम करते",                          "image": "child_eating.png"},
        {"key": "breathing.rapidly",           "text": "तुमचं मूल काळजी वाटल्यावर खूप वेगाने श्वास घेतं का?",                           "quote": "तुम्ही उत्तरे शोधून चांगले करत आहात",                               "image": "child_calm.png"},
        {"key": "anger_q2",                    "text": "तुमच्या मुलाला अस्वस्थ झाल्यानंतर शांत होण्यात अडचण येते का?",                  "quote": "भावनिक नियमन हे एक कौशल्य आहे जे विकसित केले जाऊ शकते",            "image": "child_emotions.png",   "maps_to": "anger"},
        {"key": "repetitive_q2",               "text": "तुमचं मूल कठोर दिनचर्येवर आग्रह धरतं आणि बदल झाल्यावर अस्वस्थ होतं का?",       "quote": "रचना आणि अनुमानयोग्यता मुलांना सुरक्षित वाटण्यास मदत करते",        "image": "child_routine.jpg",    "maps_to": "repetitive.behaviour"},
        {"key": "social_q2",                   "text": "तुमच्या मुलाला इतरांच्या भावना किंवा अभिव्यक्ती समजण्यात अडचण येते का?",        "quote": "सहानुभूती सौम्य मार्गदर्शनाने वाढते",                               "image": "children_playing.png", "maps_to": "introvert"},
        {"key": "sensory_q",                   "text": "तुमचं मूल आवाज, प्रकाश किंवा स्पर्शाबद्दल असामान्यपणे संवेदनशील आहे का?",      "quote": "संवेदी अनुभव मुले जगाशी कसे संवाद साधतात ते आकार देतात",           "image": "child_comfort.png",    "maps_to": "avoids.people.or.activities"},
    ],
}

RECOMMENDED_DOCTORS = [
    {"name": "Dr. Priya Sharma",    "specialty": "Child Psychologist & ASD Specialist", "hospital": "Pune Children's Development Center", "address": "Shop No. 123, Karve Road, Pune - 411004", "phone": "+91 20 2567 8901", "email": "dr.priya@punecdc.com"},
    {"name": "Dr. Rajesh Deshmukh", "specialty": "Pediatric Psychiatrist",               "hospital": "Modern Child Care Hospital",          "address": "Lane 5, Shivajinagar, Pune - 411005",   "phone": "+91 20 2568 9012", "email": "dr.rajesh@modernchildcare.in"},
    {"name": "Dr. Anita Kulkarni",  "specialty": "Developmental Pediatrician",           "hospital": "Sunrise Autism Center",               "address": "Baner Road, Pune - 411045",             "phone": "+91 20 2569 0123", "email": "dr.anita@sunriseautism.org"},
]


# =============================================================================
# FEATURE SELECTION  ★ NEW
# =============================================================================
def select_features(X, y, feature_names, n_select=N_SELECT):
    """
    Three-method consensus feature selection.

    METHOD 1 — ANOVA F-test:
      Computes F = between-class variance / within-class variance for each feature.
      High F means the feature clearly separates ASD from non-ASD.
      Best for linear separability. Constant columns get F=0 (safe).

    METHOD 2 — Mutual Information:
      Measures how much knowing a feature reduces uncertainty about the ASD label.
      MI=0 means feature and label are independent. Captures non-linear patterns.

    METHOD 3 — RFE with RandomForest:
      Trains RF, ranks features by importance, removes weakest iteratively
      until n_select remain. Accounts for feature interactions.

    CONSENSUS:
      Feature is selected if it appears in top-N of >= 2 methods.
      Ties broken by ANOVA score. If consensus < n_select, fill from ANOVA.
    """
    n = len(feature_names)

    # Method 1: ANOVA
    f_scores, _ = f_classif(X, y)
    f_scores     = np.nan_to_num(f_scores, nan=0.0)
    anova_ranked = sorted(range(n), key=lambda i: f_scores[i], reverse=True)
    top_anova    = set(anova_ranked[:n_select])

    # Method 2: Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_ranked = sorted(range(n), key=lambda i: mi_scores[i], reverse=True)
    top_mi    = set(mi_ranked[:n_select])

    # Method 3: RFE with RandomForest
    rf_rfe = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rfe    = RFE(estimator=rf_rfe, n_features_to_select=n_select, step=1)
    rfe.fit(X, y)
    top_rfe = set(i for i in range(n) if rfe.support_[i])

    # Consensus vote
    votes = {i: int(i in top_anova) + int(i in top_mi) + int(i in top_rfe)
             for i in range(n)}
    consensus_indices = sorted(
        [i for i, v in votes.items() if v >= 2],
        key=lambda i: f_scores[i], reverse=True
    )
    selected = [feature_names[i] for i in consensus_indices]

    # Fallback: fill from ANOVA if consensus < n_select
    if len(selected) < n_select:
        for i in anova_ranked:
            fn = feature_names[i]
            if fn not in selected:
                selected.append(fn)
            if len(selected) >= n_select:
                break

    report = {
        'anova':       [(feature_names[i], round(float(f_scores[i]), 2)) for i in anova_ranked[:n_select]],
        'mutual_info': [(feature_names[i], round(float(mi_scores[i]), 4)) for i in mi_ranked[:n_select]],
        'rfe':         [feature_names[i] for i in range(n) if rfe.support_[i]],
        'votes':       {feature_names[i]: votes[i] for i in range(n)},
        'selected':    selected,
    }
    return selected, report


# =============================================================================
# preprocess_input  (unchanged logic, uses dynamic MODEL_FEATURES)
# =============================================================================
def preprocess_input(raw_dict):
    """Maps 15 form answers to MODEL_FEATURES. OR logic for Q12-Q15. Unchanged."""
    feat = {}
    for mf in MODEL_FEATURES:
        feat[mf] = int(raw_dict.get('close_friend_q', 0)) if mf == 'no_close_friend' \
                   else int(raw_dict.get(mf, 0))
    if int(raw_dict.get('anger_q2', 0)) and 'anger' in MODEL_FEATURES:
        feat['anger'] = 1
    if int(raw_dict.get('repetitive_q2', 0)) and 'repetitive.behaviour' in MODEL_FEATURES:
        feat['repetitive.behaviour'] = 1
    if int(raw_dict.get('social_q2', 0)) and 'introvert' in MODEL_FEATURES:
        feat['introvert'] = 1
    if int(raw_dict.get('sensory_q', 0)) and 'avoids.people.or.activities' in MODEL_FEATURES:
        feat['avoids.people.or.activities'] = 1
    return feat


# =============================================================================
# ASDModel CLASS
# =============================================================================
class ASDModel:
    def __init__(self):
        self.scaler                   = StandardScaler()
        self.model                    = None
        self.threshold                = 0.5
        self.feature_importances_     = None
        self.metrics                  = {}
        self.selected_features        = []   # ★ set by select_features()
        self.feature_selection_report = {}   # ★ per-method scores

    def _load_and_engineer(self, csv_path):
        df = pd.read_csv(csv_path)
        df.rename(columns={'ag+1:629e': 'age'}, inplace=True)
        df['is_asd']          = (df['Disorder'] == 'ASD').astype(int)
        df['no_close_friend'] = 1 - df['close.friend']
        available = [f for f in CANDIDATE_FEATURES if f in df.columns]
        X = df[available].fillna(0).values
        y = df['is_asd'].values
        return X, y, available

    def train(self, csv_path):
        global MODEL_FEATURES
        X_all, y, candidate_names = self._load_and_engineer(csv_path)

        # ── FEATURE SELECTION ─────────────────────────────────────────────────
        print("  Running feature selection (ANOVA + Mutual Information + RFE)...")
        selected, report          = select_features(X_all, y, candidate_names, N_SELECT)
        self.selected_features        = selected
        self.feature_selection_report = report
        MODEL_FEATURES = selected
        print(f"  Selected {len(selected)} features: {selected}")

        # Reduce X to selected columns
        idx = [candidate_names.index(f) for f in selected]
        X   = X_all[:, idx]
        # ─────────────────────────────────────────────────────────────────────

        # ── ENSEMBLE (unchanged from v4) ──────────────────────────────────────
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_te_s = self.scaler.transform(X_te)

        rf  = RandomForestClassifier(n_estimators=200, max_depth=8,  random_state=42, class_weight='balanced')
        gb  = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42)
        et  = ExtraTreesClassifier(n_estimators=200, max_depth=8,  random_state=42, class_weight='balanced')
        ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=100, random_state=42)
        self.model = VotingClassifier(estimators=[('rf',rf),('gb',gb),('et',et),('ada',ada)], voting='soft')
        self.model.fit(X_tr_s, y_tr)

        probs = self.model.predict_proba(X_te_s)[:,1]
        best_t, best_f1 = 0.5, 0
        for t in np.arange(0.3, 0.75, 0.02):
            preds = (probs >= t).astype(int)
            f     = f1_score(y_te, preds, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        self.threshold = best_t
        preds = (probs >= self.threshold).astype(int)
        self.metrics = {
            'accuracy':  round(accuracy_score(y_te, preds)*100, 2),
            'precision': round(precision_score(y_te, preds, zero_division=0)*100, 2),
            'recall':    round(recall_score(y_te, preds, zero_division=0)*100, 2),
            'f1':        round(f1_score(y_te, preds, zero_division=0)*100, 2),
            'auc':       round(roc_auc_score(y_te, probs)*100, 2),
        }
        self.feature_importances_ = np.mean([
            est.feature_importances_ for est in [self.model.estimators_[0], self.model.estimators_[2]]
        ], axis=0)
        # ─────────────────────────────────────────────────────────────────────
        return self.metrics

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'model':                    self.model,
                'scaler':                   self.scaler,
                'threshold':                self.threshold,
                'metrics':                  self.metrics,
                'feature_importances':      self.feature_importances_,
                'selected_features':        self.selected_features,        # ★ NEW
                'feature_selection_report': self.feature_selection_report, # ★ NEW
            }, f)

    def load(self, path):
        global MODEL_FEATURES
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self.model                    = d['model']
        self.scaler                   = d['scaler']
        self.threshold                = d.get('threshold', 0.5)
        self.metrics                  = d.get('metrics', {})
        self.feature_importances_     = d.get('feature_importances')
        self.selected_features        = d.get('selected_features', [])
        self.feature_selection_report = d.get('feature_selection_report', {})
        # ★ Restore MODULE-LEVEL MODEL_FEATURES from saved list
        if self.selected_features:
            MODEL_FEATURES = self.selected_features
        elif not MODEL_FEATURES:
            # Backward-compat fallback for old .pkl files
            MODEL_FEATURES = ['introvert','avoids.people.or.activities','no_close_friend',
                              'repetitive.behaviour','anger','having.nightmares',
                              'social.media.addiction','over.react','panic',
                              'change.in.eating','breathing.rapidly']

    def predict(self, raw_dict):
        """Unchanged from v4."""
        feat = preprocess_input(raw_dict)
        x    = np.array([[feat[f] for f in MODEL_FEATURES]], dtype=float)
        x_s  = self.scaler.transform(x)
        prob = float(self.model.predict_proba(x_s)[0, 1])
        label = 'ASD Detected' if prob >= self.threshold else 'ASD Not Detected'
        if prob >= self.threshold:
            confidence = 'High' if prob >= self.threshold + 0.15 else 'Moderate'
        else:
            confidence = 'High' if prob <= self.threshold - 0.15 else 'Moderate'
        return prob, self.threshold, label, confidence

    def generate_shap_plot(self, raw_dict, out_dir, prefix):
        """Unchanged from v4."""
        feat = preprocess_input(raw_dict)
        x    = np.array([[feat[f] for f in MODEL_FEATURES]], dtype=float)
        x_s  = self.scaler.transform(x)
        try:
            import shap
            explainer = shap.TreeExplainer(self.model.estimators_[0])
            shap_vals = explainer.shap_values(x_s)
            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]
            else:
                sv = shap_vals[0] if shap_vals.ndim == 3 else shap_vals[0]
            contributions = [(MODEL_FEATURES[i], float(sv[i]),
                              'increases' if sv[i] >= 0 else 'decreases')
                             for i in range(len(MODEL_FEATURES))]
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        except Exception:
            fi = self.feature_importances_ if self.feature_importances_ is not None \
                 else np.ones(len(MODEL_FEATURES)) / len(MODEL_FEATURES)
            contributions = [(MODEL_FEATURES[i], float(fi[i] * x[0][i]),
                              'increases' if x[0][i] > 0 else 'neutral')
                             for i in range(len(MODEL_FEATURES))]
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        top    = contributions[:8]
        labels = [FEATURE_LABELS.get(f, f) for f, _, _ in top]
        values = [v for _, v, _ in top]
        colors = ['#4A90D9' if v >= 0 else '#7FB069' for v in values]
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor('#F8F9FA')
        ax.set_facecolor('#F8F9FA')
        ax.barh(range(len(top)), values, color=colors, alpha=0.85, height=0.6)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([l[:40] for l in labels], fontsize=9)
        ax.set_xlabel('Contribution to ASD Probability', fontsize=9)
        ax.set_title('Key Behavioral Indicators', fontsize=12, fontweight='bold', color='#1a1a2e')
        ax.axvline(0, color='#999', linewidth=0.8, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plot_path = os.path.join(out_dir, f'{prefix}_{int(time.time())}.png')
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(plot_path, dpi=120, bbox_inches='tight', facecolor='#F8F9FA')
        plt.close()
        return contributions, plot_path
