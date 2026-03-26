"""
train_model.py — AutexAI v6
============================
Run once to train and save the model with automatic feature selection.
Usage: python train_model.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.ml_model import ASDModel, MODEL_FEATURES
from config import DATASET_PATH, MODEL_PATH

def main():
    print("=" * 62)
    print("  AutexAI v6 — ANOVA + MI + RFE Feature Selection + Ensemble")
    print("=" * 62)
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at {DATASET_PATH}"); return

    model = ASDModel()
    m     = model.train(DATASET_PATH)

    # Print feature selection report
    rep = model.feature_selection_report
    print("\n── ANOVA F-scores (top selected) ──────────────────────────")
    for name, score in rep.get('anova', []):
        marker = "✓" if name in rep['selected'] else " "
        print(f"  [{marker}] {score:8.2f}  {name}")

    print("\n── Mutual Information ─────────────────────────────────────")
    for name, score in rep.get('mutual_info', []):
        marker = "✓" if name in rep['selected'] else " "
        print(f"  [{marker}] {score:.4f}  {name}")

    print("\n── RFE Selected ───────────────────────────────────────────")
    for name in rep.get('rfe', []):
        marker = "✓" if name in rep['selected'] else " "
        print(f"  [{marker}] {name}")

    print("\n── CONSENSUS: Final Selected Features ──────────────────────")
    for name in rep.get('selected', []):
        votes = rep.get('votes', {}).get(name, '?')
        print(f"  votes={votes}  {name}")

    print(f"\n── Model Metrics ───────────────────────────────────────────")
    print(f"  Accuracy  : {m['accuracy']}%")
    print(f"  Precision : {m['precision']}%")
    print(f"  Recall    : {m['recall']}%")
    print(f"  F1 Score  : {m['f1']}%")
    print(f"  ROC AUC   : {m['auc']}%")

    model.save(MODEL_PATH)
    print(f"\n  Model saved → {MODEL_PATH}")

    # Sanity check
    all_yes = {q: 1 for q in ['introvert','avoids.people.or.activities','close_friend_q',
        'repetitive.behaviour','anger','having.nightmares','social.media.addiction',
        'over.react','panic','change.in.eating','breathing.rapidly']}
    all_no = {q: 0 for q in all_yes}
    py, t, ly, cy = model.predict(all_yes)
    pn, _, ln, _  = model.predict(all_no)
    print(f"\n  All-YES: {py*100:.2f}%  → {ly} ({cy})")
    print(f"  All-NO : {pn*100:.2f}%  → {ln}")
    print("\nDone!")

if __name__ == '__main__':
    main()
