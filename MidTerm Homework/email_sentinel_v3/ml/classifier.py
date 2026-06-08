# ml/classifier.py — Phishing classifier (Random Forest + Logistic Regression)
#
# Flow:
#   1. User label email via API: POST /ml/label  {"scan_id": x, "label": "phishing"/"legit"}
#   2. Labels disimpan di DB (tabel ml_labels)
#   3. Saat data >= MIN_SAMPLES: train() otomatis dipanggil
#   4. Setelah trained: predict() dipakai di analyzer.py sebagai scoring tambahan
#   5. Model di-persist ke data/ml_model.pkl

from __future__ import annotations

import os
import pickle
import threading
import datetime
from typing import Optional, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.features import FEATURE_NAMES, N_FEATURES

# ── Config ────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "ml_model.pkl")
MIN_SAMPLES = 10   # minimum labeled samples sebelum bisa train
MIN_PER_CLASS = 3  # minimum per kelas (phishing & legit)

# ── Thread-safe model state ───────────────────────────────────────
_lock        = threading.Lock()
_rf_model    = None   # RandomForestClassifier
_lr_model    = None   # LogisticRegression (ensemble kedua)
_is_trained  = False
_train_meta  : dict  = {}


# ── Public API ────────────────────────────────────────────────────

def predict(features: list[float]) -> Optional[dict]:
    """
    Prediksi probabilitas phishing dari feature vector.
    Return dict dengan:
      - ml_score     : float 0-100 (probabilitas phishing × 100)
      - confidence   : "high"/"medium"/"low"
      - model_used   : "ensemble"/"rf"/"lr"/"none"
    Return None jika model belum trained.
    """
    with _lock:
        if not _is_trained or (_rf_model is None and _lr_model is None):
            return None

    try:
        import numpy as np
        X = np.array([features])

        probs = []
        models_used = []

        if _rf_model is not None:
            p = _rf_model.predict_proba(X)[0][1]  # prob of class 1 (phishing)
            probs.append(p)
            models_used.append("rf")

        if _lr_model is not None:
            p = _lr_model.predict_proba(X)[0][1]
            probs.append(p)
            models_used.append("lr")

        avg_prob   = sum(probs) / len(probs)
        ml_score   = round(avg_prob * 100, 1)

        # Confidence berdasarkan jarak dari 0.5
        dist = abs(avg_prob - 0.5)
        confidence = "high" if dist > 0.3 else ("medium" if dist > 0.15 else "low")

        return {
            "ml_score"   : ml_score,
            "ml_prob"    : round(avg_prob, 4),
            "confidence" : confidence,
            "model_used" : "+".join(models_used),
            "trained_on" : _train_meta.get("n_samples", 0),
            "trained_at" : _train_meta.get("trained_at", ""),
        }

    except Exception as e:
        print(f"[ML] predict error: {e}", flush=True)
        return None


def train(labeled_data: list[dict]) -> dict:
    """
    Train model dari labeled_data.
    labeled_data: list of {"features": [...], "label": 1/0}
      label: 1 = phishing, 0 = legit

    Return dict berisi metrics.
    """
    global _rf_model, _lr_model, _is_trained, _train_meta

    if len(labeled_data) < MIN_SAMPLES:
        return {"ok": False, "reason": f"Need at least {MIN_SAMPLES} labeled samples, got {len(labeled_data)}"}

    import numpy as np
    from sklearn.ensemble          import RandomForestClassifier
    from sklearn.linear_model      import LogisticRegression
    from sklearn.model_selection   import cross_val_score, StratifiedKFold
    from sklearn.preprocessing     import StandardScaler
    from sklearn.pipeline          import Pipeline

    X = np.array([d["features"] for d in labeled_data])
    y = np.array([d["label"]    for d in labeled_data])

    n_phishing = int(y.sum())
    n_legit    = int((y == 0).sum())

    if n_phishing < MIN_PER_CLASS or n_legit < MIN_PER_CLASS:
        return {
            "ok"    : False,
            "reason": f"Need >={MIN_PER_CLASS} samples per class. "
                      f"Got phishing={n_phishing}, legit={n_legit}",
        }

    # ── Random Forest ─────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators  = 100,
        max_depth     = 6,
        min_samples_leaf = 2,
        class_weight  = "balanced",   # handle imbalanced classes
        random_state  = 42,
    )

    # ── Logistic Regression (with scaling) ────────────────────────
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C            = 1.0,
            class_weight = "balanced",
            max_iter     = 500,
            random_state = 42,
        )),
    ])

    # ── Cross-validation (hanya kalau cukup data) ─────────────────
    cv_scores_rf = []
    cv_scores_lr = []
    feature_importances = {}

    n_splits = min(5, n_phishing, n_legit)  # jangan lebih dari jumlah sampel terkecil
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores_rf = cross_val_score(rf, X, y, cv=cv, scoring="f1").tolist()
        cv_scores_lr = cross_val_score(lr, X, y, cv=cv, scoring="f1").tolist()

    # ── Final training pakai semua data ───────────────────────────
    rf.fit(X, y)
    lr.fit(X, y)

    # Feature importance dari RF
    feature_importances = {
        name: round(float(imp), 4)
        for name, imp in zip(FEATURE_NAMES, rf.feature_importances_)
    }
    # Sort descending
    feature_importances = dict(
        sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    )

    meta = {
        "ok"                  : True,
        "n_samples"           : len(labeled_data),
        "n_phishing"          : n_phishing,
        "n_legit"             : n_legit,
        "cv_f1_rf"            : round(sum(cv_scores_rf)/len(cv_scores_rf), 3) if cv_scores_rf else None,
        "cv_f1_lr"            : round(sum(cv_scores_lr)/len(cv_scores_lr), 3) if cv_scores_lr else None,
        "feature_importances" : feature_importances,
        "trained_at"          : datetime.datetime.now().isoformat(),
    }

    with _lock:
        _rf_model   = rf
        _lr_model   = lr
        _is_trained = True
        _train_meta = meta

    # Persist ke disk
    _save_model()
    print(f"[ML] Model trained: {n_samples} samples, "
          f"RF F1={meta['cv_f1_rf']}, LR F1={meta['cv_f1_lr']}", flush=True)

    return meta


def load_model() -> bool:
    """Load model dari disk kalau ada. Return True jika berhasil."""
    global _rf_model, _lr_model, _is_trained, _train_meta
    if not os.path.exists(MODEL_PATH):
        return False
    try:
        with open(MODEL_PATH, "rb") as f:
            saved = pickle.load(f)
        with _lock:
            _rf_model   = saved.get("rf")
            _lr_model   = saved.get("lr")
            _train_meta = saved.get("meta", {})
            _is_trained = _rf_model is not None or _lr_model is not None
        print(f"[ML] Model loaded from {MODEL_PATH} "
              f"(trained on {_train_meta.get('n_samples','?')} samples)", flush=True)
        return True
    except Exception as e:
        print(f"[ML] Failed to load model: {e}", flush=True)
        return False


def get_status() -> dict:
    """Return status model untuk API /ml/status."""
    with _lock:
        return {
            "trained"    : _is_trained,
            "model_path" : MODEL_PATH,
            "min_samples": MIN_SAMPLES,
            **_train_meta,
        }


def _save_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"rf": _rf_model, "lr": _lr_model, "meta": _train_meta}, f)
    print(f"[ML] Model saved to {MODEL_PATH}", flush=True)


# Fix typo di train()
def train(labeled_data: list[dict]) -> dict:  # noqa: F811 — intentional redefinition to fix typo
    global _rf_model, _lr_model, _is_trained, _train_meta

    if len(labeled_data) < MIN_SAMPLES:
        return {"ok": False, "reason": f"Need at least {MIN_SAMPLES} labeled samples, got {len(labeled_data)}"}

    import numpy as np
    from sklearn.ensemble          import RandomForestClassifier
    from sklearn.linear_model      import LogisticRegression
    from sklearn.model_selection   import cross_val_score, StratifiedKFold
    from sklearn.preprocessing     import StandardScaler
    from sklearn.pipeline          import Pipeline

    X = np.array([d["features"] for d in labeled_data])
    y = np.array([d["label"]    for d in labeled_data])

    n_phishing = int(y.sum())
    n_legit    = int((y == 0).sum())

    if n_phishing < MIN_PER_CLASS or n_legit < MIN_PER_CLASS:
        return {
            "ok"    : False,
            "reason": f"Need >={MIN_PER_CLASS} samples per class. "
                      f"Got phishing={n_phishing}, legit={n_legit}",
        }

    rf = RandomForestClassifier(
        n_estimators     = 100,
        max_depth        = 6,
        min_samples_leaf = 2,
        class_weight     = "balanced",
        random_state     = 42,
    )
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=500, random_state=42,
        )),
    ])

    cv_scores_rf, cv_scores_lr = [], []
    n_splits = min(5, n_phishing, n_legit)
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores_rf = cross_val_score(rf, X, y, cv=cv, scoring="f1").tolist()
        cv_scores_lr = cross_val_score(lr, X, y, cv=cv, scoring="f1").tolist()

    rf.fit(X, y)
    lr.fit(X, y)

    feature_importances = dict(sorted(
        {name: round(float(imp), 4)
         for name, imp in zip(FEATURE_NAMES, rf.feature_importances_)}.items(),
        key=lambda x: x[1], reverse=True
    ))

    n_samples = len(labeled_data)
    meta = {
        "ok"                  : True,
        "n_samples"           : n_samples,
        "n_phishing"          : n_phishing,
        "n_legit"             : n_legit,
        "cv_f1_rf"            : round(sum(cv_scores_rf)/len(cv_scores_rf), 3) if cv_scores_rf else None,
        "cv_f1_lr"            : round(sum(cv_scores_lr)/len(cv_scores_lr), 3) if cv_scores_lr else None,
        "feature_importances" : feature_importances,
        "trained_at"          : datetime.datetime.now().isoformat(),
    }

    with _lock:
        _rf_model   = rf
        _lr_model   = lr
        _is_trained = True
        _train_meta = meta

    _save_model()
    print(f"[ML] Trained on {n_samples} samples | "
          f"RF F1={meta['cv_f1_rf']} | LR F1={meta['cv_f1_lr']}", flush=True)
    return meta
