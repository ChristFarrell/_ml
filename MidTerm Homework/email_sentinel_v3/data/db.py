# data/db.py — SQLite storage untuk history analisis
#
# Schema sederhana di V1. V3 nanti akan migrasi ke vector DB untuk memory.

import sqlite3
import json
import os
from models.schemas import AnalysisReport
import config


def init_db() -> None:
    """Buat tabel jika belum ada."""
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                email       TEXT,
                domain      TEXT,
                risk_score  INTEGER,
                risk_level  TEXT,
                signals     TEXT,      -- JSON array
                ai_analysis TEXT,
                created_at  TEXT
            )
        """)
        # Index untuk query cepat per domain (penting untuk V3 memory)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_domain ON analyses (domain)
        """)


def save_report(report: AnalysisReport) -> int:
    """Simpan laporan, return ID baris yang baru disimpan."""
    init_db()
    with _conn() as conn:
        cur = conn.execute("""
            INSERT INTO analyses (email, domain, risk_score, risk_level, signals, ai_analysis, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            report.parsed.from_address,
            report.signals.domain,
            report.risk_score,
            report.risk_level.value,
            json.dumps(report.signals.signal_notes, ensure_ascii=False),
            report.ai_analysis,
            report.created_at,
        ))
        return cur.lastrowid


def get_domain_history(domain: str) -> list[dict]:
    """
    Ambil semua analisis sebelumnya untuk domain ini.
    Berguna untuk V3 memory: apakah domain ini pernah flagged?
    """
    init_db()
    with _conn() as conn:
        rows = conn.execute("""
            SELECT email, risk_score, risk_level, signals, created_at
            FROM analyses
            WHERE domain = ?
            ORDER BY created_at DESC
            LIMIT 20
        """, (domain,)).fetchall()

    return [
        {
            "email"      : row[0],
            "risk_score" : row[1],
            "risk_level" : row[2],
            "signals"    : json.loads(row[3]),
            "created_at" : row[4],
        }
        for row in rows
    ]


def get_recent(limit: int = 10) -> list[dict]:
    """Ambil N analisis terakhir — untuk dashboard V3."""
    init_db()
    with _conn() as conn:
        rows = conn.execute("""
            SELECT email, domain, risk_score, risk_level, created_at
            FROM analyses
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return [
        {
            "email"      : row[0],
            "domain"     : row[1],
            "risk_score" : row[2],
            "risk_level" : row[3],
            "created_at" : row[4],
        }
        for row in rows
    ]


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(config.DB_PATH)


# ── ML Labels ─────────────────────────────────────────────────────
# Tabel baru: user konfirmasi setiap email → phishing / legit
# Label ini yang dipakai untuk training Random Forest

def init_ml_labels() -> None:
    """Buat tabel ml_labels + features jika belum ada."""
    init_db()
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_labels (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER,          -- FK ke tabel analyses (opsional)
                email       TEXT,
                domain      TEXT,
                label       INTEGER NOT NULL, -- 1 = phishing, 0 = legit
                features    TEXT NOT NULL,    -- JSON array of 15 floats
                source      TEXT DEFAULT 'user',  -- 'user' atau 'auto'
                created_at  TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_label ON ml_labels (label)
        """)


def save_label(email: str, domain: str, label: int,
               features: list, analysis_id: int = None,
               source: str = "user") -> int:
    """
    Simpan label dari user.
    label: 1 = phishing, 0 = legit
    features: list of float dari ml/features.py
    """
    init_ml_labels()
    import datetime
    with _conn() as conn:
        cur = conn.execute("""
            INSERT INTO ml_labels (analysis_id, email, domain, label, features, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            analysis_id,
            email,
            domain,
            int(label),
            json.dumps(features),
            source,
            datetime.datetime.now().isoformat(),
        ))
        return cur.lastrowid


def get_labeled_data() -> list[dict]:
    """
    Ambil semua labeled data untuk training.
    Return list of {"features": [...], "label": int, "email": str}
    """
    init_ml_labels()
    with _conn() as conn:
        rows = conn.execute("""
            SELECT features, label, email, domain, created_at
            FROM ml_labels
            ORDER BY created_at ASC
        """).fetchall()

    result = []
    for row in rows:
        try:
            result.append({
                "features"  : json.loads(row[0]),
                "label"     : int(row[1]),
                "email"     : row[2],
                "domain"    : row[3],
                "created_at": row[4],
            })
        except Exception:
            pass
    return result


def get_label_stats() -> dict:
    """Statistik label untuk ditampilkan di API /ml/status."""
    init_ml_labels()
    with _conn() as conn:
        total    = conn.execute("SELECT COUNT(*) FROM ml_labels").fetchone()[0]
        phishing = conn.execute("SELECT COUNT(*) FROM ml_labels WHERE label=1").fetchone()[0]
        legit    = conn.execute("SELECT COUNT(*) FROM ml_labels WHERE label=0").fetchone()[0]
    return {"total": total, "phishing": phishing, "legit": legit}


def bootstrap_labels_from_history(min_score_phishing: int = 80,
                                   max_score_legit: int = 20) -> int:
    """
    Bootstrap awal: auto-label dari historical analyses yang sudah ada.
    Email dengan risk_score >= min_score_phishing → label phishing (source='auto')
    Email dengan risk_score <= max_score_legit    → label legit (source='auto')
    Hanya label yang belum ada di ml_labels.
    Return jumlah baris yang di-label.
    """
    from ml.features import db_row_to_features
    init_ml_labels()

    # Ambil existing labeled emails agar tidak duplikat
    with _conn() as conn:
        existing = set(row[0] for row in
                       conn.execute("SELECT email FROM ml_labels").fetchall())

        rows = conn.execute("""
            SELECT id, email, domain, risk_score, risk_level, signals
            FROM analyses
            WHERE email IS NOT NULL
        """).fetchall()

    count = 0
    for row in rows:
        analysis_id, email, domain, score, level, signals_json = row
        if email in existing:
            continue

        if score >= min_score_phishing:
            label = 1
        elif score <= max_score_legit:
            label = 0
        else:
            continue  # zona abu-abu, skip

        try:
            signals_list = json.loads(signals_json) if signals_json else []
            features = db_row_to_features({
                "signals"    : signals_list,
                "risk_score" : score,
            })
            save_label(email, domain or "", label, features,
                       analysis_id=analysis_id, source="auto")
            existing.add(email)
            count += 1
        except Exception:
            pass

    return count
