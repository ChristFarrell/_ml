# data/db.py — SQLite storage untuk history analisis
#
# Schema sederhana di V1. V3 nanti akan migrasi ke vector DB untuk memory.

import sqlite3
import json
import os
from ..models.schemas import AnalysisReport
from .. import config


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
