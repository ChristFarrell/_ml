# ml/features.py — Convert EmailSignals + DomainIntelligence → numeric feature vector
#
# Feature vector (15 features, semua numeric 0/1 atau integer):
#
#  0  is_suspicious_tld       bool
#  1  is_free_email           bool
#  2  has_digit_substitution  bool
#  3  has_homoglyph           bool
#  4  impersonates_brand      bool  (1 if any brand detected)
#  5  reply_to_mismatch       bool
#  6  return_path_mismatch    bool
#  7  has_suspicious_subject  bool
#  8  subject_keyword_count   int   (capped 0–5)
#  9  hop_count_high          bool  (>8 hops)
# 10  pre_score_norm          float (pre_score / 99.0)
# 11  spf_fail                bool  (V2: 0 if no intel)
# 12  dmarc_fail              bool  (V2: 0 if no intel)
# 13  no_mx_records           bool  (V2: 0 if no intel)
# 14  young_domain            bool  (V2: 0 if no intel)

from __future__ import annotations
from typing import Optional, List

FEATURE_NAMES = [
    "is_suspicious_tld",
    "is_free_email",
    "has_digit_substitution",
    "has_homoglyph",
    "impersonates_brand",
    "reply_to_mismatch",
    "return_path_mismatch",
    "has_suspicious_subject",
    "subject_keyword_count",
    "hop_count_high",
    "pre_score_norm",
    "spf_fail",
    "dmarc_fail",
    "no_mx_records",
    "young_domain",
]

N_FEATURES = len(FEATURE_NAMES)  # 15


def signals_to_features(signals, intel=None) -> List[float]:
    """
    Ubah EmailSignals (+ DomainIntelligence opsional) ke list of float.
    Selalu return vektor dengan panjang N_FEATURES.
    """
    kw_count = min(len(signals.subject_keywords), 5) if signals.subject_keywords else 0

    vec = [
        int(signals.is_suspicious_tld),
        int(signals.is_free_email),
        int(signals.has_digit_substitution),
        int(signals.has_homoglyph),
        int(bool(signals.impersonates_brand)),
        int(signals.reply_to_mismatch),
        int(signals.return_path_mismatch),
        int(signals.has_suspicious_subject),
        kw_count,
        int(signals.received_hop_count > 8),
        round(signals.pre_score / 99.0, 4),
        # V2 network features (default 0 if intel not available)
        int(intel is not None and intel.spf_valid is False),
        int(intel is not None and intel.dmarc_valid is False),
        int(intel is not None and not intel.has_mx_records),
        int(intel is not None and intel.is_young_domain),
    ]

    assert len(vec) == N_FEATURES, f"Feature count mismatch: {len(vec)} != {N_FEATURES}"
    return vec


def db_row_to_features(row: dict) -> Optional[List[float]]:
    """
    Rebuild feature vector dari row DB (untuk training dari historical data).
    row harus punya key: signals (list of signal_note strings) + risk_score.
    Ini rekonstruksi approximate — cukup untuk bootstrap awal.
    """
    notes = row.get("signals", [])
    score = row.get("risk_score", 0)

    def _has(keyword: str) -> int:
        return int(any(keyword.lower() in n.lower() for n in notes))

    kw_count = min(sum(1 for n in notes if "keyword" in n.lower()), 5)
    hop_high = _has("routing chain")

    vec = [
        _has("suspicious tld"),
        _has("free email"),
        _has("digit substitution"),
        _has("homoglyph"),
        _has("brand name") or _has("resembles"),
        _has("reply-to"),
        _has("return-path"),
        _has("phishing keyword"),
        kw_count,
        hop_high,
        round(score / 99.0, 4),
        _has("spf") and _has("invalid"),
        _has("dmarc") and _has("invalid"),
        _has("mx") and _has("missing"),
        _has("young domain") or _has("new domain"),
    ]
    return vec
