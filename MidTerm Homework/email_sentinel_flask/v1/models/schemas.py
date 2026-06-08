# models/schemas.py — Dataclass untuk semua struktur data V1

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class RiskLevel(Enum):
    UNKNOWN  = "UNKNOWN"
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class ParsedEmail:
    """Hasil parsing dari email address atau raw header."""
    raw_input       : str
    from_address    : Optional[str] = None
    from_name       : Optional[str] = None
    from_domain     : Optional[str] = None
    reply_to        : Optional[str] = None
    return_path     : Optional[str] = None
    subject         : Optional[str] = None
    originating_ip  : Optional[str] = None
    received_chain  : list[str]     = field(default_factory=list)
    extra_headers   : dict          = field(default_factory=dict)


@dataclass
class EmailSignals:
    """Sinyal risiko yang diekstrak dari ParsedEmail."""
    # Domain
    domain                  : str  = ""
    tld                     : str  = ""
    is_suspicious_tld       : bool = False
    is_free_email           : bool = False

    # Typosquatting & impersonation
    has_digit_substitution  : bool = False   # paypa1 → paypal
    has_homoglyph           : bool = False   # pаypal (cyrillic 'а')
    impersonates_brand      : Optional[str] = None
    typosquat_target        : Optional[str] = None

    # Header anomali
    reply_to_mismatch       : bool = False   # Reply-To ≠ From
    return_path_mismatch    : bool = False
    has_suspicious_subject  : bool = False
    subject_keywords        : list[str] = field(default_factory=list)

    # IP & routing
    originating_ip          : Optional[str] = None
    received_hop_count      : int  = 0

    # Skor kalkulasi lokal (sebelum AI)
    pre_score               : int  = 0
    signal_notes            : list[str] = field(default_factory=list)


@dataclass
class AnalysisReport:
    """Laporan final dari analyzer."""
    email_input     : str
    parsed          : ParsedEmail
    signals         : EmailSignals
    risk_score      : int           = 0
    risk_level      : RiskLevel     = RiskLevel.UNKNOWN
    ai_analysis     : str           = ""
    recommendation  : str           = ""
    created_at      : str           = ""