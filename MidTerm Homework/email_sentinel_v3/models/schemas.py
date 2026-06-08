# models/schemas.py — Data schemas for Email Sentinel V1/V2/V3

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
    """Result of parsing an email address or raw header."""
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
    """Risk signals extracted from ParsedEmail — local checks only, no network."""
    # Domain
    domain                  : str  = ""
    tld                     : str  = ""
    is_suspicious_tld       : bool = False
    is_free_email           : bool = False

    # Typosquatting & impersonation
    has_digit_substitution  : bool = False
    has_homoglyph           : bool = False
    impersonates_brand      : Optional[str] = None
    typosquat_target        : Optional[str] = None

    # Header anomalies
    reply_to_mismatch       : bool = False
    return_path_mismatch    : bool = False
    has_suspicious_subject  : bool = False
    subject_keywords        : list[str] = field(default_factory=list)

    # IP & routing
    originating_ip          : Optional[str] = None
    received_hop_count      : int  = 0

    # Local score (before AI and network)
    pre_score               : int  = 0
    signal_notes            : list[str] = field(default_factory=list)


@dataclass
class DomainIntelligence:
    """Network investigation results from V2 investigators."""
    domain              : str  = ""

    # WHOIS
    registrar           : Optional[str] = None
    creation_date       : Optional[str] = None
    domain_age_days     : Optional[int] = None
    is_young_domain     : bool = False
    registrant_country  : Optional[str] = None

    # MX
    has_mx_records      : bool = False
    mx_records          : list[str] = field(default_factory=list)

    # SPF
    spf_valid           : Optional[bool] = None
    spf_record          : Optional[str]  = None

    # DKIM
    dkim_valid          : Optional[bool] = None

    # DMARC
    dmarc_valid         : Optional[bool] = None
    dmarc_policy        : Optional[str]  = None  # none / quarantine / reject

    # Network score contribution
    net_score           : int  = 0
    net_notes           : list[str] = field(default_factory=list)
    errors              : list[str] = field(default_factory=list)


@dataclass
class AnalysisReport:
    """Final analysis report — V1/V2/V3 compatible."""
    email_input     : str
    parsed          : ParsedEmail
    signals         : EmailSignals
    intel           : Optional[DomainIntelligence] = None
    risk_score      : int       = 0
    risk_level      : RiskLevel = RiskLevel.UNKNOWN
    ai_analysis     : str       = ""
    recommendation  : str       = ""
    created_at      : str       = ""
