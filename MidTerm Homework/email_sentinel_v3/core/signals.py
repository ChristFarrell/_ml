# core/signals.py — Extract risk signals from ParsedEmail (local checks, no network)

import re
import unicodedata
from models.schemas import ParsedEmail, EmailSignals
import config

SUSPICIOUS_SUBJECT_KEYWORDS = [
    # English
    "urgent", "action required", "verify your account", "suspended",
    "unusual activity", "click here", "confirm your", "password reset",
    "winner", "congratulations", "claim your", "invoice attached",
    "payment due", "limited time", "act now", "immediate action",
    "account disabled", "security alert", "login attempt",
    # Indonesian
    "segera", "verifikasi", "akun anda", "darurat",
    "konfirmasi", "hadiah", "menang", "klik di sini",
    "tindakan diperlukan",
]


def extract_signals(parsed: ParsedEmail) -> EmailSignals:
    """Run all local checks and return EmailSignals with pre_score."""
    signals = EmailSignals()
    domain  = parsed.from_domain or ""

    signals.domain = domain
    signals.tld    = _get_tld(domain)

    # Domains on the legit whitelist skip impersonation checks entirely
    is_legit = domain in config.LEGIT_DOMAINS

    # ── TLD ───────────────────────────────────────────────────────
    if not is_legit and signals.tld in config.SUSPICIOUS_TLDS:
        signals.is_suspicious_tld = True
        _note(signals, f"Suspicious TLD: {signals.tld}", score=15)

    # ── Free email provider (informational, no score) ─────────────
    if domain in config.FREE_EMAIL_DOMAINS:
        signals.is_free_email = True
        signals.signal_notes.append(f"Free email provider: {domain}")

    # ── Typosquatting: digit substitution ────────────────────────
    # Only check the label before the first dot (e.g. "paypa1" from "paypa1.com")
    label = domain.lower().split(".")[0]
    if not is_legit and re.search(r"[0-9]", label):
        for norm_fn in (_norm_l, _norm_i):
            hit = False
            for brand in config.IMPERSONATION_TARGETS:
                if brand in norm_fn(label) and brand != label:
                    signals.has_digit_substitution = True
                    signals.impersonates_brand     = brand
                    _note(signals,
                          f"Digit substitution: '{label}' resembles '{brand}'",
                          score=50)
                    hit = True
                    break
            if hit:
                break

    # ── Homoglyph attack ──────────────────────────────────────────
    if not is_legit and _has_homoglyph(domain):
        signals.has_homoglyph = True
        _note(signals, "Domain contains Unicode lookalike characters (homoglyph attack)", score=40)

    # ── Brand name in domain without being legit ──────────────────
    if not is_legit and not signals.impersonates_brand:
        for brand in config.IMPERSONATION_TARGETS:
            if brand in label and label != brand:
                signals.impersonates_brand = brand
                signals.typosquat_target   = brand
                _note(signals,
                      f"Domain contains brand name '{brand}' but is not the official domain",
                      score=25)
                break

    # ── Hard floor for impersonation / homoglyph ──────────────────
    # These are always at least MEDIUM severity regardless of other signals
    if signals.impersonates_brand or signals.has_homoglyph:
        signals.pre_score = max(signals.pre_score, 65)

    # ── Header anomalies ──────────────────────────────────────────
    if parsed.reply_to and parsed.from_address:
        from_dom    = parsed.from_address.split("@")[-1].lower()
        replyto_dom = parsed.reply_to.split("@")[-1].lower()
        if from_dom != replyto_dom:
            signals.reply_to_mismatch = True
            _note(signals,
                  f"Reply-To domain ({replyto_dom}) differs from From domain ({from_dom})",
                  score=25)

    if parsed.return_path and parsed.from_address:
        from_dom  = parsed.from_address.split("@")[-1].lower()
        rpath_dom = parsed.return_path.split("@")[-1].lower()
        if from_dom != rpath_dom:
            signals.return_path_mismatch = True
            _note(signals,
                  f"Return-Path domain ({rpath_dom}) differs from From domain ({from_dom})",
                  score=15)

    # ── Subject keywords ──────────────────────────────────────────
    if parsed.subject:
        subj_lower = parsed.subject.lower()
        found = [kw for kw in SUSPICIOUS_SUBJECT_KEYWORDS if kw in subj_lower]
        if found:
            signals.has_suspicious_subject = True
            signals.subject_keywords       = found
            _note(signals, f"Subject contains phishing keywords: {', '.join(found)}", score=10)

    # ── Routing ───────────────────────────────────────────────────
    signals.originating_ip     = parsed.originating_ip
    signals.received_hop_count = len(parsed.received_chain)
    if signals.received_hop_count > 8:
        _note(signals,
              f"Long routing chain: {signals.received_hop_count} hops (possible obfuscation)",
              score=10)

    signals.pre_score = min(signals.pre_score, 99)
    return signals


# ── Helpers ──────────────────────────────────────────────────────

def _get_tld(domain: str) -> str:
    parts = domain.lower().rsplit(".", 1)
    return f".{parts[-1]}" if len(parts) > 1 else ""

def _norm_l(s: str) -> str:
    """Digit→letter mapping where '1'→'l' (paypa1→paypal)."""
    return "".join({"0":"o","1":"l","3":"e","4":"a","5":"s","7":"t"}.get(c,c) for c in s)

def _norm_i(s: str) -> str:
    """Digit→letter mapping where '1'→'i' (m1crosoft→microsoft)."""
    return "".join({"0":"o","1":"i","3":"e","4":"a","5":"s","7":"t"}.get(c,c) for c in s)

def _has_homoglyph(domain: str) -> bool:
    for char in domain:
        if ord(char) > 127:
            try:
                name = unicodedata.name(char, "").lower()
                if any(w in name for w in ["latin", "cyrillic", "greek"]):
                    return True
            except Exception:
                return True
    return False

def _note(signals: EmailSignals, message: str, score: int = 0) -> None:
    signals.signal_notes.append(message)
    signals.pre_score += score
