# core/signals.py — Extract risk signals from ParsedEmail
#
# All checks here are purely local (no network calls).
# Network checks (WHOIS, DNS) go into V2.

import re
import unicodedata
from ..models.schemas import ParsedEmail, EmailSignals
from .. import config


# Common homoglyph mappings (Unicode characters that look like Latin letters)
HOMOGLYPHS = {
    'a': ['а', 'ä', 'â', 'à', 'á', 'ạ', 'ă'],  # cyrillic 'а' very common
    'e': ['е', 'ë', 'ê', 'è', 'é'],              # cyrillic 'е'
    'o': ['о', 'ö', 'ô', 'ò', 'ó', '0'],         # cyrillic 'о'
    'i': ['і', 'ï', 'î', 'ì', 'í', '1', 'l'],    # cyrillic 'і'
    'p': ['р'],                                   # cyrillic 'р'
    'c': ['с'],                                   # cyrillic 'с'
    'x': ['х'],                                   # cyrillic 'х'
}

# Subject keywords commonly found in phishing/spam
SUSPICIOUS_SUBJECT_KEYWORDS = [
    "urgent", "action required", "verify your account", "suspended",
    "unusual activity", "click here", "confirm your", "password",
    "winner", "congratulations", "claim your", "invoice", "payment due",
    "your account", "limited time", "act now", "immediate",
    # Indonesian
    "segera", "verifikasi", "akun anda", "darurat", "tindakan",
    "konfirmasi", "hadiah", "menang", "klik di sini",
]


def extract_signals(parsed: ParsedEmail) -> EmailSignals:
    """Run all checks and return EmailSignals with pre_score."""
    signals = EmailSignals()
    domain  = parsed.from_domain or ""

    # ── Base domain ───────────────────────────────────────────────
    signals.domain = domain
    signals.tld    = _get_tld(domain)

    # Flag whether this is a known-legit domain — used to skip impersonation check
    is_legit_domain = domain in config.LEGIT_DOMAINS

    if not is_legit_domain and signals.tld in config.SUSPICIOUS_TLDS:
        signals.is_suspicious_tld = True
        _note(signals, f"Suspicious TLD: {signals.tld}", score=15)

    if domain in config.FREE_EMAIL_DOMAINS:
        signals.is_free_email = True
        # Gmail etc. are noted but don't add score — this is normal
        signals.signal_notes.append(f"Free email domain: {domain}")

    # ── Typosquatting & impersonation ─────────────────────────────
    domain_lower = domain.lower().split(".")[0]  # grab part before first TLD

    # Digit substitution: letters replaced with numbers (paypa1, m1crosoft, g00gle)
    if not is_legit_domain and re.search(r'[0-9]', domain_lower):
        found_sub = False
        for norm_fn in (_normalize_digits, _normalize_digits_alt):
            normalized = norm_fn(domain_lower)
            for target in config.IMPERSONATION_TARGETS:
                if target in normalized and target != domain_lower:
                    signals.has_digit_substitution = True
                    signals.impersonates_brand      = target
                    _note(signals, f"Digit substitution: '{domain_lower}' resembles '{target}'", score=55)
                    found_sub = True
                    break
            if found_sub:
                break

    # Homoglyph attack: Unicode characters that look like Latin letters
    if not is_legit_domain and _has_homoglyph(domain):
        signals.has_homoglyph = True
        _note(signals, "Domain contains Unicode characters resembling Latin letters (homoglyph attack)", score=35)

    # Brand directly present in subdomain/domain without digit trick
    if not is_legit_domain and not signals.impersonates_brand:
        for target in config.IMPERSONATION_TARGETS:
            # Must contain the brand name BUT not be the real domain already whitelisted
            if target in domain_lower and domain_lower != target:
                # Stricter check: real domain is usually exactly "brand.com" or "brand.co.xx"
                # If there are extra words (support, secure, login, etc.) → suspicious
                full_domain = domain.lower()
                is_legit = full_domain in config.LEGIT_DOMAINS
                if not is_legit:
                    signals.impersonates_brand = target
                    signals.typosquat_target   = target
                    _note(signals, f"Domain contains brand name '{target}' but is not the official domain", score=20)
                break

    # ── Header anomalies ──────────────────────────────────────────
    if parsed.reply_to and parsed.from_address:
        from_dom   = parsed.from_address.split("@")[-1].lower()
        replyto_dom = parsed.reply_to.split("@")[-1].lower()
        if from_dom != replyto_dom:
            signals.reply_to_mismatch = True
            _note(signals, f"Reply-To ({replyto_dom}) differs from From ({from_dom})", score=25)

    if parsed.return_path and parsed.from_address:
        from_dom  = parsed.from_address.split("@")[-1].lower()
        rpath_dom = parsed.return_path.split("@")[-1].lower()
        if from_dom != rpath_dom:
            signals.return_path_mismatch = True
            _note(signals, f"Return-Path ({rpath_dom}) differs from From ({from_dom})", score=15)

    # ── Subject keywords ──────────────────────────────────────────
    if parsed.subject:
        subj_lower = parsed.subject.lower()
        found_kw   = [kw for kw in SUSPICIOUS_SUBJECT_KEYWORDS if kw in subj_lower]
        if found_kw:
            signals.has_suspicious_subject = True
            signals.subject_keywords        = found_kw
            _note(signals, f"Subject contains phishing keywords: {', '.join(found_kw)}", score=10)

    # ── Routing ───────────────────────────────────────────────────
    signals.originating_ip  = parsed.originating_ip
    signals.received_hop_count = len(parsed.received_chain)

    # Too many hops (often used for obfuscation)
    if signals.received_hop_count > 8:
        _note(signals, f"Long routing chain: {signals.received_hop_count} hops", score=10)

    # Floor score: brand impersonation must NOT pass as LOW
    if signals.impersonates_brand and signals.pre_score < 70:
        signals.pre_score = 70
    if signals.has_homoglyph and signals.pre_score < 70:
        signals.pre_score = 70

    signals.pre_score = min(signals.pre_score, 99)  # cap at 99, AI decides final
    return signals


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_tld(domain: str) -> str:
    """Get TLD from domain: 'evil.co.uk' → '.uk', 'phish.xyz' → '.xyz'."""
    parts = domain.lower().rsplit(".", 1)
    return f".{parts[-1]}" if len(parts) > 1 else ""


def _normalize_digits(s: str) -> str:
    """
    Try all possible digit → letter substitutions.
    '1' can mean 'l' (paypa1→paypal) or 'i' (m1crosoft→microsoft).
    This version returns '1→l' (most common in brand spoofing).
    """
    mapping = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "@": "a", "7": "t"}
    return "".join(mapping.get(c, c) for c in s)


def _normalize_digits_alt(s: str) -> str:
    """Alternative: 1→i for cases like m1crosoft→microsoft."""
    mapping = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "@": "a", "7": "t"}
    return "".join(mapping.get(c, c) for c in s)


def _has_homoglyph(domain: str) -> bool:
    """Check whether the domain contains non-ASCII characters that resemble Latin letters."""
    for char in domain:
        if ord(char) > 127:  # non-ASCII
            try:
                name = unicodedata.name(char, "").lower()
                if any(w in name for w in ["latin", "cyrillic", "greek"]):
                    return True
            except Exception:
                return True
    return False


def _note(signals: EmailSignals, message: str, score: int = 0) -> None:
    """Append a note to signals and accumulate the score."""
    signals.signal_notes.append(message)
    signals.pre_score += score