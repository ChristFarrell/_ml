# core/signals.py — Ekstrak sinyal risiko dari ParsedEmail
#
# Semua pengecekan di sini murni lokal (no network calls).
# Network checks (WHOIS, DNS) akan masuk di V2.

import re
import unicodedata
from ..models.schemas import ParsedEmail, EmailSignals
from .. import config


# Pemetaan homoglyph umum (Unicode karakter yang mirip huruf Latin)
HOMOGLYPHS = {
    'a': ['а', 'ä', 'â', 'à', 'á', 'ạ', 'ă'],  # cyrillic 'а' sangat umum
    'e': ['е', 'ë', 'ê', 'è', 'é'],              # cyrillic 'е'
    'o': ['о', 'ö', 'ô', 'ò', 'ó', '0'],         # cyrillic 'о'
    'i': ['і', 'ï', 'î', 'ì', 'í', '1', 'l'],    # cyrillic 'і'
    'p': ['р'],                                   # cyrillic 'р'
    'c': ['с'],                                   # cyrillic 'с'
    'x': ['х'],                                   # cyrillic 'х'
}

# Keyword subject yang sering ada di phishing/spam
SUSPICIOUS_SUBJECT_KEYWORDS = [
    "urgent", "action required", "verify your account", "suspended",
    "unusual activity", "click here", "confirm your", "password",
    "winner", "congratulations", "claim your", "invoice", "payment due",
    "your account", "limited time", "act now", "immediate",
    # Bahasa Indonesia
    "segera", "verifikasi", "akun anda", "darurat", "tindakan",
    "konfirmasi", "hadiah", "menang", "klik di sini",
]


def extract_signals(parsed: ParsedEmail) -> EmailSignals:
    """Jalanka semua check dan kembalikan EmailSignals dengan pre_score."""
    signals = EmailSignals()
    domain  = parsed.from_domain or ""

    # ── Domain dasar ──────────────────────────────────────────────
    signals.domain = domain
    signals.tld    = _get_tld(domain)

    # Flag apakah ini domain resmi — dipakai untuk skip impersonation check
    is_legit_domain = domain in config.LEGIT_DOMAINS

    if not is_legit_domain and signals.tld in config.SUSPICIOUS_TLDS:
        signals.is_suspicious_tld = True
        _note(signals, f"TLD mencurigakan: {signals.tld}", score=15)

    if domain in config.FREE_EMAIL_DOMAINS:
        signals.is_free_email = True
        # Gmail dll tetap dicatat tapi tidak tambah skor — ini normal
        signals.signal_notes.append(f"Domain email gratis: {domain}")

    # ── Typosquatting & impersonation ─────────────────────────────
    domain_lower = domain.lower().split(".")[0]  # ambil bagian sebelum TLD pertama

    # Digit substitution: huruf diganti angka (paypa1, m1crosoft, g00gle)
    if not is_legit_domain and re.search(r'[0-9]', domain_lower):
        found_sub = False
        for norm_fn in (_normalize_digits, _normalize_digits_alt):
            normalized = norm_fn(domain_lower)
            for target in config.IMPERSONATION_TARGETS:
                if target in normalized and target != domain_lower:
                    signals.has_digit_substitution = True
                    signals.impersonates_brand      = target
                    _note(signals, f"Digit substitution: '{domain_lower}' mirip '{target}'", score=55)
                    found_sub = True
                    break
            if found_sub:
                break

    # Homoglyph attack: karakter Unicode mirip Latin
    if not is_legit_domain and _has_homoglyph(domain):
        signals.has_homoglyph = True
        _note(signals, "Domain mengandung karakter Unicode mirip Latin (homoglyph attack)", score=35)

    # Brand langsung ada di subdomain/domain tanpa digit trick
    if not is_legit_domain and not signals.impersonates_brand:
        for target in config.IMPERSONATION_TARGETS:
            # Harus mengandung nama brand TAPI bukan domain resmi yang sudah di-whitelist
            if target in domain_lower and domain_lower != target:
                # Cek lebih ketat: domain asli biasanya tepat "brand.com" atau "brand.co.xx"
                # Kalau ada kata tambahan (support, secure, login, dll) → curigai
                full_domain = domain.lower()
                is_legit = full_domain in config.LEGIT_DOMAINS
                if not is_legit:
                    signals.impersonates_brand = target
                    signals.typosquat_target   = target
                    _note(signals, f"Domain mengandung nama brand '{target}' tapi bukan domain resmi", score=20)
                break

    # ── Header anomali ────────────────────────────────────────────
    if parsed.reply_to and parsed.from_address:
        from_dom   = parsed.from_address.split("@")[-1].lower()
        replyto_dom = parsed.reply_to.split("@")[-1].lower()
        if from_dom != replyto_dom:
            signals.reply_to_mismatch = True
            _note(signals, f"Reply-To ({replyto_dom}) berbeda dari From ({from_dom})", score=25)

    if parsed.return_path and parsed.from_address:
        from_dom  = parsed.from_address.split("@")[-1].lower()
        rpath_dom = parsed.return_path.split("@")[-1].lower()
        if from_dom != rpath_dom:
            signals.return_path_mismatch = True
            _note(signals, f"Return-Path ({rpath_dom}) berbeda dari From ({from_dom})", score=15)

    # ── Subject keywords ──────────────────────────────────────────
    if parsed.subject:
        subj_lower = parsed.subject.lower()
        found_kw   = [kw for kw in SUSPICIOUS_SUBJECT_KEYWORDS if kw in subj_lower]
        if found_kw:
            signals.has_suspicious_subject = True
            signals.subject_keywords        = found_kw
            _note(signals, f"Subject mengandung keyword phishing: {', '.join(found_kw)}", score=10)

    # ── Routing ───────────────────────────────────────────────────
    signals.originating_ip  = parsed.originating_ip
    signals.received_hop_count = len(parsed.received_chain)

    # Terlalu banyak hop (sering dipakai untuk obfuscation)
    if signals.received_hop_count > 8:
        _note(signals, f"Rantai routing panjang: {signals.received_hop_count} hop", score=10)

    # Floor score: impersonation brand TIDAK BOLEH lolos sebagai RENDAH
    if signals.impersonates_brand and signals.pre_score < 70:
        signals.pre_score = 70
    if signals.has_homoglyph and signals.pre_score < 70:
        signals.pre_score = 70

    signals.pre_score = min(signals.pre_score, 99)  # cap di 99, AI yang putuskan final
    return signals


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_tld(domain: str) -> str:
    """Ambil TLD dari domain: 'evil.co.uk' → '.uk', 'phish.xyz' → '.xyz'."""
    parts = domain.lower().rsplit(".", 1)
    return f".{parts[-1]}" if len(parts) > 1 else ""


def _normalize_digits(s: str) -> str:
    """
    Coba semua kemungkinan substitusi digit → huruf.
    '1' bisa berarti 'l' (paypa1→paypal) atau 'i' (m1crosoft→microsoft).
    Return tuple kedua kemungkinan, caller cek keduanya.
    Fungsi ini return versi '1→l' (paling umum di brand spoofing).
    """
    mapping = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "@": "a", "7": "t"}
    return "".join(mapping.get(c, c) for c in s)


def _normalize_digits_alt(s: str) -> str:
    """Alternatif: 1→i untuk kasus seperti m1crosoft→microsoft."""
    mapping = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "@": "a", "7": "t"}
    return "".join(mapping.get(c, c) for c in s)


def _has_homoglyph(domain: str) -> bool:
    """Cek apakah domain mengandung karakter non-ASCII yang mirip Latin."""
    for char in domain:
        if ord(char) > 127:  # non-ASCII
            # Cek apakah karakter ini punya nama Unicode yang mirip huruf Latin
            try:
                name = unicodedata.name(char, "").lower()
                if any(w in name for w in ["latin", "cyrillic", "greek"]):
                    return True
            except Exception:
                return True
    return False


def _note(signals: EmailSignals, message: str, score: int = 0) -> None:
    """Tambah catatan ke sinyal dan akumulasi skor."""
    signals.signal_notes.append(message)
    signals.pre_score += score