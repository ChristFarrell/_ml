# tests/test_signals.py — Unit test untuk core/signals.py
#
# Jalankan: python -m pytest tests/ -v
# Atau tanpa pytest: python tests/test_signals.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .core.parser  import parse_input
from .core.signals import extract_signals


def test_digit_substitution():
    parsed  = parse_input("support@paypa1.com")
    signals = extract_signals(parsed)
    assert signals.has_digit_substitution, "Harus deteksi paypa1 → paypal"
    assert signals.impersonates_brand == "paypal"
    assert signals.pre_score > 0


def test_suspicious_tld():
    parsed  = parse_input("info@legit-store.xyz")
    signals = extract_signals(parsed)
    assert signals.is_suspicious_tld, "TLD .xyz harus dianggap mencurigakan"


def test_clean_email():
    parsed  = parse_input("hello@gmail.com")
    signals = extract_signals(parsed)
    assert not signals.has_digit_substitution
    assert not signals.is_suspicious_tld
    assert signals.is_free_email  # gmail.com adalah free email


def test_reply_to_mismatch():
    raw = "From: bank@bca.co.id\nReply-To: attacker@evil.ru\nSubject: Hi"
    parsed  = parse_input(raw)
    signals = extract_signals(parsed)
    assert signals.reply_to_mismatch, "Reply-To berbeda domain harus terdeteksi"


def test_suspicious_subject():
    raw = "From: x@test.com\nSubject: URGENT: Verify your account now"
    parsed  = parse_input(raw)
    signals = extract_signals(parsed)
    assert signals.has_suspicious_subject
    assert len(signals.subject_keywords) > 0


def test_brand_in_domain():
    parsed  = parse_input("noreply@microsoft-support.tk")
    signals = extract_signals(parsed)
    assert signals.impersonates_brand == "microsoft"
    assert signals.is_suspicious_tld   # .tk


def test_normal_corporate():
    parsed  = parse_input("john.doe@anthropic.com")
    signals = extract_signals(parsed)
    assert signals.pre_score < 30, "Email corporate normal tidak boleh skor tinggi"


if __name__ == "__main__":
    tests = [
        test_digit_substitution,
        test_suspicious_tld,
        test_clean_email,
        test_reply_to_mismatch,
        test_suspicious_subject,
        test_brand_in_domain,
        test_normal_corporate,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {t.__name__}: {e}")
        except Exception as e:
            print(f"  💥 {t.__name__}: {type(e).__name__}: {e}")

    print(f"\n{passed}/{len(tests)} test passed")
