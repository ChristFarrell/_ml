# investigators/mx_check.py — Check whether domain has valid MX records

import dns.resolver
import dns.exception
from models.schemas import DomainIntelligence
import config


def run(intel: DomainIntelligence) -> DomainIntelligence:
    """
    Check MX records for the domain.
    Phishing domains often lack MX records — they only send, never receive.
    """
    try:
        resolver          = dns.resolver.Resolver()
        resolver.lifetime = config.DNS_TIMEOUT

        answers = resolver.resolve(intel.domain, "MX")
        intel.has_mx_records = True
        intel.mx_records = sorted([str(r.exchange).rstrip(".") for r in answers])
        _note(intel, f"MX records found: {', '.join(intel.mx_records[:2])}", score=0)

    except dns.resolver.NXDOMAIN:
        intel.has_mx_records = False
        _note(intel, "Domain does not exist in DNS (NXDOMAIN) — likely a fake domain", score=30)

    except dns.resolver.NoAnswer:
        intel.has_mx_records = False
        _note(intel, "Domain has no MX records — cannot receive email (one-way spammer?)", score=15)

    except dns.exception.Timeout:
        intel.errors.append("MX check timed out")

    except Exception as e:
        intel.errors.append(f"MX error: {type(e).__name__}: {e}")

    return intel


def _note(intel: DomainIntelligence, msg: str, score: int = 0):
    intel.net_notes.append(msg)
    intel.net_score += score