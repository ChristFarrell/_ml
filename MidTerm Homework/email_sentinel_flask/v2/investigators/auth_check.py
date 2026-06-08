# investigators/auth_check.py — Check SPF, DKIM, DMARC via checkdmarc

import checkdmarc
from ..models.schemas import DomainIntelligence
from .. import config


def run(intel: DomainIntelligence) -> DomainIntelligence:
    """
    Run SPF + DKIM + DMARC checks.

    Scoring logic:
    - No DMARC                      → +20 (domain not protected against spoofing)
    - DMARC policy = none           → +10 (DMARC present but not enforced)
    - SPF invalid / missing         → +15
    - DMARC reject + SPF valid      → -10 (more trustworthy domain)
    """
    try:
        results = checkdmarc.check_domains(
            [intel.domain],
            timeout=config.DNS_TIMEOUT,
        )
        result = results[0] if isinstance(results, list) else results
        _process_spf(intel, result)
        _process_dmarc(intel, result)

    except checkdmarc.DMARCError as e:
        intel.dmarc_valid = False
        _note(intel, f"DMARC missing or invalid: {e}", score=20)
    except Exception as e:
        intel.errors.append(f"Auth check error: {type(e).__name__}: {e}")

    return intel


def _process_spf(intel: DomainIntelligence, result: dict):
    spf = result.get("spf", {})
    if not spf:
        intel.spf_valid = False
        _note(intel, "SPF record not found", score=15)
        return

    valid = spf.get("valid", False)
    intel.spf_valid  = valid
    intel.spf_record = spf.get("record", "")

    if not valid:
        reason = spf.get("error", "unknown")
        _note(intel, f"SPF invalid: {reason}", score=15)
    else:
        _note(intel, "SPF valid ✓", score=0)


def _process_dmarc(intel: DomainIntelligence, result: dict):
    dmarc = result.get("dmarc", {})
    if not dmarc:
        intel.dmarc_valid = False
        _note(intel, "DMARC not found — domain is unprotected against email spoofing", score=20)
        return

    valid  = dmarc.get("valid", False)
    policy = dmarc.get("tags", {}).get("p", {}).get("value", "none")
    intel.dmarc_valid  = valid
    intel.dmarc_policy = policy

    if not valid:
        _note(intel, "DMARC invalid", score=15)
    elif policy == "none":
        _note(intel, "DMARC present but policy=none — email spoofing is NOT blocked", score=10)
    elif policy == "quarantine":
        _note(intel, "DMARC policy=quarantine ✓", score=0)
    elif policy == "reject":
        _note(intel, "DMARC policy=reject ✓✓", score=-10)


def _note(intel: DomainIntelligence, msg: str, score: int = 0):
    intel.net_notes.append(msg)
    intel.net_score += score