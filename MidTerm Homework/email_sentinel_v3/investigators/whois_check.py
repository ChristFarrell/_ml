# investigators/whois_check.py — Check domain age and registrar info via WHOIS

import datetime
import whois
from models.schemas import DomainIntelligence
import config


def run(intel: DomainIntelligence) -> DomainIntelligence:
    """
    Query WHOIS for the domain.
    Fills: registrar, creation_date, domain_age_days, is_young_domain, registrant_country.
    """
    try:
        data = whois.whois(intel.domain)

        intel.registrar = _str(data.registrar)

        created = data.creation_date
        if isinstance(created, list):
            created = created[0]

        if created:
            if isinstance(created, str):
                for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%Y-%m-%dT%H:%M:%S"):
                    try:
                        created = datetime.datetime.strptime(created[:10], fmt[:8])
                        break
                    except ValueError:
                        pass

            if isinstance(created, datetime.datetime):
                intel.creation_date   = created.strftime("%Y-%m-%d")
                age = (datetime.datetime.now() - created).days
                intel.domain_age_days = age
                if age < config.YOUNG_DOMAIN_DAYS:
                    intel.is_young_domain = True
                    _note(intel,
                          f"Very new domain: registered {age} days ago ({intel.creation_date})",
                          score=25)

        country = data.country or (data.registrant_country if hasattr(data, "registrant_country") else None)
        intel.registrant_country = _str(country)

        if intel.domain_age_days and intel.domain_age_days > 0:
            age_str = (f"{intel.domain_age_days} days"
                       if intel.domain_age_days < 365
                       else f"{intel.domain_age_days // 365} years")
            _note(intel, f"Domain registered since: {intel.creation_date} ({age_str})", score=0)

    except Exception as e:
        intel.errors.append(f"WHOIS error: {type(e).__name__}: {e}")

    return intel


def _str(val) -> str | None:
    if val is None:
        return None
    if isinstance(val, list):
        val = val[0]
    return str(val).strip() or None


def _note(intel: DomainIntelligence, msg: str, score: int = 0):
    intel.net_notes.append(msg)
    intel.net_score += score