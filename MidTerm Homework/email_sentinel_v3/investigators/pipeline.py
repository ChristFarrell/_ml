# investigators/pipeline.py — Orkestrator V2: jalankan semua investigator secara paralel

import concurrent.futures
from models.schemas import DomainIntelligence
from investigators import whois_check, mx_check, auth_check


def investigate(domain: str) -> DomainIntelligence:
    """
    Jalankan semua investigator secara paralel (ThreadPoolExecutor).
    Tiap investigator mengisi bagian berbeda dari DomainIntelligence.
    Return DomainIntelligence yang sudah terisi penuh.
    """
    intel = DomainIntelligence(domain=domain)

    # Jalankan 3 investigator paralel — total waktu = max(whois, mx, auth)
    # bukan whois + mx + auth (yang bisa 15+ detik kalau serial)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(whois_check.run, intel): "whois",
            pool.submit(mx_check.run,    intel): "mx",
            pool.submit(auth_check.run,  intel): "auth",
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                # Merge hasil balik ke intel utama
                _merge(intel, result)
            except Exception as e:
                intel.errors.append(f"{name} investigator crashed: {e}")

    intel.net_score = max(0, min(intel.net_score, 60))  # cap kontribusi network di 60
    return intel


def _merge(target: DomainIntelligence, source: DomainIntelligence):
    """
    Merge field non-default dari source ke target.
    Tiap investigator hanya mengisi field-nya sendiri,
    jadi merge aman tanpa collision.
    """
    for field_name, value in source.__dict__.items():
        if field_name in ("domain",):
            continue
        current = getattr(target, field_name)
        # Update kalau source punya nilai dan target masih default
        if value and not current:
            setattr(target, field_name, value)
        # Untuk list dan score, gabungkan
        elif isinstance(value, list) and isinstance(current, list):
            for item in value:
                if item not in current:
                    current.append(item)
        elif field_name == "net_score":
            target.net_score += source.net_score
