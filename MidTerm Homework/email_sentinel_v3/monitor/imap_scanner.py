# monitor/imap_scanner.py — Connect to mailbox via IMAP, scan latest emails

import imaplib
import email
from email import policy
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.parser   import parse_input
from core.signals  import extract_signals
from core.analyzer import analyze
from data import db
import config


def scan_inbox(use_v2: bool = False) -> list[dict]:
    """
    Connect to IMAP, fetch latest IMAP_SCAN_LIMIT emails, analyze each one.
    Returns list of result dicts for Telegram summary and API.
    """
    if not config.IMAP_HOST or not config.IMAP_USER or not config.IMAP_PASSWORD:
        raise ValueError(
            "IMAP not configured. Set IMAP_HOST, IMAP_USER, IMAP_PASSWORD in config.py"
        )

    results = []

    import datetime
    def _log(msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[Sentinel {ts}] {msg}", flush=True)

    _log(f"━━━ Scan started ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    _log(f"Connecting to {config.IMAP_HOST}:{config.IMAP_PORT} as {config.IMAP_USER}...")

    try:
        mail = imaplib.IMAP4_SSL(config.IMAP_HOST, config.IMAP_PORT)
        mail.login(config.IMAP_USER, config.IMAP_PASSWORD)
        _log(f"✓ IMAP login OK")

        mail.select(config.IMAP_FOLDER)
        _log(f"✓ Folder selected: {config.IMAP_FOLDER}")

        # Fetch latest N message IDs (newest first)
        _, data = mail.search(None, "ALL")
        all_ids = data[0].split()
        ids_to_scan = list(reversed(all_ids[-config.IMAP_SCAN_LIMIT:]))

        total = len(ids_to_scan)
        _log(f"✓ Found {len(all_ids)} emails total — scanning latest {total}")
        _log(f"  V2 network analysis: {'enabled' if use_v2 else 'disabled'}")
        _log(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        for i, msg_id in enumerate(ids_to_scan):
            try:
                _log(f"[{i+1:2d}/{total}] Fetching + analyzing...")
                result = _analyze_message(mail, msg_id, use_v2)
                results.append(result)

                level = result["risk_level"]
                score = result.get("risk_score", "?")
                icon  = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "⛔"}.get(level, "⚪")
                from_ = result.get("from", "?")[:55]
                subj  = result.get("subject", "?")[:45]
                sigs  = result.get("signals", [])
                sig   = f" | {sigs[0]}" if sigs else ""

                _log(f"[{i+1:2d}/{total}] {icon} {level:8s} score={score:>3}  {from_}")
                _log(f"         Subject : {subj}{sig}")

                if level in ("HIGH", "CRITICAL"):
                    _log(f"         ⚠️  THREAT — all signals: {', '.join(sigs[:4])}")

                # Progress bar (VSCode terminal friendly)
                done    = i + 1
                pct     = round(done / total * 100)
                filled  = round(done / total * 20)
                bar     = "█" * filled + "░" * (20 - filled)
                print(f"         [{bar}] {pct}%  ({done}/{total})", flush=True)

            except Exception as e:
                _log(f"[{i+1:2d}/{total}] ✗ Error: {e}")
                results.append({"error": str(e), "from": "unknown", "risk_level": "UNKNOWN"})

        mail.logout()
        _log(f"✓ IMAP connection closed")

    except imaplib.IMAP4.error as e:
        raise ConnectionError(f"IMAP login failed: {e}")

    # ── Final summary in terminal ─────────────────────────────────
    counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0, "UNKNOWN": 0}
    for r in results:
        counts[r.get("risk_level", "UNKNOWN")] = counts.get(r.get("risk_level", "UNKNOWN"), 0) + 1

    _log(f"━━━ Scan complete ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    _log(f"  Total   : {len(results)}")
    _log(f"  🟢 LOW     : {counts['LOW']}")
    _log(f"  🟡 MEDIUM  : {counts['MEDIUM']}")
    _log(f"  🔴 HIGH    : {counts['HIGH']}")
    _log(f"  ⛔ CRITICAL: {counts['CRITICAL']}")
    _log(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    return results


def _analyze_message(mail, msg_id: bytes, use_v2: bool) -> dict:
    """Fetch a single message and run the full analysis pipeline."""
    _, msg_data = mail.fetch(msg_id, "(RFC822)")
    raw_bytes   = msg_data[0][1]

    # Parse raw bytes → email object → extract headers as string
    msg     = email.message_from_bytes(raw_bytes, policy=policy.default)
    subject = str(msg.get("Subject", ""))
    from_   = str(msg.get("From", ""))

    # Reconstruct header string for our parser
    header_str = _headers_to_str(msg)

    parsed  = parse_input(header_str)
    signals = extract_signals(parsed)

    intel = None
    if use_v2 and signals.domain:
        from investigators.pipeline import investigate
        intel = investigate(signals.domain)

    report = analyze(parsed, signals, intel)

    # Save to DB
    try:
        db.save_report(report)
    except Exception:
        pass

    return {
        "from"           : from_,
        "subject"        : subject[:80],
        "domain"         : signals.domain,
        "risk_score"     : report.risk_score,
        "risk_level"     : report.risk_level.value,
        "signals"        : signals.signal_notes,
        "recommendation" : report.recommendation,
        "created_at"     : report.created_at,
    }


def _headers_to_str(msg) -> str:
    """Reconstruct a flat header string from email.message object."""
    lines = []
    for key in ("From", "Reply-To", "Return-Path", "Subject",
                "Received", "X-Originating-IP", "X-Forwarded-For"):
        values = msg.get_all(key) or []
        for v in values:
            lines.append(f"{key}: {v}")
    return "\n".join(lines)