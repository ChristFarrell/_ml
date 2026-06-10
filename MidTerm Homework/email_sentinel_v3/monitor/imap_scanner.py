# monitor/imap_scanner.py — Connect to mailbox via IMAP, scan latest emails
# V3.1: adds Ollama body summarization per email (1-sentence, English, timeout-safe)

import imaplib
import email
import json
import urllib.request
import urllib.error
from email import policy
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.parser   import parse_input
from core.signals  import extract_signals
from core.analyzer import analyze
from data import db
import config


# ── Ollama summarizer ─────────────────────────────────────────────

def _summarize_body(subject: str, body_text: str) -> str:
    """
    Ask Ollama to summarize the email in one sentence.
    Timeout: 8 seconds — if Ollama is slow or unavailable, returns "" gracefully.
    Body is truncated to 600 chars to keep prompt short and response fast.
    """
    if not body_text.strip():
        return ""

    snippet = body_text.strip()[:600].replace("\n", " ")

    prompt = (
        f"Email subject: {subject}\n"
        f"Email body: {snippet}\n\n"
        "Summarize what this email is about in ONE sentence. "
        "What does the sender want or say? "
        "Reply in English only. No preamble."
    )

    payload = json.dumps({
        "model" : config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode()

    try:
        req = urllib.request.Request(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            text = data.get("response", "").strip()
            # Trim to one sentence max
            for sep in [".", "!", "?"]:
                idx = text.find(sep)
                if idx != -1 and idx > 10:
                    return text[:idx+1].strip()
            return text[:200].strip()
    except (urllib.error.URLError, TimeoutError):
        return ""
    except Exception:
        return ""


# ── Body extractor ────────────────────────────────────────────────

def _extract_body(msg) -> str:
    """Extract plain text body from email.message object."""
    body = ""
    try:
        if msg.is_multipart():
            for part in msg.walk():
                ct = part.get_content_type()
                cd = str(part.get("Content-Disposition", ""))
                if ct == "text/plain" and "attachment" not in cd:
                    charset = part.get_content_charset() or "utf-8"
                    body = part.get_payload(decode=True).decode(charset, errors="replace")
                    break
            # Fallback to HTML if no plain text
            if not body:
                for part in msg.walk():
                    if part.get_content_type() == "text/html":
                        charset = part.get_content_charset() or "utf-8"
                        html = part.get_payload(decode=True).decode(charset, errors="replace")
                        # Strip HTML tags crudely
                        import re
                        body = re.sub(r"<[^>]+>", " ", html)
                        body = re.sub(r"\s+", " ", body).strip()
                        break
        else:
            charset = msg.get_content_charset() or "utf-8"
            body = msg.get_payload(decode=True).decode(charset, errors="replace")
    except Exception:
        pass
    return body.strip()


# ── Main scan ─────────────────────────────────────────────────────

def scan_inbox(use_v2: bool = False) -> list[dict]:
    """
    Connect to IMAP, fetch latest IMAP_SCAN_LIMIT emails, analyze + summarize each one.
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

        _, data = mail.search(None, "ALL")
        all_ids = data[0].split()
        ids_to_scan = list(reversed(all_ids[-config.IMAP_SCAN_LIMIT:]))

        total = len(ids_to_scan)
        _log(f"✓ Found {len(all_ids)} emails total — scanning latest {total}")
        _log(f"  V2 network analysis : {'enabled' if use_v2 else 'disabled'}")
        _log(f"  Ollama summarization: enabled (8s timeout per email)")
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
                summ  = result.get("summary", "")

                _log(f"[{i+1:2d}/{total}] {icon} {level:8s} score={score:>3}  {from_}")
                _log(f"         Subject : {subj}")
                if summ:
                    _log(f"         Summary : {summ[:80]}")
                if sigs:
                    _log(f"         Signals : {sigs[0]}")
                if level in ("HIGH", "CRITICAL"):
                    _log(f"         ⚠️  THREAT — {', '.join(sigs[:3])}")

                done   = i + 1
                pct    = round(done / total * 100)
                filled = round(done / total * 20)
                bar    = "█" * filled + "░" * (20 - filled)
                print(f"         [{bar}] {pct}%  ({done}/{total})", flush=True)

            except Exception as e:
                _log(f"[{i+1:2d}/{total}] ✗ Error: {e}")
                results.append({"error": str(e), "from": "unknown",
                                 "risk_level": "UNKNOWN", "summary": ""})

        mail.logout()
        _log(f"✓ IMAP connection closed")

    except imaplib.IMAP4.error as e:
        raise ConnectionError(f"IMAP login failed: {e}")

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


# ── Per-message analysis ──────────────────────────────────────────

def _analyze_message(mail, msg_id: bytes, use_v2: bool) -> dict:
    """Fetch a single message, run detection pipeline, and summarize body."""
    _, msg_data = mail.fetch(msg_id, "(RFC822)")
    raw_bytes   = msg_data[0][1]

    msg     = email.message_from_bytes(raw_bytes, policy=policy.default)
    subject = str(msg.get("Subject", ""))
    from_   = str(msg.get("From", ""))

    # ── Security analysis (headers only) ─────────────────────────
    header_str = _headers_to_str(msg)
    parsed     = parse_input(header_str)
    signals    = extract_signals(parsed)

    intel = None
    if use_v2 and signals.domain:
        from investigators.pipeline import investigate
        intel = investigate(signals.domain)

    report = analyze(parsed, signals, intel)

    # ── Ollama body summary (separate, timeout-safe) ──────────────
    body    = _extract_body(msg)
    summary = _summarize_body(subject, body)

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
        "summary"        : summary,          # ← NEW: Ollama 1-sentence summary
        "created_at"     : report.created_at,
    }


def _headers_to_str(msg) -> str:
    lines = []
    for key in ("From", "Reply-To", "Return-Path", "Subject",
                "Received", "X-Originating-IP", "X-Forwarded-For"):
        values = msg.get_all(key) or []
        for v in values:
            lines.append(f"{key}: {v}")
    return "\n".join(lines)