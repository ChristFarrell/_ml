# monitor/telegram_bot.py — Send email scan summaries to Telegram

import urllib.request
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Helpers ───────────────────────────────────────────────────────

def _esc(text: str) -> str:
    """Escape HTML special chars so Telegram parse_mode=HTML never chokes."""
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))

def _risk_icon(level: str) -> str:
    return {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "⛔"}.get(level, "⚪")


# ── Core send ─────────────────────────────────────────────────────

def send_message(text: str) -> bool:
    """Send a plain-text/HTML message to the configured Telegram chat."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        print("[Telegram] Not configured — skipping send.")
        return False

    url     = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = json.dumps({
        "chat_id"   : config.TELEGRAM_CHAT_ID,
        "text"      : text,
        "parse_mode": "HTML",
    }).encode()

    try:
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"[Telegram] Send error: {e} — {body}")
        return False
    except Exception as e:
        print(f"[Telegram] Send error: {e}")
        return False


# ── Summary builder ───────────────────────────────────────────────

def build_summary(results: list[dict], scan_number: int = 0) -> str:
    """
    Build Telegram HTML summary:
      • Overall stats + safety bar
      • Top threats (up to 5)
      • Description of first 10 emails
    All user-supplied strings are HTML-escaped.
    """
    total = len(results)
    if total == 0:
        return "📭 <b>Email Sentinel</b>\n\nNo emails scanned."

    counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0, "UNKNOWN": 0}
    for r in results:
        level = r.get("risk_level", "UNKNOWN")
        counts[level] = counts.get(level, 0) + 1

    safe_pct = round(counts["LOW"] / total * 100)
    filled   = round(safe_pct / 10)
    bar      = "🟩" * filled + "🟥" * (10 - filled)

    lines = [
        f"🛡 <b>Email Sentinel V3 — Scan #{scan_number}</b>",
        "",
        f"📬 Emails scanned: <b>{total}</b>",
        "",
        f"Safety: {bar}",
        f"🟢 Safe (LOW)      : {counts['LOW']} ({safe_pct}%)",
        f"🟡 Caution (MEDIUM): {counts['MEDIUM']}",
        f"🔴 Dangerous (HIGH): {counts['HIGH']}",
        f"⛔ Critical        : {counts['CRITICAL']}",
    ]

    # Top threats
    threats = [r for r in results if r.get("risk_level") in ("HIGH", "CRITICAL")]
    if threats:
        lines += ["", f"⚠️ <b>Top threats ({len(threats)} total):</b>"]
        for t in threats[:5]:
            icon  = "⛔" if t["risk_level"] == "CRITICAL" else "🔴"
            from_ = _esc(t.get("from", "?"))[:50]
            subj  = _esc(t.get("subject", "-"))[:50]
            score = t.get("risk_score", "?")
            lines.append(f"{icon} <code>{from_}</code>")
            lines.append(f"   {subj} [{score}/100]")

    # First 20 emails description
    lines += ["", f"📧 <b>First {min(20, total)} emails:</b>"]
    for i, r in enumerate(results[:20], 1):
        level  = r.get("risk_level", "UNKNOWN")
        from_  = _esc(r.get("from", "unknown"))[:50]
        subj   = _esc(r.get("subject", "(no subject)"))[:50]
        score  = r.get("risk_score", "?")
        rec    = _esc(r.get("recommendation", ""))
        sigs   = r.get("signals", [])
        sig    = _esc(sigs[0]) if sigs else "—"

        lines.append(
            f"\n<b>{i}.</b> {_risk_icon(level)} <b>[{level}]</b> "
            f"<code>{from_}</code>\n"
            f"   {subj}\n"
            f"   Score: {score}/100 | {sig}\n"
            f"   {rec}"
        )

    lines += ["", f"🕐 Next scan in {config.TELEGRAM_INTERVAL_HOURS}h"]
    return "\n".join(lines)


def send_summary(results: list[dict], scan_number: int = 0) -> bool:
    """Build and send the digest; split into chunks if over 4096 chars."""
    text = build_summary(results, scan_number)
    for chunk in _split_message(text, 4000):   # 4000 to be safe
        if not send_message(chunk):
            print("[Telegram] Failed to send chunk.")
            return False
    print(f"[Telegram] Summary sent ({len(results)} emails, scan #{scan_number}).")
    return True


def send_alert(result: dict) -> bool:
    """Immediate alert for a single HIGH/CRITICAL email."""
    level = result.get("risk_level", "")
    if level not in ("HIGH", "CRITICAL"):
        return False

    icon     = "⛔" if level == "CRITICAL" else "🔴"
    from_    = _esc(result.get("from", "?"))
    subject  = _esc(result.get("subject", "-"))
    rec      = _esc(result.get("recommendation", ""))
    sigs     = result.get("signals", [])[:5]
    sig_text = "\n".join(f"  • {_esc(s)}" for s in sigs) if sigs else "  —"

    text = (
        f"{icon} <b>Email Sentinel — Threat Detected</b>\n\n"
        f"<b>From:</b> <code>{from_}</code>\n"
        f"<b>Subject:</b> {subject}\n"
        f"<b>Risk score:</b> {result.get('risk_score','?')}/100\n"
        f"<b>Level:</b> {level}\n\n"
        f"<b>Signals:</b>\n{sig_text}\n\n"
        f"<b>Recommendation:</b>\n{rec}"
    )
    return send_message(text)


# ── Util ──────────────────────────────────────────────────────────

def _split_message(text: str, max_len: int) -> list[str]:
    """Split long message into chunks at newline boundaries."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks