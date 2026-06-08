# core/parser.py — Parse email address atau raw header menjadi ParsedEmail

import re
import email
from email import policy
from typing import Optional
from ..models.schemas import ParsedEmail


def parse_input(raw_input: str) -> ParsedEmail:
    """
    Smart parser: otomatis deteksi apakah input adalah
    - email address biasa   → "user@domain.com"
    - raw header string     → "From: ...\nTo: ...\n..."
    - file path             → di-handle di CLI sebelum masuk sini
    """
    raw_input = raw_input.strip()

    # Kalau mengandung newline atau header keywords → raw header
    if "\n" in raw_input or raw_input.lower().startswith(("from:", "received:", "mime")):
        return _parse_raw_header(raw_input)

    # Kalau terlihat seperti email address
    if "@" in raw_input:
        return _parse_address_only(raw_input)

    # Fallback: coba raw header
    return _parse_raw_header(raw_input)


def _parse_address_only(address: str) -> ParsedEmail:
    """Parse email address sederhana: 'Name <user@domain.com>' atau 'user@domain.com'."""
    name, addr = _extract_name_and_addr(address)
    domain = addr.split("@")[-1].lower() if "@" in addr else ""
    return ParsedEmail(
        raw_input    = address,
        from_address = addr,
        from_name    = name,
        from_domain  = domain,
    )


def _parse_raw_header(raw_header: str) -> ParsedEmail:
    """
    Parse raw email header string.
    Gunakan email.message_from_string dari stdlib Python.
    """
    # email.message_from_string butuh body, tambahkan newline kosong
    if "\n\n" not in raw_header:
        raw_header += "\n\n"

    msg = email.message_from_string(raw_header, policy=policy.default)

    # From
    from_raw   = msg.get("From", "")
    from_name, from_addr = _extract_name_and_addr(from_raw)
    from_domain = from_addr.split("@")[-1].lower() if "@" in from_addr else ""

    # Reply-To
    reply_to_raw  = msg.get("Reply-To", "")
    _, reply_addr = _extract_name_and_addr(reply_to_raw)

    # Return-Path
    return_path_raw  = msg.get("Return-Path", "")
    _, return_addr   = _extract_name_and_addr(return_path_raw)

    # Subject
    subject = msg.get("Subject", "")

    # Received chain — ambil semua header Received
    received_chain = msg.get_all("Received") or []

    # Originating IP dari header X-Originating-IP atau X-Forwarded-For
    orig_ip = (
        msg.get("X-Originating-IP")
        or msg.get("X-Forwarded-For")
        or _extract_ip_from_received(received_chain)
    )

    return ParsedEmail(
        raw_input      = raw_header,
        from_address   = from_addr   or None,
        from_name      = from_name   or None,
        from_domain    = from_domain or None,
        reply_to       = reply_addr  or None,
        return_path    = return_addr or None,
        subject        = subject     or None,
        originating_ip = orig_ip,
        received_chain = list(received_chain),
        extra_headers  = {k: v for k, v in msg.items()
                          if k.lower().startswith("x-")},
    )


def _extract_name_and_addr(raw: str) -> tuple[str, str]:
    """
    Ekstrak display name dan email address dari string seperti:
    'John Doe <john@example.com>'  →  ('John Doe', 'john@example.com')
    'john@example.com'             →  ('', 'john@example.com')
    """
    if not raw:
        return "", ""

    # Format: Name <addr>
    match = re.match(r'^"?([^"<]*)"?\s*<([^>]+)>', raw.strip())
    if match:
        return match.group(1).strip(), match.group(2).strip().lower()

    # Hanya address
    addr_match = re.search(r'[\w.+-]+@[\w.-]+\.\w+', raw)
    if addr_match:
        return "", addr_match.group(0).lower()

    return "", raw.strip().lower()


def _extract_ip_from_received(received_chain: list[str]) -> Optional[str]:
    """Coba ekstrak IP dari header Received paling akhir (originating)."""
    ip_pattern = re.compile(r'\[(\d{1,3}(?:\.\d{1,3}){3})\]')
    # Received chain diurutkan newest-first, kita mau yang paling lama (index terakhir)
    for received in reversed(received_chain):
        m = ip_pattern.search(received)
        if m:
            ip = m.group(1)
            # Skip IP lokal/private
            if not _is_private_ip(ip):
                return ip
    return None


def _is_private_ip(ip: str) -> bool:
    """Cek apakah IP termasuk range private/loopback."""
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        a, b = int(parts[0]), int(parts[1])
        return (
            a == 127
            or a == 10
            or (a == 172 and 16 <= b <= 31)
            or (a == 192 and b == 168)
        )
    except ValueError:
        return False
