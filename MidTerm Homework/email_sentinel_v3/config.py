import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (same folder as this file)
load_dotenv(Path(__file__).parent / ".env")

def _require(key: str) -> str:
    """Get a required env var — raises clear error if missing."""
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(
            f"Missing required config: '{key}'\n"
            f"Add it to your .env file. See .env.example for reference."
        )
    return val

def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default)

def _int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


# ── Ollama ───────────────────────────────────────────────────────
OLLAMA_BASE_URL = _get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = _get("OLLAMA_MODEL",    "llama3.2:1b")
OLLAMA_TIMEOUT  = 30

# ── Database ─────────────────────────────────────────────────────
DB_PATH = "data/sentinel.db"

# ── Risk thresholds ───────────────────────────────────────────────
RISK_LOW    = 30
RISK_MEDIUM = 60
RISK_HIGH   = 80

# ── Suspicious TLDs ───────────────────────────────────────────────
SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".click", ".pw", ".cc", ".tk",
    ".gq", ".ml", ".ga", ".cf", ".work", ".men",
}

# ── Free email providers ─────────────────────────────────────────
FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "protonmail.com", "tutanota.com", "guerrillamail.com",
    "mailinator.com", "tempmail.com", "10minutemail.com",
}

# ── Brand impersonation targets ───────────────────────────────────
IMPERSONATION_TARGETS = [
    "paypal", "amazon", "apple", "microsoft", "google",
    "netflix", "facebook", "instagram", "whatsapp",
    "bank", "bca", "mandiri", "bri", "bni", "dhl", "fedex",
]

# ── Legitimate domains (never flagged as spoofing) ────────────────
LEGIT_DOMAINS = {
    "google.com", "google.co.id", "gmail.com",
    "paypal.com", "paypal.co.id",
    "amazon.com", "amazon.co.id",
    "apple.com", "icloud.com",
    "microsoft.com", "outlook.com", "live.com", "hotmail.com",
    "netflix.com", "facebook.com", "fb.com",
    "instagram.com", "whatsapp.com",
    "bca.co.id", "klikbca.com", "bankmandiri.co.id", "mandiri.co.id",
    "bri.co.id", "bni.co.id", "dhl.com", "fedex.com",
    "github.com", "gitlab.com",
}

# ── V2: Network investigation ─────────────────────────────────────
DNS_TIMEOUT       = 5
WHOIS_TIMEOUT     = 10
YOUNG_DOMAIN_DAYS = 30

# ── V3: IMAP ─────────────────────────────────────────────────────
IMAP_HOST       = _get("IMAP_HOST",   "imap.gmail.com")
IMAP_PORT       = _int("IMAP_PORT",   993)
IMAP_USER       = _get("IMAP_USER",   "")
IMAP_PASSWORD   = _get("IMAP_PASSWORD", "")
IMAP_FOLDER     = "INBOX"
IMAP_SCAN_LIMIT = _int("IMAP_SCAN_LIMIT", 10)

# ── V3: Telegram ─────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN      = _get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID        = _get("TELEGRAM_CHAT_ID",   "")
TELEGRAM_INTERVAL_HOURS = _int("TELEGRAM_INTERVAL_HOURS", 1)

# ── V3: Local API (used by Firefox extension) ─────────────────────
API_HOST = _get("API_HOST", "127.0.0.1")
API_PORT = _int("API_PORT", 7842)
