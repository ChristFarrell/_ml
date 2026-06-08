# Email Sentinel V3 — The Monitor

Autonomous email security agent with IMAP scanning, Telegram alerts, and Firefox extension.

---

## Quick Setup

### 1. Install Python dependencies
```bash
pip install checkdmarc dnspython python-whois
```

### 2. Configure credentials
Edit `config.py` and fill in:
```python
# IMAP (use App Password, NOT your real password)
IMAP_HOST     = "imap.gmail.com"    # or imap.mail.yahoo.com, etc.
IMAP_USER     = "you@gmail.com"
IMAP_PASSWORD = "xxxx xxxx xxxx"    # Gmail App Password

# Telegram
TELEGRAM_BOT_TOKEN = "123456:ABC..."  # from @BotFather
TELEGRAM_CHAT_ID   = "987654321"      # your chat ID (message @userinfobot)
```

**Gmail App Password setup:**
1. myaccount.google.com → Security → 2-Step Verification → App passwords
2. Generate password for "Mail"
3. Paste the 16-char password into IMAP_PASSWORD

### 3. Start the agent
```bash
# Full agent (IMAP + Telegram + API for extension)
python monitor/run.py

# With V2 network investigation per email (slower, more thorough)
python monitor/run.py --v2

# One-time scan and exit
python monitor/run.py --scan-now

# API only (if you only want the Firefox extension)
python monitor/run.py --api-only
```

---

## Firefox Extension Setup

1. Open Firefox → `about:debugging`
2. Click **"This Firefox"** → **"Load Temporary Add-on"**
3. Navigate to the `extension/` folder and select `manifest.json`
4. The extension icon appears in your toolbar

**Make sure the local agent is running** (`python monitor/run.py --api-only` at minimum).
The extension talks to the local API at `http://127.0.0.1:7842`.

---

## Telegram Bot Setup

1. Message @BotFather on Telegram → `/newbot` → follow prompts
2. Copy the bot token into `config.py`
3. Message your new bot once (so it can send to you)
4. Message @userinfobot to get your chat ID

**What you'll receive every 6 hours:**
```
🛡 Email Sentinel — 6-Hour Summary

📬 Emails scanned: 50

Safety rating: 🟩🟩🟩🟩🟩🟩🟩🟩🟥🟥
✅ Safe (LOW)      : 40 emails (80%)
🟡 Caution (MEDIUM): 6 emails
🔴 Dangerous (HIGH): 3 emails
⛔ Critical        : 1 emails

⚠️ Top threats detected:
⛔ support@paypa1.com
   Subject: Urgent: Verify your account [92/100]
```

---

## CLI (V1/V2 still works)
```bash
python -m cli.analyze --email "support@paypa1.com"
python -m cli.analyze --email "x@evil.xyz" --v2
python -m cli.analyze --history paypa1.com
```

---

## Project Structure
```
email-sentinel-v3/
├── config.py                  ← All settings
├── core/
│   ├── parser.py              ← Parse email address / raw header
│   ├── signals.py             ← Local risk signals (no network)
│   └── analyzer.py            ← Scoring pipeline + Ollama
├── investigators/             ← V2 network checks
│   ├── whois_check.py
│   ├── mx_check.py
│   ├── auth_check.py          ← SPF/DKIM/DMARC
│   └── pipeline.py            ← Runs all 3 in parallel
├── monitor/                   ← V3 agent
│   ├── run.py                 ← Main entry point ← START HERE
│   ├── imap_scanner.py        ← Scan inbox, analyze 50 emails
│   ├── scheduler.py           ← 6-hour loop
│   ├── telegram_bot.py        ← Send summaries + alerts
│   └── api_server.py          ← Local REST API for extension
├── extension/                 ← Firefox extension
│   ├── manifest.json
│   ├── background.js          ← Service worker
│   ├── content.js             ← Inject badges into Gmail/Outlook
│   ├── popup.html/js          ← Toolbar popup UI
│   └── overlay.css
├── models/schemas.py
├── reports/formatter.py
├── data/db.py
└── tests/test_signals.py
```
