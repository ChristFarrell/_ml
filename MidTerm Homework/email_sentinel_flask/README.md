# Email Sentinel — Flask Web App

Single Flask website combining V1 and V2 email analysis engines.

## Structure

```
flask_app/
├── app.py              ← Flask server (entry point)
├── requirements.txt
├── templates/
│   └── index.html      ← Web UI (design inspired by EMSentinel V3 extension)
├── v1/                 ← Email Sentinel V1 source (local signals + Ollama)
│   ├── core/
│   ├── models/
│   ├── reports/
│   └── config.py
└── v2/                 ← Email Sentinel V2 source (+ WHOIS/MX/SPF/DMARC)
    ├── core/
    ├── investigators/
    ├── models/
    ├── reports/
    └── config.py
```

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000

## Features

- **Version toggle** — choose V1 (fast, local only) or V2 (+ network investigation)
- **Flexible input** — paste an email address, `Name <email>` format, or full raw headers
- **Full result display** — risk score, local signals, network intel (V2), AI analysis, recommendation
- **Ollama AI** — make sure `ollama serve` is running with `llama3.2:1b` pulled

## Notes

- V2 makes DNS/WHOIS network calls and takes 5–15 seconds
- If Ollama is not running, AI analysis is skipped and local scoring is used
- Ctrl+Enter in the textarea submits the form
