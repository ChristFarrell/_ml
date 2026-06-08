# v1/core/analyzer.py — Combine local signals with Ollama AI, produce AnalysisReport

import json
import datetime
import urllib.request
import urllib.error
from ..models.schemas import ParsedEmail, EmailSignals, AnalysisReport, RiskLevel
from .. import config


def analyze(parsed: ParsedEmail, signals: EmailSignals) -> AnalysisReport:
    """
    Scoring pipeline:
      1. Start with signals.pre_score (local checks)
      2. Query Ollama — take MAX(ai_score, pre_score) so AI never lowers the score
      3. Apply hard floors for critical signals
    """
    ai_text, ai_score = _query_ollama(parsed, signals)

    if ai_score is not None:
        final_score = max(ai_score, signals.pre_score)
    else:
        final_score = signals.pre_score

    # Hard floors — critical signals can never result in LOW
    if signals.impersonates_brand or signals.has_homoglyph:
        final_score = max(final_score, 70)
    if signals.has_digit_substitution:
        final_score = max(final_score, 75)

    final_score = min(final_score, 100)
    risk_level  = _score_to_level(final_score)

    return AnalysisReport(
        email_input    = parsed.raw_input,
        parsed         = parsed,
        signals        = signals,
        risk_score     = final_score,
        risk_level     = risk_level,
        ai_analysis    = ai_text,
        recommendation = _recommendation(risk_level),
        created_at     = datetime.datetime.now().isoformat(),
    )


def _query_ollama(parsed, signals) -> tuple[str, int | None]:
    prompt  = _build_prompt(parsed, signals)
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
        with urllib.request.urlopen(req, timeout=config.OLLAMA_TIMEOUT) as resp:
            response = json.loads(resp.read()).get("response", "").strip()
        return _strip_score(response), _parse_score(response)

    except urllib.error.URLError:
        return "[Ollama unavailable — AI analysis skipped. Run 'ollama serve' to enable.]", None
    except Exception as e:
        return f"[Ollama error: {e}]", None


def _build_prompt(parsed, signals) -> str:
    signal_lines = "\n".join(f"  - {n}" for n in signals.signal_notes) \
                   or "  - No signals detected"

    return f"""You are an expert email security analyst.
Analyze the signals below and determine whether this email is dangerous.

EMAIL DATA:
  From        : {parsed.from_address or 'unknown'}
  Display name: {parsed.from_name or '-'}
  Domain      : {signals.domain or '-'}
  Reply-To    : {parsed.reply_to or '-'}
  Subject     : {parsed.subject or '-'}
  Origin IP   : {signals.originating_ip or 'unknown'}

RISK SIGNALS DETECTED (local score: {signals.pre_score}/100):
{signal_lines}

INSTRUCTIONS:
1. In 2-3 sentences, explain whether this email is suspicious and why.
2. Name the specific attack technique used (if any).
3. On the LAST line write ONLY: SCORE: [0-100]

Be concise and direct. Respond in English."""


def _parse_score(response: str) -> int | None:
    import re
    m = re.search(r"SCORE:\s*(\d{1,3})", response, re.IGNORECASE)
    return min(int(m.group(1)), 100) if m else None

def _strip_score(response: str) -> str:
    import re
    return re.sub(r"\n?SCORE:\s*\d{1,3}\s*$", "", response, flags=re.IGNORECASE).strip()

def _score_to_level(score: int) -> RiskLevel:
    if score >= 95:                 return RiskLevel.CRITICAL
    if score >= config.RISK_HIGH:   return RiskLevel.HIGH
    if score >= config.RISK_MEDIUM: return RiskLevel.MEDIUM
    return RiskLevel.LOW

def _recommendation(level: RiskLevel) -> str:
    return {
        RiskLevel.CRITICAL: "⛔ BLOCK IMMEDIATELY. Almost certainly phishing/malware. Do not open or click.",
        RiskLevel.HIGH    : "🚫 Do not click any links. Report as phishing to your email provider.",
        RiskLevel.MEDIUM  : "⚠️  Proceed with caution. Verify the sender through official channels first.",
        RiskLevel.LOW     : "✅ Appears safe, but stay alert with any links or attachments.",
    }.get(level, "ℹ️  Insufficient data. Proceed carefully.")
