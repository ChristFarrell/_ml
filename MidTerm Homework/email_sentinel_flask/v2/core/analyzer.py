# core/analyzer.py — Send signals to Ollama, receive analysis, determine risk level
# V2: also receives DomainIntelligence from investigators

import json
import datetime
import urllib.request
import urllib.error
from ..models.schemas import ParsedEmail, EmailSignals, AnalysisReport, RiskLevel
from typing import Optional
from .. import config


def analyze(parsed: ParsedEmail, signals: EmailSignals, intel=None) -> AnalysisReport:
    """
    Combine local signals + network intel (V2) with Ollama analysis.
    intel: DomainIntelligence | None — None means V1 mode (no network)
    """
    ai_text, ai_score = _query_ollama(parsed, signals, intel)

    # FIX: AI must not LOWER the local score — take the maximum
    local_score = signals.pre_score + (intel.net_score if intel else 0)
    if ai_score is None:
        final_score = local_score
    else:
        final_score = max(ai_score, local_score)

    # FIX: hard floor — brand impersonation can NEVER be LOW
    if signals.impersonates_brand or signals.has_homoglyph:
        final_score = max(final_score, 70)

    final_score = min(final_score, 100)
    risk_level  = _score_to_level(final_score)
    recommendation = _get_recommendation(risk_level, signals)

    return AnalysisReport(
        email_input    = parsed.raw_input,
        parsed         = parsed,
        signals        = signals,
        intel          = intel,
        risk_score     = final_score,
        risk_level     = risk_level,
        ai_analysis    = ai_text,
        recommendation = recommendation,
        created_at     = datetime.datetime.now().isoformat(),
    )


def _query_ollama(parsed, signals, intel) -> tuple[str, int | None]:
    prompt  = _build_prompt(parsed, signals, intel)
    payload = json.dumps({
        "model" : config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=config.OLLAMA_TIMEOUT) as resp:
            data     = json.loads(resp.read().decode("utf-8"))
            response = data.get("response", "").strip()

        score      = _parse_score(response)
        clean_text = _strip_score(response)
        return clean_text, score

    except urllib.error.URLError:
        return ("[Ollama unavailable — AI analysis skipped. Run 'ollama serve' to enable.]", None)
    except Exception as e:
        return (f"[Ollama error: {e}]", None)


def _build_prompt(parsed, signals, intel) -> str:
    local_lines = "\n".join(f"  - {n}" for n in signals.signal_notes) \
                  or "  - No local signals detected"

    net_section = ""
    if intel:
        net_lines = "\n".join(f"  - {n}" for n in intel.net_notes) \
                    or "  - No network findings"
        net_section = f"""

NETWORK INVESTIGATION (network score: {intel.net_score}/60):
  Registrar    : {intel.registrar or 'unknown'}
  Domain age   : {f"{intel.domain_age_days} days" if intel.domain_age_days else 'unknown'}
  MX record    : {'present' if intel.has_mx_records else 'MISSING'}
  SPF          : {'valid' if intel.spf_valid else 'INVALID/missing'}
  DMARC        : {'valid, policy=' + intel.dmarc_policy if intel.dmarc_valid else 'INVALID/missing'}
{net_lines}"""

    return f"""You are an experienced email security analyst.
Analyze the following data and determine whether this email is dangerous.

EMAIL DATA:
  From     : {parsed.from_address or 'unknown'}
  Domain   : {signals.domain or '-'}
  Reply-To : {parsed.reply_to or '-'}
  Subject  : {parsed.subject or '-'}
  Origin IP: {signals.originating_ip or 'unknown'}

LOCAL SIGNALS (local score: {signals.pre_score}/100):
{local_lines}{net_section}

INSTRUCTIONS:
1. In 2-3 sentences, explain whether this email is suspicious and why.
2. Name the specific phishing technique used (if any).
3. On the LAST line, write ONLY: SCORE: [0-100]

Respond in English. Be concise and direct."""


def _parse_score(response: str) -> int | None:
    import re
    m = re.search(r'SCORE:\s*(\d{1,3})', response, re.IGNORECASE)
    return min(int(m.group(1)), 100) if m else None


def _strip_score(response: str) -> str:
    import re
    return re.sub(r'\n?SCORE:\s*\d{1,3}\s*$', '', response, flags=re.IGNORECASE).strip()


def _score_to_level(score: int) -> RiskLevel:
    if score >= config.RISK_HIGH:
        return RiskLevel.CRITICAL if score >= 95 else RiskLevel.HIGH
    if score >= config.RISK_MEDIUM:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _get_recommendation(level: RiskLevel, signals: EmailSignals) -> str:
    if level == RiskLevel.CRITICAL:
        return "⛔ BLOCK IMMEDIATELY. Almost certainly phishing/malware. Do not open, do not click."
    if level == RiskLevel.HIGH:
        return "🚫 Do not click any links. Report as phishing to your email provider."
    if level == RiskLevel.MEDIUM:
        return "⚠️  Proceed with caution. Verify the sender through official channels before replying or clicking."
    return "✅ Appears safe, but stay cautious with any links or attachments."