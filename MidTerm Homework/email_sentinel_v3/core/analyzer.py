# core/analyzer.py — Combine local signals + network intel, query Ollama, produce report

import json
import datetime
import urllib.request
import urllib.error
from typing import Optional
from models.schemas import ParsedEmail, EmailSignals, DomainIntelligence, AnalysisReport, RiskLevel
import config


def analyze(parsed: ParsedEmail,
            signals: EmailSignals,
            intel: Optional[DomainIntelligence] = None) -> AnalysisReport:
    """
    Scoring pipeline:
      1. Start with signals.pre_score  (local rule-based checks)
      2. Add intel.net_score           (network checks, V2)
      3. ML model prediction           (V3 — if trained)
      4. Query Ollama                  (AI narrative + score)
      5. Blend semua sumber → final_score
      6. Apply hard floors untuk sinyal kritis
    """
    base_score = signals.pre_score + (intel.net_score if intel else 0)
    base_score = min(base_score, 99)

    # ── V3: ML prediction ────────────────────────────────────────
    ml_result = None
    try:
        from ml.features   import signals_to_features
        from ml.classifier import predict as ml_predict
        features  = signals_to_features(signals, intel)
        ml_result = ml_predict(features)
    except Exception:
        pass  # ML not available — graceful fallback

    # ── Ollama AI ─────────────────────────────────────────────────
    ai_text, ai_score = _query_ollama(parsed, signals, intel)

    # ── Blend scores ──────────────────────────────────────────────
    # Prioritas: ML (jika confident) > AI > rule-based
    if ml_result and ml_result["confidence"] in ("high", "medium"):
        ml_score = ml_result["ml_score"]
        # Ensemble: 50% ML + 30% rule-based + 20% AI (kalau ada)
        if ai_score is not None:
            final_score = round(ml_score * 0.5 + base_score * 0.3 + ai_score * 0.2)
        else:
            final_score = round(ml_score * 0.6 + base_score * 0.4)
    elif ai_score is not None:
        # AI + rule-based (tanpa ML)
        final_score = max(ai_score, base_score)
    else:
        final_score = base_score

    # Hard floors — sinyal kritis tidak boleh hasilkan LOW
    if signals.impersonates_brand or signals.has_homoglyph:
        final_score = max(final_score, 70)
    if signals.has_digit_substitution:
        final_score = max(final_score, 75)

    final_score = min(final_score, 100)
    risk_level  = _score_to_level(final_score)

    # Tambahkan info ML ke ai_analysis supaya terlihat di UI
    ai_text_full = ai_text
    if ml_result:
        ml_note = (
            f"\n\n[ML Model] Score: {ml_result['ml_score']}/100 "
            f"| Confidence: {ml_result['confidence']} "
            f"| Trained on: {ml_result['trained_on']} samples "
            f"| Model: {ml_result['model_used']}"
        )
        ai_text_full = ai_text + ml_note

    return AnalysisReport(
        email_input    = parsed.raw_input,
        parsed         = parsed,
        signals        = signals,
        intel          = intel,
        risk_score     = final_score,
        risk_level     = risk_level,
        ai_analysis    = ai_text_full,
        recommendation = _recommendation(risk_level),
        created_at     = datetime.datetime.now().isoformat(),
    )


def _query_ollama(parsed, signals, intel) -> tuple[str, Optional[int]]:
    prompt  = _build_prompt(parsed, signals, intel)
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


def _build_prompt(parsed, signals, intel) -> str:
    local_lines = "\n".join(f"  - {n}" for n in signals.signal_notes) \
                  or "  - No local signals detected"

    net_section = ""
    if intel:
        net_lines = "\n".join(f"  - {n}" for n in intel.net_notes) \
                    or "  - No network findings"
        net_section = f"""
NETWORK INVESTIGATION (net score: {intel.net_score}/60):
  Registrar   : {intel.registrar or 'unknown'}
  Domain age  : {f"{intel.domain_age_days} days" if intel.domain_age_days else 'unknown'}
  MX record   : {'present' if intel.has_mx_records else 'MISSING'}
  SPF         : {'valid' if intel.spf_valid else 'INVALID or missing'}
  DMARC       : {'valid, policy=' + (intel.dmarc_policy or '?') if intel.dmarc_valid else 'INVALID or missing'}
{net_lines}"""

    return f"""You are an expert email security analyst.
Analyze the following data and determine whether this email is dangerous.

EMAIL DATA:
  From        : {parsed.from_address or 'unknown'}
  Display name: {parsed.from_name or '-'}
  Domain      : {signals.domain or '-'}
  Reply-To    : {parsed.reply_to or '-'}
  Subject     : {parsed.subject or '-'}
  Origin IP   : {signals.originating_ip or 'unknown'}

LOCAL SIGNALS (local score: {signals.pre_score}/100):
{local_lines}{net_section}

INSTRUCTIONS:
1. In 2-3 sentences, explain whether this email is suspicious and why.
2. Name the specific attack technique used (if any).
3. On the LAST line write ONLY: SCORE: [0-100]

Be concise and direct. Respond in English."""


def _parse_score(response: str) -> Optional[int]:
    import re
    m = re.search(r"SCORE:\s*(\d{1,3})", response, re.IGNORECASE)
    return min(int(m.group(1)), 100) if m else None

def _strip_score(response: str) -> str:
    import re
    return re.sub(r"\n?SCORE:\s*\d{1,3}\s*$", "", response, flags=re.IGNORECASE).strip()

def _score_to_level(score: int) -> RiskLevel:
    if score >= 95:              return RiskLevel.CRITICAL
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
