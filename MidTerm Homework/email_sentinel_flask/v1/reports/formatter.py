# reports/formatter.py — Format AnalysisReport to plain text or JSON

import json
from ..models.schemas import AnalysisReport, RiskLevel


RISK_EMOJI = {
    RiskLevel.LOW      : "🟢",
    RiskLevel.MEDIUM   : "🟡",
    RiskLevel.HIGH     : "🔴",
    RiskLevel.CRITICAL : "⛔",
    RiskLevel.UNKNOWN  : "⚪",
}


def format_plain(report: AnalysisReport) -> str:
    """Format report as human-readable terminal text."""
    sep   = "━" * 50
    r     = report
    s     = r.signals
    emoji = RISK_EMOJI.get(r.risk_level, "⚪")

    lines = [
        "",
        sep,
        "📧  EMAIL SENTINEL — ANALYSIS REPORT V1",
        sep,
        f"  Email    : {r.parsed.from_address or r.email_input}",
        f"  Domain   : {s.domain or '-'}",
        f"  Risk     : {emoji} {r.risk_level.value} ({r.risk_score}/100)",
        "",
    ]

    if s.signal_notes:
        lines.append("SIGNALS DETECTED:")
        for note in s.signal_notes:
            lines.append(f"  ⚠  {note}")
        lines.append("")

    if r.ai_analysis and not r.ai_analysis.startswith("["):
        lines.append("AI ANALYSIS:")
        for line in r.ai_analysis.splitlines():
            lines.append(f"  {line}")
        lines.append("")
    elif r.ai_analysis.startswith("["):
        lines.append(f"  {r.ai_analysis}")
        lines.append("")

    lines.append("RECOMMENDATION:")
    lines.append(f"  {r.recommendation}")
    lines.append("")
    lines.append(f"  Analyzed at: {r.created_at[:19].replace('T', ' ')}")
    lines.append(sep)
    lines.append("")

    return "\n".join(lines)


def format_json(report: AnalysisReport) -> str:
    """Format report as JSON (for V2/V3 integration)."""
    return json.dumps({
        "email"          : report.parsed.from_address,
        "domain"         : report.signals.domain,
        "risk_score"     : report.risk_score,
        "risk_level"     : report.risk_level.value,
        "signals"        : report.signals.signal_notes,
        "ai_analysis"    : report.ai_analysis,
        "recommendation" : report.recommendation,
        "metadata": {
            "reply_to"       : report.parsed.reply_to,
            "originating_ip" : report.signals.originating_ip,
            "hop_count"      : report.signals.received_hop_count,
            "created_at"     : report.created_at,
        },
    }, ensure_ascii=False, indent=2)