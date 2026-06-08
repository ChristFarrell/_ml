# reports/formatter.py — Format AnalysisReport to plain text or JSON (V1+V2)

import json
from ..models.schemas import AnalysisReport, RiskLevel

RISK_EMOJI = {
    RiskLevel.LOW     : "🟢",
    RiskLevel.MEDIUM  : "🟡",
    RiskLevel.HIGH    : "🔴",
    RiskLevel.CRITICAL: "⛔",
    RiskLevel.UNKNOWN : "⚪",
}


def format_plain(report: AnalysisReport) -> str:
    sep   = "━" * 52
    r, s  = report, report.signals
    emoji = RISK_EMOJI.get(r.risk_level, "⚪")

    lines = [
        "", sep,
        "📧  EMAIL SENTINEL — ANALYSIS REPORT",
        sep,
        f"  Email    : {r.parsed.from_address or r.email_input}",
        f"  Domain   : {s.domain or '-'}",
        f"  Risk     : {emoji} {r.risk_level.value} ({r.risk_score}/100)",
        "",
    ]

    # Local signals
    if s.signal_notes:
        lines.append("LOCAL SIGNALS:")
        for note in s.signal_notes:
            lines.append(f"  ⚠  {note}")
        lines.append("")

    # Network intelligence (V2)
    if report.intel:
        intel = report.intel
        lines.append("NETWORK INVESTIGATION:")

        age_str = f"{intel.domain_age_days} days" if intel.domain_age_days else "unknown"
        lines.append(f"  🕐 Domain age   : {age_str}" +
                     (" ← VERY NEW" if intel.is_young_domain else ""))
        lines.append(f"  📮 MX record    : {'✓ present' if intel.has_mx_records else '✗ missing'}")
        lines.append(f"  🔐 SPF          : {'✓ valid' if intel.spf_valid else '✗ invalid/missing'}")

        if intel.dmarc_valid:
            lines.append(f"  🛡  DMARC        : ✓ valid (policy={intel.dmarc_policy})")
        else:
            lines.append(f"  🛡  DMARC        : ✗ missing / invalid")

        if intel.net_notes:
            lines.append("")
            for note in intel.net_notes:
                prefix = "  ✓" if ("✓" in note or "valid" in note.lower()) else "  ⚠"
                lines.append(f"{prefix}  {note}")

        if intel.errors:
            lines.append("")
            for err in intel.errors:
                lines.append(f"  ℹ️  {err}")
        lines.append("")

    # AI analysis
    if r.ai_analysis and not r.ai_analysis.startswith("["):
        lines.append("AI ANALYSIS:")
        for line in r.ai_analysis.splitlines():
            lines.append(f"  {line}")
        lines.append("")
    elif r.ai_analysis:
        lines.append(f"  {r.ai_analysis}")
        lines.append("")

    lines += [
        "RECOMMENDATION:",
        f"  {r.recommendation}",
        "",
        f"  Analyzed at: {r.created_at[:19].replace('T', ' ')}",
        sep, "",
    ]
    return "\n".join(lines)


def format_json(report: AnalysisReport) -> str:
    intel = report.intel
    return json.dumps({
        "email"       : report.parsed.from_address,
        "domain"      : report.signals.domain,
        "risk_score"  : report.risk_score,
        "risk_level"  : report.risk_level.value,
        "signals"     : report.signals.signal_notes,
        "network"     : {
            "domain_age_days" : intel.domain_age_days  if intel else None,
            "is_young_domain" : intel.is_young_domain  if intel else None,
            "has_mx"          : intel.has_mx_records   if intel else None,
            "spf_valid"       : intel.spf_valid        if intel else None,
            "dmarc_valid"     : intel.dmarc_valid      if intel else None,
            "dmarc_policy"    : intel.dmarc_policy     if intel else None,
            "notes"           : intel.net_notes        if intel else [],
        } if intel else None,
        "ai_analysis"    : report.ai_analysis,
        "recommendation" : report.recommendation,
        "created_at"     : report.created_at,
    }, ensure_ascii=False, indent=2)