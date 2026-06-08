"""
Email Sentinel — Flask Web Interface
Supports V1 (local signals + Ollama) and V2 (+ network investigation)
"""

from flask import Flask, render_template, request, jsonify

# V1 engine
from v1.core.parser   import parse_input    as v1_parse
from v1.core.signals  import extract_signals as v1_signals
from v1.core.analyzer import analyze         as v1_analyze

# V2 engine
from v2.core.parser            import parse_input    as v2_parse
from v2.core.signals           import extract_signals as v2_signals
from v2.core.analyzer          import analyze         as v2_analyze
from v2.investigators.pipeline import investigate     as v2_investigate

app = Flask(__name__)


# ── V1 ───────────────────────────────────────────────────────────

def run_v1(email_input: str) -> dict:
    parsed  = v1_parse(email_input)
    signals = v1_signals(parsed)
    report  = v1_analyze(parsed, signals)
    return _v1_to_dict(report)

def _v1_to_dict(report) -> dict:
    return {
        "version":        "v1",
        "email":          report.parsed.from_address or report.email_input,
        "from_name":      report.parsed.from_name,
        "domain":         report.signals.domain or "-",
        "reply_to":       report.parsed.reply_to,
        "originating_ip": report.signals.originating_ip,
        "hop_count":      report.signals.received_hop_count,
        "risk_score":     report.risk_score,
        "risk_level":     report.risk_level.value,
        "signal_notes":   report.signals.signal_notes,
        "ai_analysis":    report.ai_analysis,
        "recommendation": report.recommendation,
        "created_at":     report.created_at[:19].replace("T", " "),
        "intel":          None,
    }


# ── V2 ───────────────────────────────────────────────────────────

def run_v2(email_input: str) -> dict:
    parsed  = v2_parse(email_input)
    signals = v2_signals(parsed)
    intel   = v2_investigate(signals.domain) if signals.domain else None
    report  = v2_analyze(parsed, signals, intel)
    return _v2_to_dict(report)

def _v2_to_dict(report) -> dict:
    intel      = report.intel
    intel_data = None
    if intel:
        intel_data = {
            "registrar":       intel.registrar,
            "domain_age_days": intel.domain_age_days,
            "is_young_domain": intel.is_young_domain,
            "has_mx_records":  intel.has_mx_records,
            "spf_valid":       intel.spf_valid,
            "dmarc_valid":     intel.dmarc_valid,
            "dmarc_policy":    getattr(intel, "dmarc_policy", None),
            "net_score":       intel.net_score,
            "net_notes":       intel.net_notes,
            "errors":          intel.errors,
        }
    return {
        "version":        "v2",
        "email":          report.parsed.from_address or report.email_input,
        "from_name":      report.parsed.from_name,
        "domain":         report.signals.domain or "-",
        "reply_to":       report.parsed.reply_to,
        "originating_ip": report.signals.originating_ip,
        "hop_count":      report.signals.received_hop_count,
        "risk_score":     report.risk_score,
        "risk_level":     report.risk_level.value,
        "signal_notes":   report.signals.signal_notes,
        "ai_analysis":    report.ai_analysis,
        "recommendation": report.recommendation,
        "created_at":     report.created_at[:19].replace("T", " "),
        "intel":          intel_data,
    }


# ── Routes ───────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_email():
    data        = request.get_json()
    email_input = (data.get("email") or "").strip()
    version     = data.get("version", "v1")
    if not email_input:
        return jsonify({"error": "Email input is required."}), 400
    try:
        result = run_v2(email_input) if version == "v2" else run_v1(email_input)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5000
    print("=" * 52)
    print("  📧  Email Sentinel — Flask Web App")
    print("=" * 52)
    print(f"  Running at  : http://{host}:{port}")
    print(f"  Local alias : http://localhost:{port}")
    print(f"  Press Ctrl+C to stop")
    print("=" * 52)
    app.run(debug=True, host=host, port=port, use_reloader=False)
