# monitor/api_server.py — Local REST API consumed by the Firefox extension
#
# Endpoints:
#   GET  /status          → scheduler state + last scan summary
#   GET  /results         → last 50 scan results (JSON)
#   POST /analyze         → one-off analysis {"email": "x@y.com", "v2": true/false}
#   POST /scan-now        → trigger immediate full IMAP scan in background
#   GET  /health          → simple alive check

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.parser   import parse_input
from core.signals  import extract_signals
from core.analyzer import analyze
from monitor       import scheduler as sched

# ── Load ML model on startup (non-fatal if not available) ─────────
try:
    from ml.classifier import load_model
    load_model()
except Exception as _e:
    print(f"[ML] Startup load skipped: {_e}", flush=True)


class _Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # silence default HTTP log spam

    # ── Routing ──────────────────────────────────────────────────

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/health":
            self._json({"ok": True, "version": "v3"})
        elif path == "/status":
            self._status()
        elif path == "/results":
            self._results()
        elif path == "/ml/status":
            self._ml_status()
        else:
            self._error(404, "Not found")

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/analyze":
            self._analyze()
        elif path == "/scan-now":
            self._scan_now()
        elif path == "/ml/label":
            self._ml_label()
        elif path == "/ml/train":
            self._ml_train()
        else:
            self._error(404, "Not found")

    def do_OPTIONS(self):
        # CORS preflight for Firefox extension
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    # ── Handlers ─────────────────────────────────────────────────

    def _status(self):
        with sched._lock:
            st = dict(sched.state)
        st.pop("last_results", None)

        counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        with sched._lock:
            for r in sched.state.get("last_results", []):
                level = r.get("risk_level", "UNKNOWN")
                if level in counts:
                    counts[level] += 1

        total      = sum(counts.values())
        safe_pct   = round(counts["LOW"] / total * 100) if total else 0
        threat_cnt = counts["HIGH"] + counts["CRITICAL"]

        self._safe_json({
            "running"        : st.get("running"),
            "last_scan_time" : st.get("last_scan_time"),
            "next_scan_time" : st.get("next_scan_time"),
            "scan_count"     : st.get("scan_count"),
            "scan_in_progress": st.get("scan_in_progress", False),
            "total_scanned"  : total,
            "safe_percent"   : safe_pct,
            "threat_count"   : threat_cnt,
            "breakdown"      : counts,
        })

    def _results(self):
        with sched._lock:
            results = list(sched.state.get("last_results", []))
        self._safe_json({"results": results, "total": len(results)})

    def _analyze(self):
        try:
            length  = int(self.headers.get("Content-Length", 0))
            body    = json.loads(self.rfile.read(length))
            email_  = body.get("email", "").strip()
            use_v2  = body.get("v2", False)

            if not email_:
                self._error(400, "Missing 'email' field")
                return

            parsed  = parse_input(email_)
            signals = extract_signals(parsed)

            intel = None
            if use_v2 and signals.domain:
                from investigators.pipeline import investigate
                intel = investigate(signals.domain)

            report = analyze(parsed, signals, intel)

            # Compute features untuk dikirim ke extension (dipakai saat user label)
            ml_features = []
            try:
                from ml.features import signals_to_features
                ml_features = signals_to_features(signals, intel)
            except Exception:
                pass

            self._safe_json({
                "email"          : parsed.from_address,
                "domain"         : signals.domain,
                "risk_score"     : report.risk_score,
                "risk_level"     : report.risk_level.value,
                "signals"        : signals.signal_notes,
                "net_notes"      : intel.net_notes if intel else [],
                "ai_analysis"    : report.ai_analysis,
                "recommendation" : report.recommendation,
                "ml_features"    : ml_features,   # ← untuk labeling
            })
        except Exception as e:
            self._error(500, str(e))

    def _scan_now(self):
        """Trigger an immediate IMAP scan in a background thread."""
        try:
            length  = int(self.headers.get("Content-Length", 0) or 0)
            body    = {}
            if length:
                body = json.loads(self.rfile.read(length))
            use_v2  = body.get("v2", False)

            with sched._lock:
                already = sched.state.get("scan_in_progress", False)

            if already:
                self._safe_json({"ok": False, "message": "Scan already in progress"})
                return

            # Run in background so we can return 202 immediately
            def _run():
                with sched._lock:
                    sched.state["scan_in_progress"] = True
                try:
                    sched.run_scan(use_v2=use_v2)
                finally:
                    with sched._lock:
                        sched.state["scan_in_progress"] = False

            threading.Thread(target=_run, daemon=True, name="OnDemandScan").start()
            self._safe_json({"ok": True, "message": "Scan started"}, status=202)

        except Exception as e:
            self._error(500, str(e))

    # ── Helpers ──────────────────────────────────────────────────

    def _ml_status(self):
        """GET /ml/status — model info + label stats."""
        try:
            from ml.classifier import get_status
            from data.db       import get_label_stats
            status = get_status()
            stats  = get_label_stats()
            self._safe_json({**status, "label_stats": stats})
        except Exception as e:
            self._error(500, str(e))

    def _ml_label(self):
        """
        POST /ml/label
        Body: {"email": "x@y.com", "domain": "y.com", "label": "phishing"/"legit",
               "features": [...15 floats...]}
        Simpan label, lalu auto-train jika data sudah cukup.
        """
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length))

            email   = body.get("email", "")
            domain  = body.get("domain", "")
            label_s = body.get("label", "")
            features= body.get("features", [])

            if label_s not in ("phishing", "legit"):
                self._error(400, "label must be 'phishing' or 'legit'")
                return
            if len(features) != 15:
                self._error(400, f"features must have 15 elements, got {len(features)}")
                return

            label_int = 1 if label_s == "phishing" else 0

            from data.db import save_label, get_labeled_data
            row_id = save_label(email, domain, label_int, features)

            # Auto-train jika data cukup
            labeled = get_labeled_data()
            train_result = None
            if len(labeled) >= 10:
                from ml.classifier import train
                train_result = train(labeled)

            self._safe_json({
                "ok"          : True,
                "label_id"    : row_id,
                "total_labels": len(labeled),
                "auto_train"  : train_result,
            })
        except Exception as e:
            self._error(500, str(e))

    def _ml_train(self):
        """
        POST /ml/train — paksa retrain manual.
        Juga bisa bootstrap dari historical data dulu.
        Body (opsional): {"bootstrap": true}
        """
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
            body   = json.loads(self.rfile.read(length)) if length else {}
            do_bootstrap = body.get("bootstrap", False)

            bootstrapped = 0
            if do_bootstrap:
                from data.db import bootstrap_labels_from_history
                bootstrapped = bootstrap_labels_from_history()
                print(f"[ML] Bootstrapped {bootstrapped} labels from history", flush=True)

            from data.db       import get_labeled_data
            from ml.classifier import train
            labeled      = get_labeled_data()
            train_result = train(labeled)

            self._safe_json({
                "ok"         : True,
                "bootstrapped": bootstrapped,
                "result"     : train_result,
            })
        except Exception as e:
            self._error(500, str(e))

    def _safe_json(self, data: dict, status: int = 200):
        """Send JSON response; silently swallow BrokenPipeError on client disconnect."""
        try:
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self._cors_headers()
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Client disconnected before we could respond — not an error

    # Keep _json as alias for backward-compat
    def _json(self, data: dict, status: int = 200):
        self._safe_json(data, status)

    def _error(self, code: int, message: str):
        try:
            self._safe_json({"error": message}, status=code)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


def start(host: str = None, port: int = None):
    """Start the API server (blocking). Call start_background() for non-blocking."""
    host = host or config.API_HOST
    port = port or config.API_PORT
    server = HTTPServer((host, port), _Handler)
    print(f"[API] Listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def start_background(host: str = None, port: int = None) -> threading.Thread:
    """Start API server in a daemon thread."""
    t = threading.Thread(
        target=start, args=(host, port), daemon=True, name="SentinelAPI"
    )
    t.start()
    return t
