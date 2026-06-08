# Email Sentinel V3 — Detection & Architecture Detail

This document covers everything added in V3 on top of V1/V2: Machine Learning, IMAP scanning, the local API server, the Firefox extension, Telegram alerts, and Native Messaging.

## 1. Machine Learning

V3 adds a feedback loop where you label emails as phishing or legit, and the system trains a classifier that gets progressively more accurate.

### Feature Extraction (`ml/features.py`)

Before any prediction or training, an email is converted to a fixed 15-element numeric vector:

```python
# ml/features.py

FEATURE_NAMES = [
    "is_suspicious_tld",       # 0 or 1
    "is_free_email",           # 0 or 1
    "has_digit_substitution",  # 0 or 1
    "has_homoglyph",           # 0 or 1
    "impersonates_brand",      # 0 or 1
    "reply_to_mismatch",       # 0 or 1
    "return_path_mismatch",    # 0 or 1
    "has_suspicious_subject",  # 0 or 1
    "subject_keyword_count",   # integer 0–5
    "hop_count_high",          # 0 or 1  (>8 hops)
    "pre_score_norm",          # float 0.0–1.0  (pre_score / 99)
    "spf_fail",                # 0 or 1  (0 if no V2 intel available)
    "dmarc_fail",              # 0 or 1  (0 if no V2 intel available)
    "no_mx_records",           # 0 or 1  (0 if no V2 intel available)
    "young_domain",            # 0 or 1  (0 if no V2 intel available)
]
```

The conversion function maps `EmailSignals` + optional `DomainIntelligence` to this vector:

```python
# ml/features.py

def signals_to_features(signals, intel=None) -> List[float]:
    kw_count = min(len(signals.subject_keywords), 5) if signals.subject_keywords else 0

    vec = [
        int(signals.is_suspicious_tld),
        int(signals.is_free_email),
        int(signals.has_digit_substitution),
        int(signals.has_homoglyph),
        int(bool(signals.impersonates_brand)),
        int(signals.reply_to_mismatch),
        int(signals.return_path_mismatch),
        int(signals.has_suspicious_subject),
        kw_count,
        int(signals.received_hop_count > 8),
        round(signals.pre_score / 99.0, 4),
        # V2 features — default to 0 if no network intel
        int(intel is not None and intel.spf_valid is False),
        int(intel is not None and intel.dmarc_valid is False),
        int(intel is not None and not intel.has_mx_records),
        int(intel is not None and intel.is_young_domain),
    ]
    return vec
```

Features 11–14 (SPF, DMARC, MX, domain age) are only non-zero when V2 investigation runs. Without V2 they default to 0 — the model still works, just with less information.

### Two Models in Ensemble (`ml/classifier.py`)

Two classifiers are trained simultaneously and their predictions averaged:

```python
# ml/classifier.py

# Random Forest — 100 trees, balanced class weight
rf = RandomForestClassifier(
    n_estimators     = 100,
    max_depth        = 6,
    min_samples_leaf = 2,
    class_weight     = "balanced",   # handles imbalanced phishing/legit ratio
    random_state     = 42,
)

# Logistic Regression — with StandardScaler
lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=500, random_state=42,
    )),
])
```

`class_weight="balanced"` to balance the phising and legit emails.

### Prediction and Confidence (`ml/classifier.py`)

Both models predict the probability that the email is phishing (class 1). The average becomes the `ml_score`:

```python
# ml/classifier.py

def predict(features: list[float]) -> Optional[dict]:
    X = np.array([features])

    probs = []
    if _rf_model is not None:
        p = _rf_model.predict_proba(X)[0][1]   # probability of class 1 (phishing)
        probs.append(p)
    if _lr_model is not None:
        p = _lr_model.predict_proba(X)[0][1]
        probs.append(p)

    avg_prob = sum(probs) / len(probs)
    ml_score = round(avg_prob * 100, 1)

    # Confidence = distance from the decision boundary (0.5)
    dist = abs(avg_prob - 0.5)
    confidence = "high" if dist > 0.3 else ("medium" if dist > 0.15 else "low")

    return {
        "ml_score"  : ml_score,
        "ml_prob"   : round(avg_prob, 4),
        "confidence": confidence,
        "model_used": "rf+lr",
        "trained_on": _train_meta.get("n_samples", 0),
    }
```

| `avg_prob` | `ml_score` | `dist` from 0.5 | Confidence |
|---|---|---|---|
| 0.92 | 92 | 0.42 | **high** (confidently phising)|
| 0.72 | 72 | 0.22 | **medium** |
| 0.58 | 58 | 0.08 | **low** (ignored) |
| 0.15 | 15 | 0.35 | **high** (confidently legit) |

Only `high` and `medium` confidence results influence the final score. `low` means the model is uncertain and the system falls back to rule-based + Ollama.

### Training Flow (`ml/classifier.py`)

Training requires at least 10 labeled samples (3 per class minimum):

```python
# ml/classifier.py

MIN_SAMPLES   = 10
MIN_PER_CLASS = 3

def train(labeled_data: list[dict]) -> dict:
    X = np.array([d["features"] for d in labeled_data])
    y = np.array([d["label"]    for d in labeled_data])   # 1=phishing, 0=legit

    # Cross-validation with StratifiedKFold (up to 5 folds)
    n_splits = min(5, n_phishing, n_legit)
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores_rf = cross_val_score(rf, X, y, cv=cv, scoring="f1").tolist()
        cv_scores_lr = cross_val_score(lr, X, y, cv=cv, scoring="f1").tolist()

    # Final training on all data
    rf.fit(X, y)
    lr.fit(X, y)

    # Save to disk so model survives agent restarts
    _save_model()   # → data/ml_model.pkl
```

Model is persisted with `pickle` and auto-loaded at startup:

```python
# ml/classifier.py

def load_model() -> bool:
    if not os.path.exists(MODEL_PATH):
        return False
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    _rf_model   = saved.get("rf")
    _lr_model   = saved.get("lr")
    _train_meta = saved.get("meta", {})
    _is_trained = _rf_model is not None or _lr_model is not None
    return True
```

### Score Blending (`core/analyzer.py`)

ML integrates into the scoring pipeline in `analyzer.py`:

```python
# core/analyzer.py

base_score = signals.pre_score + (intel.net_score if intel else 0)

# Try ML prediction
ml_result = None
try:
    from ml.features   import signals_to_features
    from ml.classifier import predict as ml_predict
    features  = signals_to_features(signals, intel)
    ml_result = ml_predict(features)
except Exception:
    pass  # ML not available — graceful fallback

# Blend based on ML confidence
if ml_result and ml_result["confidence"] in ("high", "medium"):
    ml_score = ml_result["ml_score"]
    if ai_score is not None:
        final_score = round(ml_score * 0.5 + base_score * 0.3 + ai_score * 0.2)
    else:
        final_score = round(ml_score * 0.6 + base_score * 0.4)
elif ai_score is not None:
    final_score = max(ai_score, base_score)
else:
    final_score = base_score
```

| Situation | Formula |
|---|---|
| ML confident + Ollama available | `ML×0.5 + rules×0.3 + AI×0.2` |
| ML confident, no Ollama | `ML×0.6 + rules×0.4` |
| ML not confident + Ollama available | `max(AI, rules)` |
| No ML, no Ollama | `rules only` |

ML result note is appended to `ai_analysis` so it appears in the UI and API response:

```python
# core/analyzer.py

if ml_result:
    ml_note = (
        f"\n\n[ML Model] Score: {ml_result['ml_score']}/100 "
        f"| Confidence: {ml_result['confidence']} "
        f"| Trained on: {ml_result['trained_on']} samples "
        f"| Model: {ml_result['model_used']}"
    )
    ai_text_full = ai_text + ml_note
```

## 2. IMAP Scanner (`monitor/imap_scanner.py`)

The scanner connects to gmailbox over IMAP SSL and runs the full detection pipeline on the latest emails.

### Connection and Fetch

```python
# monitor/imap_scanner.py

mail = imaplib.IMAP4_SSL(config.IMAP_HOST, config.IMAP_PORT)
mail.login(config.IMAP_USER, config.IMAP_PASSWORD)
mail.select(config.IMAP_FOLDER)

_, data = mail.search(None, "ALL")
all_ids = data[0].split()
# Take the last IMAP_SCAN_LIMIT (newest first after reverse)
ids_to_scan = list(reversed(all_ids[-config.IMAP_SCAN_LIMIT:]))
```

### Per-Message Analysis

```python
# monitor/imap_scanner.py

def _analyze_message(mail, msg_id: bytes, use_v2: bool) -> dict:
    _, msg_data = mail.fetch(msg_id, "(RFC822)")   # full RFC822 message
    raw_bytes   = msg_data[0][1]

    msg = email.message_from_bytes(raw_bytes, policy=policy.default)

    # Extract only the security-relevant headers
    header_str = _headers_to_str(msg)
    parsed     = parse_input(header_str)
    signals    = extract_signals(parsed)

    intel = None
    if use_v2 and signals.domain:
        from investigators.pipeline import investigate
        intel = investigate(signals.domain)   # WHOIS + MX + SPF/DMARC in parallel

    report = analyze(parsed, signals, intel)
    db.save_report(report)   # persist to sentinel.db

    return {
        "from"          : str(msg.get("From", "")),
        "subject"       : str(msg.get("Subject", ""))[:80],
        "domain"        : signals.domain,
        "risk_score"    : report.risk_score,
        "risk_level"    : report.risk_level.value,
        "signals"       : signals.signal_notes,
        "recommendation": report.recommendation,
        "created_at"    : report.created_at,
    }
```

## 3. Scheduler (`monitor/scheduler.py`)

The scheduler runs scans on a fixed interval and maintains shared state readable by the API server.

### Shared State

```python
# monitor/scheduler.py

state = {
    "last_results"     : [],       # results from last scan — served by /results
    "last_scan_time"   : None,
    "scan_count"       : 0,
    "running"          : False,
    "scan_in_progress" : False,    # true while scan is running
    "next_scan_time"   : None,
}
_lock = threading.Lock()           # protects state across threads
```

### Scan Loop

```python
# monitor/scheduler.py

def start(use_v2: bool = False):
    interval_sec = config.TELEGRAM_INTERVAL_HOURS * 3600

    while state["running"]:
        with _lock:
            state["scan_in_progress"] = True
        try:
            run_scan(use_v2=use_v2)
        finally:
            with _lock:
                state["scan_in_progress"] = False
        time.sleep(interval_sec)   # wait 6 hours (default)
```

### After Each Scan

```python
# monitor/scheduler.py

def run_scan(use_v2: bool = False) -> list[dict]:
    results = scan_inbox(use_v2=use_v2)

    with _lock:
        state["last_results"]   = results
        state["last_scan_time"] = datetime.datetime.now().isoformat()
        state["scan_count"]    += 1
        state["next_scan_time"] = (datetime.datetime.now() +
            datetime.timedelta(seconds=interval)).isoformat()

    # Immediate alert for HIGH/CRITICAL emails
    for r in results:
        if r.get("risk_level") in ("HIGH", "CRITICAL"):
            send_alert(r)

    # 6-hour digest
    send_summary(results, scan_number=state["scan_count"])

    return results
```

## 4. Telegram Alerts (`monitor/telegram_bot.py`)

### Immediate Alert

Sent right after each scan for any HIGH or CRITICAL email:

```python
# monitor/telegram_bot.py

def send_alert(result: dict) -> bool:
    icon    = "⛔" if level == "CRITICAL" else "🔴"
    sig_text = "\n".join(f"  • {_esc(s)}" for s in sigs[:5])

    text = (
        f"{icon} <b>Email Sentinel — Threat Detected</b>\n\n"
        f"<b>From:</b> <code>{from_}</code>\n"
        f"<b>Subject:</b> {subject}\n"
        f"<b>Risk score:</b> {result.get('risk_score','?')}/100\n"
        f"<b>Level:</b> {level}\n\n"
        f"<b>Signals:</b>\n{sig_text}\n\n"
        f"<b>Recommendation:</b>\n{rec}"
    )
    return send_message(text)
```

### 6-Hour Digest

Includes a safety bar, breakdown by level, top threats, and a description of the first 20 emails:

```python
# monitor/telegram_bot.py

def build_summary(results, scan_number=0) -> str:
    safe_pct = round(counts["LOW"] / total * 100)
    filled   = round(safe_pct / 10)
    bar      = "🟩" * filled + "🟥" * (10 - filled)

    lines = [
        f"🛡 <b>Email Sentinel V3 — Scan #{scan_number}</b>",
        f"Safety: {bar}",
        f"🟢 Safe (LOW)      : {counts['LOW']} ({safe_pct}%)",
        f"🟡 Caution (MEDIUM): {counts['MEDIUM']}",
        f"🔴 Dangerous (HIGH): {counts['HIGH']}",
        f"⛔ Critical        : {counts['CRITICAL']}",
    ]
    # ... top threats + first 20 email descriptions
```

## 5. Local API Server (`monitor/api_server.py`)

A lightweight HTTP server on `127.0.0.1:7842`. The Firefox extension calls it directly.

### Why no Flask?

Using stdlib `http.server` avoids adding a dependency just for the local API. The server handles CORS preflight so the extension (`moz-extension://`) can make cross-origin requests:

```python
# monitor/api_server.py

def do_OPTIONS(self):
    self.send_response(200)
    self._cors_headers()
    self.end_headers()

def _cors_headers(self):
    self.send_header("Access-Control-Allow-Origin", "*")
    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    self.send_header("Access-Control-Allow-Headers", "Content-Type")
```

### `/analyze` endpoint

The main endpoint used by both the extension's content script (badge injection) and popup:

```python
# monitor/api_server.py

def _analyze(self):
    body    = json.loads(self.rfile.read(length))
    email_  = body.get("email", "").strip()
    use_v2  = body.get("v2", False)

    parsed  = parse_input(email_)
    signals = extract_signals(parsed)

    intel = None
    if use_v2 and signals.domain:
        from investigators.pipeline import investigate
        intel = investigate(signals.domain)

    report = analyze(parsed, signals, intel)

    # Also return ml_features so extension can use them for labeling
    ml_features = signals_to_features(signals, intel)

    self._safe_json({
        "email"         : parsed.from_address,
        "risk_score"    : report.risk_score,
        "risk_level"    : report.risk_level.value,
        "signals"       : signals.signal_notes,
        "ai_analysis"   : report.ai_analysis,
        "recommendation": report.recommendation,
        "ml_features"   : ml_features,    # ← used when labeling via extension
    })
```

### `/ml/label` endpoint — auto-trains when enough data

```python
# monitor/api_server.py

def _ml_label(self):
    label_int = 1 if label_s == "phishing" else 0
    row_id    = save_label(email, domain, label_int, features)

    # Auto-train when >= 10 labeled samples exist
    labeled = get_labeled_data()
    train_result = None
    if len(labeled) >= 10:
        from ml.classifier import train
        train_result = train(labeled)

    self._safe_json({
        "ok"          : True,
        "total_labels": len(labeled),
        "auto_train"  : train_result,
    })
```

### `/scan-now` endpoint — non-blocking

Returns 202 immediately and runs the scan in a background thread:

```python
# monitor/api_server.py

def _scan_now(self):
    def _run():
        with sched._lock:
            sched.state["scan_in_progress"] = True
        try:
            sched.run_scan(use_v2=use_v2)
        finally:
            with sched._lock:
                sched.state["scan_in_progress"] = False

    threading.Thread(target=_run, daemon=True).start()
    self._safe_json({"ok": True, "message": "Scan started"}, status=202)
```

ML model is auto-loaded at API server startup:

```python
# monitor/api_server.py

try:
    from ml.classifier import load_model
    load_model()
except Exception as _e:
    print(f"[ML] Startup load skipped: {_e}")
```

## 6. Firefox Extension

### background.js — Service Worker

The background script runs persistently and is the single point of contact between the extension UI and the Python agent.

**Polling:** Every 15 minutes an alarm fires and refreshes the status cache:

```javascript
// extension/background.js

const REFRESH_MINUTES = 15;

chrome.runtime.onInstalled.addListener(() => {
  chrome.alarms.create(ALARM_NAME, { periodInMinutes: REFRESH_MINUTES });
  fetchAndCacheStatus();
});

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === ALARM_NAME) fetchAndCacheStatus();
});
```

**Tab listener:** Also refreshes on every Gmail/Outlook page load:

```javascript
// extension/background.js

chrome.tabs.onUpdated.addListener((tabId, info, tab) => {
  if (info.status === "complete" && isWebmail(tab.url)) fetchAndCacheStatus();
});

function isWebmail(url) {
  return url.includes("mail.google.com")  ||
         url.includes("outlook.live.com") ||
         url.includes("outlook.office.com");
}
```

**Badge:** Shows threat count in red or ✓ in green:

```javascript
// extension/background.js

function updateBadge(status) {
  if (status.scan_in_progress) {
    chrome.action.setBadgeText({ text: "…" });
    chrome.action.setBadgeBackgroundColor({ color: "#1565C0" });
    return;
  }
  const threats = status.threat_count || 0;
  if (threats > 0) {
    chrome.action.setBadgeText({ text: String(threats) });
    chrome.action.setBadgeBackgroundColor({ color: "#CC0000" });
  } else {
    chrome.action.setBadgeText({ text: "✓" });
    chrome.action.setBadgeBackgroundColor({ color: "#2E7D32" });
  }
}
```

**Email analysis (called by content.js):**

```javascript
// extension/background.js

async function analyzeEmail(email) {
  const res = await fetch(`${API}/analyze`, {
    method : "POST",
    headers: { "Content-Type": "application/json" },
    body   : JSON.stringify({ email, v2: true }),   // always V2
    signal : AbortSignal.timeout(30000),
  });
  return await res.json();
}
```

### content.js — Inbox Badge Injection

Injected into Gmail and Outlook pages. Uses `MutationObserver` to detect new email rows as the user scrolls:

```javascript
// extension/content.js

const PROCESSED = new WeakSet();   // prevent double-processing same row
const CACHE     = new Map();       // domain → result (avoid repeated API calls)

const observer = new MutationObserver(() => scanEmailRows());
observer.observe(document.body, { childList: true, subtree: true });
setTimeout(scanEmailRows, 2000);   // initial scan after page settles
```

For each row, the sender address is extracted using Gmail-specific CSS classes:

```javascript
// extension/content.js

function extractSender(row) {
  // Gmail: sender in .yP or .zF element
  const gmailSender = row.querySelector(".yP, .zF");
  if (gmailSender) {
    return gmailSender.getAttribute("email") || gmailSender.textContent.trim();
  }
  // Outlook: parse from aria-label
  const olEl = row.querySelector("[data-id], [aria-label]");
  if (olEl) {
    const label = olEl.getAttribute("aria-label") || "";
    const match = label.match(/[\w.+%-]+@[\w.-]+\.[a-z]{2,}/i);
    if (match) return match[0];
  }
  return null;
}
```

Badge is injected next to the sender name. Dangerous rows get a background color:

```javascript
// extension/content.js

function injectBadge(row, result) {
  const level = result.risk_level || "UNKNOWN";
  const cfg = {
    LOW     : { icon: "✅", color: "#2E7D32", bg: "#E8F5E9", label: "Safe"     },
    MEDIUM  : { icon: "⚠️", color: "#E65100", bg: "#FFF3E0", label: "Caution"  },
    HIGH    : { icon: "🚫", color: "#B71C1C", bg: "#FFEBEE", label: "Danger"   },
    CRITICAL: { icon: "⛔", color: "#7B0000", bg: "#FFCDD2", label: "CRITICAL" },
  }[level];

  const badge = document.createElement("span");
  badge.className = "sentinel-badge";
  badge.title     = `Email Sentinel: ${level} (${result.risk_score}/100)\n`
                  + (result.signals || []).join("\n");
  badge.textContent = `${cfg.icon} ${cfg.label}`;

  const senderEl = getSenderElement(row);
  if (senderEl) senderEl.appendChild(badge);

  if (level === "HIGH" || level === "CRITICAL") {
    row.style.backgroundColor = cfg.bg;
  }
}
```

## 7. Native Messaging (`native_host/email_sentinel_host.py`)

Native Messaging lets the extension start the Python agent without the user opening a terminal.

### Wire Protocol

Firefox communicates with the native host via stdin/stdout using a simple length-prefixed protocol:

```python
# native_host/email_sentinel_host.py

def read_message():
    raw_len = sys.stdin.buffer.read(4)           # 4-byte little-endian uint32
    msg_len = struct.unpack("<I", raw_len)[0]
    payload = sys.stdin.buffer.read(msg_len)
    return json.loads(payload.decode("utf-8"))

def send_message(obj):
    payload = json.dumps(obj).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("<I", len(payload)))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()
```

### Agent Lifecycle

```python
# native_host/email_sentinel_host.py

PID_FILE = "data/sentinel_agent.pid"
LOG_FILE = "data/sentinel_agent.log"

def spawn_agent():
    if is_agent_running():
        return {"ok": True, "status": "already_running"}

    log_fd = open(LOG_FILE, "a", buffering=1)   # line-buffered: logs appear in realtime

    _agent_proc = subprocess.Popen(
        [sys.executable, "-u", RUN_PY],   # -u = unbuffered Python output
        cwd=PROJECT_DIR,
        stdout=log_fd,
        stderr=log_fd,
        start_new_session=True,           # detach from Firefox process group
    )
    open(PID_FILE, "w").write(str(_agent_proc.pid))
    return {"ok": True, "status": "spawned", "pid": _agent_proc.pid, "log": LOG_FILE}

def stop_agent():
    pid = int(open(PID_FILE).read().strip())
    os.kill(pid, signal.SIGTERM)
    os.remove(PID_FILE)
    return {"ok": True, "status": "stopped"}

def is_agent_running():
    if os.path.exists(PID_FILE):
        pid = int(open(PID_FILE).read().strip())
        os.kill(pid, 0)   # signal 0 = check existence only, no actual signal
        return True
    return False
```

### Extension Side — `spawnAgentViaNative()`

After spawning, the extension polls the HTTP API every second until it responds (max 30s):

```javascript
// extension/background.js

async function spawnAgentViaNative() {
  const spawnReply = await nativeSend({ action: "SPAWN_AGENT" });
  if (!spawnReply.ok) return spawnReply;

  // Wait for HTTP API to come online (agent needs ~2s to start)
  const ready = await waitForApi(30, 1000);
  if (ready) {
    await fetchAndCacheStatus();
    return { ok: true, status: spawnReply.status, apiReady: true };
  }
  return { ok: true, status: spawnReply.status, apiReady: false };
}

async function waitForApi(maxSeconds, intervalMs) {
  const deadline = Date.now() + maxSeconds * 1000;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${API}/status`, { signal: AbortSignal.timeout(2000) });
      if (res.ok) return true;
    } catch (_) { /* not ready yet */ }
    await new Promise(r => setTimeout(r, intervalMs));
  }
  return false;
}
```

## 8. Full V3 Flow Summary

```
Extension popup opened
    → popup.js: GET_STATUS → background.js → GET /status
    → background.js: GET_RESULTS → GET /results
    → If status null (API offline) → show "Start Agent" button
    → If status ok → renderOnline() with safe rate, breakdown, threats

User clicks "Start Agent"
    → popup.js: SPAWN_AGENT → background.js
    → background.js: nativeSend({action:"SPAWN_AGENT"})
    → Firefox spawns email_sentinel_host.py via run_host.sh
    → host: subprocess.Popen(["python3", "monitor/run.py"])
    → host writes PID → data/sentinel_agent.pid
    → background.js polls GET /health every 1s (max 30s)
    → API responds → fetchAndCacheStatus() → popup re-renders

Agent starts (monitor/run.py)
    → load_model() from data/ml_model.pkl
    → start_background(api_server) → HTTP on 127.0.0.1:7842
    → start_background(scheduler)  → runs scan immediately

Scheduler scan cycle (every 6 hours)
    → imap_scanner.scan_inbox()
        → IMAP4_SSL login → SELECT INBOX → SEARCH ALL
        → take last 20 IDs → fetch each (RFC822)
        → for each email:
            parse_input() → extract_signals() → [investigate() if V2]
            → ml.predict() [if trained] → analyze() → db.save_report()
    → send_alert() for each HIGH/CRITICAL
    → send_summary() digest to Telegram
    → state["last_results"] updated (served to extension via /results)

User opens Gmail
    → content.js MutationObserver fires on each email row
    → extractSender() from .yP/.zF (Gmail) or aria-label (Outlook)
    → check CACHE by domain → if hit, injectBadge() immediately
    → if miss: ANALYZE_EMAIL → background.js → POST /analyze {email, v2:true}
    → API runs full pipeline → returns risk_level + signals + ml_features
    → CACHE.set(domain, result) → injectBadge()
        LOW      → ✅ Safe badge
        MEDIUM   → ⚠️ Caution badge
        HIGH     → 🚫 Danger badge + row highlight
        CRITICAL → ⛔ CRITICAL badge + row highlight
```
