// popup.js — Email Sentinel V3

const RISK_COLORS = {
  LOW: "#2E7D32", MEDIUM: "#E65100", HIGH: "#C62828", CRITICAL: "#7B0000",
};
const RISK_ICONS = {
  LOW: "✅", MEDIUM: "⚠️", HIGH: "🚫", CRITICAL: "⛔",
};

// ── State ─────────────────────────────────────────────────────────
let _spawnStartTime  = null;
let _scanPollTimer   = null;  // single active polling timer — prevents duplicates

document.addEventListener("DOMContentLoaded", () => loadAndRender());

// ── Main render dispatcher ────────────────────────────────────────
async function loadAndRender() {
  let [status, results] = await Promise.all([
    sendMsg("GET_STATUS"),
    sendMsg("GET_RESULTS"),
  ]);

  // Retry once — API may need a moment to respond on first open
  if (!status) {
    await new Promise(r => setTimeout(r, 1500));
    status  = await sendMsg("GET_STATUS");
    results = await sendMsg("GET_RESULTS");
  }

  const main = document.getElementById("mainContent");

  if (!status) {
    stopScanPoll();
    renderOffline(main);
    return;
  }

  renderOnline(main, status, results || []);

  // If scan is in progress, start polling (idempotent)
  if (status.scan_in_progress) {
    startScanPoll();
  } else {
    stopScanPoll();
  }
}

// ── Scan polling — single timer, no duplicates ────────────────────
function startScanPoll() {
  if (_scanPollTimer) return; // already polling
  _scanPollTimer = setInterval(async () => {
    const status = await sendMsg("GET_STATUS");
    if (!status || !status.scan_in_progress) {
      stopScanPoll();
      await loadAndRender();
    } else {
      // Update header text only — no full re-render to avoid glitch
      const sub = document.getElementById("headerSub");
      if (sub) sub.textContent = "🔍 Scan in progress...";
    }
  }, 1500);
}

function stopScanPoll() {
  if (_scanPollTimer) {
    clearInterval(_scanPollTimer);
    _scanPollTimer = null;
  }
}

// ── Offline screen ────────────────────────────────────────────────
function renderOffline(container) {
  document.getElementById("statusDot").className = "status-dot dot-red";
  document.getElementById("headerSub").textContent = "Agent offline";

  container.innerHTML = `
    <div class="section offline-msg">
      <div class="icon">🔌</div>
      <div style="margin-bottom:8px"><strong>Agent not running</strong></div>
      <div style="margin-bottom:12px;color:#555;font-size:12px">
        Klik tombol di bawah untuk start agent Python secara otomatis.
      </div>
    </div>
    <div class="section" style="display:flex;flex-direction:column;gap:6px">
      <button class="btn btn-primary" id="startBtn">🚀 Start Agent</button>
      <button class="btn btn-secondary" id="retryBtn">🔄 Retry Connection</button>
    </div>
  `;

  document.getElementById("retryBtn").onclick = () => loadAndRender();
  document.getElementById("startBtn").onclick  = () => startAgentWithLoading();
}

// ── Spawning / loading screen ─────────────────────────────────────
function renderSpawning(container, phase) {
  if (!_spawnStartTime) _spawnStartTime = Date.now();
  const elapsed = Math.round((Date.now() - _spawnStartTime) / 1000);

  const phases = [
    { icon: "⚙️",  text: "Initializing agent..."     },
    { icon: "🐍",  text: "Starting Python process..."  },
    { icon: "🔌",  text: "Connecting to local API..."  },
    { icon: "📬",  text: "Scanning inbox..."            },
  ];
  const p = phases[Math.min(phase, phases.length - 1)];

  document.getElementById("statusDot").className = "status-dot dot-blue dot-pulse";
  document.getElementById("headerSub").textContent = "Starting agent...";

  container.innerHTML = `
    <div class="section spawning-screen">
      <div class="spawn-icon">${p.icon}</div>
      <div class="spawn-label">${p.text}</div>
      <div class="spawn-bar-wrap">
        <div class="spawn-bar-fill" style="width:${Math.min(100, (phase + 1) * 25)}%"></div>
      </div>
      <div style="font-size:11px;color:#9fa8da;margin-bottom:10px">
        ${elapsed}s — scan 10 email need ~30 detik
      </div>
      <div class="spawn-dots">
        <span class="dot-anim"></span>
        <span class="dot-anim" style="animation-delay:.2s"></span>
        <span class="dot-anim" style="animation-delay:.4s"></span>
      </div>
    </div>
  `;
}

async function startAgentWithLoading() {
  const main = document.getElementById("mainContent");

  // If agent is already running (started from terminal), just show status
  const quickCheck = await sendMsg("GET_STATUS");
  if (quickCheck) {
    await loadAndRender();
    return;
  }

  _spawnStartTime = Date.now();
  renderSpawning(main, 0);
  setTimeout(() => renderSpawning(main, 1), 600);

  const reply = await sendMsg("SPAWN_AGENT");

  if (!reply || !reply.ok) {
    const errMsg = reply?.error || "Native Messaging host not installed.";
    const isNotInstalled = !reply || errMsg.includes("not found") ||
                           errMsg.includes("disconnected") || errMsg.includes("host");
    document.getElementById("statusDot").className = "status-dot dot-red";
    document.getElementById("headerSub").textContent = "Agent offline";
    main.innerHTML = `
      <div class="section offline-msg">
        <div class="icon">⛔</div>
        <div style="margin-bottom:8px;color:#c62828"><strong>Gagal start agent</strong></div>
        ${isNotInstalled ? `
        <div style="font-size:12px;color:#555;text-align:left;line-height:1.7">
          Native host belum diinstall.<br>
          Jalankan <strong>sekali saja</strong> di terminal:<br><br>
          <code style="background:#f5f5f5;padding:4px 8px;border-radius:4px;display:block;margin:4px 0">
            bash native_host/install.sh
          </code>
          Lalu reload extension, klik Start Agent lagi.
        </div>
        ` : `<div style="font-size:11px;color:#555;margin-bottom:10px">${escHtml(errMsg)}</div>`}
      </div>
      <div class="section">
        <button class="btn btn-secondary" id="retryBtn">🔄 Retry</button>
      </div>
    `;
    document.getElementById("retryBtn").onclick = () => loadAndRender();
    return;
  }

  // Log path ke browser console (terlihat di about:debugging → Inspect)
  if (reply.log) {
    console.log(`[Sentinel] Agent log: ${reply.log}`);
  }

  // Process spawned — poll until HTTP API ready
  renderSpawning(main, 2);

  const deadline = Date.now() + 60_000;
  let   phase    = 2;
  let   tick     = 0;

  const waitTimer = setInterval(async () => {
    tick++;
    if (tick % 5 === 0) phase = Math.min(3, phase + 1);
    renderSpawning(main, phase);

    if (Date.now() > deadline) {
      clearInterval(waitTimer);
      await loadAndRender();
      return;
    }

    const status = await sendMsg("GET_STATUS");
    if (status) {
      clearInterval(waitTimer);
      await loadAndRender();
    }
  }, 1000);
}

// ── Online screen ─────────────────────────────────────────────────
function renderOnline(container, status, results) {
  const scanning = status.scan_in_progress || false;

  if (scanning) {
    document.getElementById("statusDot").className = "status-dot dot-blue";
    document.getElementById("headerSub").textContent = "🔍 Scan in progress...";
  } else {
    document.getElementById("statusDot").className = "status-dot dot-green";
    document.getElementById("headerSub").textContent =
      `Agent active · Scan #${status.scan_count || 0}`;
  }

  const total    = status.total_scanned || 0;
  const safePct  = status.safe_percent  || 0;
  const threats  = status.threat_count  || 0;
  const bd       = status.breakdown     || {};
  const barColor = threats > 0 ? "#EF5350" : "#66BB6A";

  const lastScan = status.last_scan_time
    ? new Date(status.last_scan_time).toLocaleTimeString() : "—";
  const nextScan = status.next_scan_time
    ? new Date(status.next_scan_time).toLocaleTimeString() : "—";

  const dangerItems = (results || [])
    .filter(r => r.risk_level === "HIGH" || r.risk_level === "CRITICAL")
    .slice(0, 5);

  container.innerHTML = `
    <div class="section">
      <div class="stat-row">
        <span class="stat-label">Emails scanned</span>
        <span class="stat-value">${total}</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Safe rate</span>
        <span class="stat-value" style="color:${barColor}">${safePct}%</span>
      </div>
      <div class="bar-wrap">
        <div class="bar-fill" style="width:${safePct}%;background:${barColor}"></div>
      </div>
      <div class="breakdown">
        <div class="bd-item bd-low">     ✅ Safe     <strong>${bd.LOW      || 0}</strong></div>
        <div class="bd-item bd-medium">  ⚠️ Caution  <strong>${bd.MEDIUM   || 0}</strong></div>
        <div class="bd-item bd-high">    🚫 Danger   <strong>${bd.HIGH     || 0}</strong></div>
        <div class="bd-item bd-critical">⛔ Critical <strong>${bd.CRITICAL || 0}</strong></div>
      </div>
      ${scanning
        ? `<div class="scan-progress"><div class="spinner"></div> Scanning inbox, please wait...</div>`
        : `<div class="scan-time">Last: ${lastScan} · Next: ${nextScan}</div>`}
    </div>

    ${dangerItems.length ? `
    <div class="section">
      <div class="stat-label" style="margin-bottom:8px;font-weight:600">⚠️ Top threats</div>
      <div class="email-list">
        ${dangerItems.map(r => `
          <div class="email-item">
            <span>${RISK_ICONS[r.risk_level] || "❓"}</span>
            <span class="email-from" title="${escHtml(r.from || "")}">${escHtml(r.from || "unknown")}</span>
            <span class="email-score" style="color:${RISK_COLORS[r.risk_level]}">${r.risk_score}/100</span>
          </div>
        `).join("")}
      </div>
    </div>
    ` : scanning ? `` : `
    <div class="section" style="text-align:center;color:#4CAF50;padding:12px">
      ✅ No threats detected
    </div>`}

    <div class="section" style="display:flex;flex-direction:column;gap:6px">
      <button class="btn btn-primary" id="refreshBtn">🔄 Refresh</button>
      <button class="btn btn-danger ${scanning ? "btn-scanning" : ""}" id="scanBtn"
        ${scanning ? "disabled" : ""}>
        ${scanning ? '<span class="scanning-pulse">⏳ Scanning…</span>' : "📬 Scan Now"}
      </button>
      <button class="btn btn-secondary" id="stopBtn">⏹ Stop Agent</button>
    </div>
  `;

  document.getElementById("refreshBtn").onclick = async () => {
    document.getElementById("refreshBtn").textContent = "Refreshing…";
    await sendMsg("REFRESH");
    await loadAndRender();
  };

  document.getElementById("scanBtn").onclick = async () => {
    const btn = document.getElementById("scanBtn");
    btn.disabled = true;
    btn.innerHTML = '<span class="scanning-pulse">⏳ Starting scan…</span>';
    const result = await sendMsg("SCAN_NOW");
    if (result && result.ok) {
      await loadAndRender(); // will call startScanPoll automatically
    } else {
      alert("Scan error: " + ((result && (result.message || result.error)) || "Unknown error"));
      btn.disabled = false;
      btn.innerHTML = "📬 Scan Now";
    }
  };

  document.getElementById("stopBtn").onclick = async () => {
    stopScanPoll();
    document.getElementById("stopBtn").textContent = "Stopping…";
    await sendMsg("STOP_AGENT");
    setTimeout(() => loadAndRender(), 800);
  };
}

// ── Helpers ───────────────────────────────────────────────────────
function sendMsg(type, extra = {}) {
  return new Promise(resolve => {
    const timeout = setTimeout(() => resolve(null), 5000); // 5s timeout
    chrome.runtime.sendMessage({ type, ...extra }, resp => {
      clearTimeout(timeout);
      if (chrome.runtime.lastError) resolve(null);
      else resolve(resp);
    });
  });
}

function escHtml(str) {
  return String(str)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
    .replace(/"/g,"&quot;").replace(/'/g,"&#039;");
}