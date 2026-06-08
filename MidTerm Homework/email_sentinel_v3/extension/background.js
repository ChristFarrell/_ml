// background.js — Service worker: manage cache, alarms, badge + Native Messaging

const API        = "http://127.0.0.1:7842";
const ALARM_NAME = "sentinel-refresh";
const REFRESH_MINUTES = 15;
const NATIVE_HOST = "com.emailsentinel.host";

// ── Startup ──────────────────────────────────────────────────────
chrome.runtime.onInstalled.addListener(() => {
  console.log("[Sentinel] Extension installed.");
  chrome.alarms.create(ALARM_NAME, { periodInMinutes: REFRESH_MINUTES });
  fetchAndCacheStatus();
});

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === ALARM_NAME) fetchAndCacheStatus();
});

chrome.tabs.onUpdated.addListener((tabId, info, tab) => {
  if (info.status === "complete" && isWebmail(tab.url)) fetchAndCacheStatus();
});

// ── Message bridge (popup <-> background) ────────────────────────
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {

  if (msg.type === "GET_STATUS") {
    // Always fetch live from API — never use stale cache
    fetchLiveStatus().then(sendResponse);
    return true;
  }

  if (msg.type === "GET_RESULTS") {
    // Always fetch live from API
    fetchLiveResults().then(sendResponse);
    return true;
  }

  if (msg.type === "ANALYZE_EMAIL") {
    analyzeEmail(msg.email).then(sendResponse);
    return true;
  }

  if (msg.type === "REFRESH") {
    fetchAndCacheStatus().then(sendResponse);
    return true;
  }

  if (msg.type === "SCAN_NOW") {
    triggerScanNow().then(sendResponse);
    return true;
  }

  // ── NEW: Native Messaging — spawn / stop / status ────────────
  if (msg.type === "SPAWN_AGENT") {
    spawnAgentViaNative().then(sendResponse);
    return true;
  }

  if (msg.type === "STOP_AGENT") {
    nativeSend({ action: "STOP_AGENT" }).then(sendResponse);
    return true;
  }

  if (msg.type === "GET_AGENT_STATUS") {
    nativeSend({ action: "GET_AGENT_STATUS" }).then(sendResponse);
    return true;
  }

  // ── ML ────────────────────────────────────────────────────────
  if (msg.type === "ML_LABEL") {
    submitLabel(msg).then(sendResponse);
    return true;
  }
  if (msg.type === "ML_STATUS") {
    fetchMLStatus().then(sendResponse);
    return true;
  }
  if (msg.type === "ML_TRAIN") {
    triggerTrain(msg.bootstrap || false).then(sendResponse);
    return true;
  }
});

// ── Native Messaging helpers ──────────────────────────────────────

/**
 * Send one message to the native host and get a reply.
 * Returns null on failure (host not installed / not running).
 */
function nativeSend(payload) {
  return new Promise((resolve) => {
    try {
      const port = chrome.runtime.connectNative(NATIVE_HOST);
      let replied = false;

      port.onMessage.addListener((reply) => {
        replied = true;
        port.disconnect();
        resolve(reply);
      });

      port.onDisconnect.addListener(() => {
        if (!replied) {
          const err = chrome.runtime.lastError?.message || "Native host disconnected";
          console.warn("[Sentinel] Native host error:", err);
          resolve({ ok: false, error: err });
        }
      });

      port.postMessage(payload);

    } catch (e) {
      resolve({ ok: false, error: e.message });
    }
  });
}

/**
 * Ask the native host to spawn monitor/run.py,
 * then poll the HTTP API until it comes online.
 */
async function spawnAgentViaNative() {
  const spawnReply = await nativeSend({ action: "SPAWN_AGENT" });

  if (!spawnReply.ok) return spawnReply;

  // Already running or just spawned — wait for HTTP API to be ready
  const ready = await waitForApi(30, 1000); // max 30s, poll every 1s
  if (ready) {
    await fetchAndCacheStatus();
    return { ok: true, status: spawnReply.status, apiReady: true };
  }

  return { ok: true, status: spawnReply.status, apiReady: false, warning: "API did not respond in 30s" };
}

/**
 * Poll GET /status until HTTP 200 or timeout.
 * @param {number} maxSeconds
 * @param {number} intervalMs
 */
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

// ── Core HTTP functions ───────────────────────────────────────────

// ── Live fetch (always direct to API, no stale cache) ────────────

async function fetchLiveStatus() {
  try {
    const res = await fetch(`${API}/status`, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) return null;
    const status = await res.json();
    updateBadge(status);
    chrome.storage.local.set({ sentinelStatus: status, lastFetch: new Date().toISOString() });
    return status;
  } catch {
    return null;
  }
}

async function fetchLiveResults() {
  try {
    const res = await fetch(`${API}/results`, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) return [];
    const { results } = await res.json();
    chrome.storage.local.set({ sentinelResults: results });
    return results;
  } catch {
    return [];
  }
}

async function fetchAndCacheStatus() {
  try {
    const [statusRes, resultsRes] = await Promise.all([
      fetch(`${API}/status`,  { signal: AbortSignal.timeout(5000) }),
      fetch(`${API}/results`, { signal: AbortSignal.timeout(5000) }),
    ]);

    if (!statusRes.ok || !resultsRes.ok) throw new Error("API error");

    const status         = await statusRes.json();
    const { results }    = await resultsRes.json();

    await chrome.storage.local.set({
      sentinelStatus : status,
      sentinelResults: results,
      lastFetch      : new Date().toISOString(),
    });

    updateBadge(status);
    return { status, results };

  } catch (e) {
    chrome.action.setBadgeText({ text: "OFF" });
    chrome.action.setBadgeBackgroundColor({ color: "#888888" });
    await chrome.storage.local.set({ sentinelStatus: null });
    return null;
  }
}

async function analyzeEmail(email) {
  try {
    const res = await fetch(`${API}/analyze`, {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({ email, v2: true }),
      signal : AbortSignal.timeout(30000),
    });
    return await res.json();
  } catch (e) {
    return { error: "API unavailable: " + e.message };
  }
}

async function triggerScanNow() {
  try {
    const res = await fetch(`${API}/scan-now`, {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({ v2: true }),
      signal : AbortSignal.timeout(10000),
    });
    const data = await res.json();
    // Note: polling handled by popup.js — no duplicate poll here
    return data;
  } catch (e) {
    return { error: "API unavailable: " + e.message };
  }
}

function updateBadge(status) {
  if (!status) return;
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

function isWebmail(url) {
  if (!url) return false;
  return url.includes("mail.google.com")  ||
         url.includes("outlook.live.com") ||
         url.includes("outlook.office.com");
}

// ── ML helpers ────────────────────────────────────────────────────

async function submitLabel(msg) {
  try {
    const res = await fetch(`${API}/ml/label`, {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({
        email   : msg.email    || "",
        domain  : msg.domain   || "",
        label   : msg.label,        // "phishing" | "legit"
        features: msg.features || [],
      }),
      signal: AbortSignal.timeout(10000),
    });
    return await res.json();
  } catch (e) { return { error: e.message }; }
}

async function fetchMLStatus() {
  try {
    const res = await fetch(`${API}/ml/status`, { signal: AbortSignal.timeout(5000) });
    return await res.json();
  } catch (e) { return { error: e.message }; }
}

async function triggerTrain(bootstrap = false) {
  try {
    const res = await fetch(`${API}/ml/train`, {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({ bootstrap }),
      signal : AbortSignal.timeout(30000),
    });
    return await res.json();
  } catch (e) { return { error: e.message }; }
}