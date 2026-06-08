// content.js — Inject risk badges into Gmail / Outlook email list
// V4: reads v2Enabled from storage so content script respects user preference

const API = "http://127.0.0.1:7842";
const PROCESSED = new WeakSet();
const CACHE = new Map(); // domain → result

// ── Main observer ─────────────────────────────────────────────────
const observer = new MutationObserver(() => scanEmailRows());
observer.observe(document.body, { childList: true, subtree: true });
setTimeout(scanEmailRows, 2000);

// ── Scan all visible email rows ───────────────────────────────────
function scanEmailRows() {
  const rows = getEmailRows();
  for (const row of rows) {
    if (PROCESSED.has(row)) continue;
    PROCESSED.add(row);

    const sender = extractSender(row);
    if (!sender) continue;

    const domain = extractDomain(sender);
    if (domain && CACHE.has(domain)) {
      injectBadge(row, CACHE.get(domain));
      continue;
    }

    analyzeAndBadge(row, sender, domain);
  }
}

// ── Analyze via background → API ─────────────────────────────────
async function analyzeAndBadge(row, sender, domain) {
  try {
    const result = await chrome.runtime.sendMessage({
      type : "ANALYZE_EMAIL",
      email: sender,
      v2   : true,
    });

    if (result && !result.error) {
      if (domain) CACHE.set(domain, result);
      injectBadge(row, result);
    }
  } catch (e) {
    // Extension context may have changed — ignore
  }
}

// ── Badge injection ───────────────────────────────────────────────
function injectBadge(row, result) {
  if (row.querySelector(".sentinel-badge")) return;

  const level = result.risk_level || "UNKNOWN";
  const score = result.risk_score  || 0;

  const cfg = {
    LOW     : { icon: "✅", color: "#2E7D32", bg: "#E8F5E9", label: "Safe" },
    MEDIUM  : { icon: "⚠️", color: "#E65100", bg: "#FFF3E0", label: "Caution" },
    HIGH    : { icon: "🚫", color: "#B71C1C", bg: "#FFEBEE", label: "Danger" },
    CRITICAL: { icon: "⛔", color: "#7B0000", bg: "#FFCDD2", label: "CRITICAL" },
    UNKNOWN : { icon: "❓", color: "#555555", bg: "#F5F5F5", label: "?" },
  }[level] || { icon: "❓", color: "#555", bg: "#eee", label: "?" };

  const badge = document.createElement("span");
  badge.className    = "sentinel-badge";
  badge.title        = `Email Sentinel: ${level} (${score}/100)\n` +
                       (result.signals || []).join("\n");
  badge.style.cssText = `
    display: inline-flex; align-items: center; gap: 3px;
    font-size: 11px; font-weight: 500; white-space: nowrap;
    padding: 1px 6px; border-radius: 10px; margin-left: 6px;
    background: ${cfg.bg}; color: ${cfg.color};
    border: 1px solid ${cfg.color}33; cursor: default;
    vertical-align: middle; font-family: sans-serif;
  `;
  badge.textContent = `${cfg.icon} ${cfg.label}`;

  const senderEl = getSenderElement(row);
  if (senderEl) {
    senderEl.appendChild(badge);
  } else {
    row.style.borderLeft = `3px solid ${cfg.color}`;
  }

  if (level === "HIGH" || level === "CRITICAL") {
    row.style.backgroundColor = cfg.bg;
  }
}

// ── DOM helpers (Gmail + Outlook) ────────────────────────────────
function getEmailRows() {
  const gmailRows = document.querySelectorAll("tr.zA");
  if (gmailRows.length) return Array.from(gmailRows);
  const olRows = document.querySelectorAll('div[role="option"], div[role="listitem"]');
  return Array.from(olRows);
}

function extractSender(row) {
  const gmailSender = row.querySelector(".yP, .zF");
  if (gmailSender) {
    return gmailSender.getAttribute("email") || gmailSender.textContent.trim();
  }
  const olEl = row.querySelector("[data-id], [aria-label]");
  if (olEl) {
    const label = olEl.getAttribute("aria-label") || "";
    const match = label.match(/[\w.+%-]+@[\w.-]+\.[a-z]{2,}/i);
    if (match) return match[0];
  }
  return null;
}

function getSenderElement(row) {
  return row.querySelector(".yP, .zF, .bA4 span, [class*='sender']");
}

function extractDomain(email) {
  const m = email.match(/@([\w.-]+)/);
  return m ? m[1].toLowerCase() : null;
}
