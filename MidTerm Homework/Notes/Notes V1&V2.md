# Email Sentinel V1 & V2 — Flask Web App

A local phishing and email threat analyzer powered by rule-based detection, AI (Ollama), and optional network investigation. Built in Python + Flask, with two analysis modes: **V1** and **V2**.

## Project Structure

```
flask_app/
├── app.py               ← Flask routes — connects V1/V2 to the web UI
├── templates/
│   └── index.html       ← Single-page UI
├── v1/                  ← V1 engine (local only)
│   └── core/
│       ├── parser.py    ← Parse email address or raw headers
│       ├── signals.py   ← All local detection checks
│       └── analyzer.py  ← Scoring pipeline + Ollama prompt
└── v2/                  ← V2 engine (V1 + network)
    ├── core/            ← Same pipeline as V1
    └── investigators/
        ├── whois_check.py
        ├── mx_check.py
        ├── auth_check.py   ← SPF / DMARC via checkdmarc
        └── pipeline.py     ← Runs all 3 in parallel
```

## How It Connects to Flask

```json
{ "email": "support@paypa1.com", "version": "v1" }
```

Flask routes this to either `run_v1()` or `run_v2()` in `app.py`, which runs the full detection pipeline and returns a JSON report. The frontend renders the result — risk score, signals, AI analysis, and network intel — without any page reload.

```
Browser  →  POST /analyze  →  app.py
                               ├── version=v1  →  v1/core/parser → signals → analyzer
                               └── version=v2  →  v2/core/parser → signals → investigators → analyzer
                                                                      ↑
                                                           WHOIS + MX + SPF/DMARC
```

## V1 — Local + AI Analysis

V1 is fast because mostly it worked by using of rules and detection that was already made in the program. There are eight categories, which is typosquatting, homoglyph, brand name, suspicious TLD, reply and return path that was mismatch, suspicious subject, long routing chain. Score calculation was calculated from 0-100. More higher score, more dangerous the email.

### What V1 Detects

**1. Digit Substitution (typosquatting)**
```python
# core/signals.py

def _norm_l(s: str) -> str:
    """Digit→letter mapping where '1'→'l' (paypa1→paypal)."""
    return "".join({"0":"o","1":"l","3":"e","4":"a","5":"s","7":"t"}.get(c,c) for c in s)

def _norm_i(s: str) -> str:
    """Digit→letter mapping where '1'→'i' (m1crosoft→microsoft)."""
    return "".join({"0":"o","1":"i","3":"e","4":"a","5":"s","7":"t"}.get(c,c) for c in s)

# Mapping for more high detection
label = domain.lower().split(".")[0]
if not is_legit and re.search(r"[0-9]", label):
    for norm_fn in (_norm_l, _norm_i):
        hit = False
        for brand in config.IMPERSONATION_TARGETS:
            if brand in norm_fn(label) and brand != label:
                signals.has_digit_substitution = True
                signals.impersonates_brand     = brand
                _note(signals,
                      f"Digit substitution: '{label}' resembles '{brand}'",
                      score=50)
                hit = True
                break
        if hit:
            break
```
Detects when letters in a brand name are replaced with similar-looking digits.

| Phishing domain | Technique | Target brand |
|---|---|---|
| `paypa1.com` | `l` → `1` | paypal |
| `m1crosoft.com` | `i` → `1` | microsoft |
| `g00gle.com` | `o` → `0` | google |
| `amaz0n.com` | `o` → `0` | amazon |

Score added: **+50**. Hard floor: score cannot fall below **75**

**2. Homoglyph Attack**
```python
# core/signals.py

def _has_homoglyph(domain: str) -> bool:
    for char in domain:
        if ord(char) > 127:          # non-ASCII character found
            try:
                name = unicodedata.name(char, "").lower()
                if any(w in name for w in ["latin", "cyrillic", "greek"]):
                    return True      # lookalike script detected
            except Exception:
                return True          # unknown non-ASCII → treat as suspicious
    return False
```
Detects Unicode characters that look identical to Latin letters but are actually different characters (e.g. Cyrillic `а` vs Latin `a`). These bypass naive string matching.

```
pаypal.com  ← the 'а' is Cyrillic U+0430, not Latin
```

Score added: **+40**. Hard floor: **70**.

**3. Brand Name in Domain**
```python
IMPERSONATION_TARGETS = [
    "paypal", "amazon", "apple", "microsoft", "google",
    "netflix", "facebook", "instagram", "whatsapp",
    "bank", "bca", "mandiri", "bri", "bni", "dhl", "fedex",

# core/signals.py
if not is_legit and not signals.impersonates_brand:
    for brand in config.IMPERSONATION_TARGETS:
        if brand in label and label != brand:
            signals.impersonates_brand = brand
            signals.typosquat_target   = brand
            _note(signals,
                  f"Domain contains brand name '{brand}' but is not the official domain",
                  score=25)
            break
]
```
Detects when a known brand name appears in a domain that is not the official domain.

```
microsoft-support.tk   ← contains "microsoft" but not microsoft.com
bca-secure.xyz         ← contains "bca" but not bca.co.id
```

Score added: **+25**.

**4. Suspicious TLD**
```python
# core/signals.py

def _get_tld(domain: str) -> str:
    parts = domain.lower().rsplit(".", 1)
    return f".{parts[-1]}" if len(parts) > 1 else ""
```

```python
# core/signals.py

if not is_legit and signals.tld in config.SUSPICIOUS_TLDS:
    signals.is_suspicious_tld = True
    _note(signals, f"Suspicious TLD: {signals.tld}", score=15)
```
Certain top-level domains are disproportionately used in phishing campaigns.

Flagged TLDs: `.xyz` `.top` `.click` `.pw` `.cc` `.tk` `.gq` `.ml` `.ga` `.cf` `.work` `.men`

Score added: **+15**.

**5. Reply-To Mismatch**
```python
# core/signals.py

if parsed.reply_to and parsed.from_address:
    from_dom    = parsed.from_address.split("@")[-1].lower()
    replyto_dom = parsed.reply_to.split("@")[-1].lower()
    if from_dom != replyto_dom:
        signals.reply_to_mismatch = True
        _note(signals,
              f"Reply-To domain ({replyto_dom}) differs from From domain ({from_dom})",
              score=25)
```
Legitimate emails almost always have the same domain in `From` and `Reply-To`. A mismatch means replies go to a different, attacker-controlled address.

```
From:     bank@bca.co.id
Reply-To: attacker@evil.ru   ← different domain → flagged
```

Score added: **+25**.

**6. Return-Path Mismatch**
```python
# core/signals.py

if parsed.return_path and parsed.from_address:
    from_dom  = parsed.from_address.split("@")[-1].lower()
    rpath_dom = parsed.return_path.split("@")[-1].lower()
    if from_dom != rpath_dom:
        signals.return_path_mismatch = True
        _note(signals,
              f"Return-Path domain ({rpath_dom}) differs from From domain ({from_dom})",
              score=15)
```
Similar to Reply-To mismatch — the `Return-Path` header (where bounces go) differs from the `From` domain.

Score added: **+15**.

**7. Suspicious Subject Keywords**
```python
# core/signals.py

if parsed.subject:
    subj_lower = parsed.subject.lower()
    found = [kw for kw in SUSPICIOUS_SUBJECT_KEYWORDS if kw in subj_lower]
    if found:
        signals.has_suspicious_subject = True
        signals.subject_keywords       = found
        _note(signals, f"Subject contains phishing keywords: {', '.join(found)}", score=10)
```
Scans the subject line for words and phrases commonly used to create urgency in phishing emails.

English keywords: `urgent`, `action required`, `verify your account`, `suspended`, `unusual activity`, `click here`, `password`, `winner`, `congratulations`, `claim your`, `invoice`, `payment due`, `limited time`, `act now`, `account disabled`, `security alert`

Score added: **+10**.

**8. Long Routing Chain**
```python
# core/signals.py

signals.originating_ip     = parsed.originating_ip
signals.received_hop_count = len(parsed.received_chain)
if signals.received_hop_count > 8:
    _note(signals,
          f"Long routing chain: {signals.received_hop_count} hops (possible obfuscation)",
          score=10)
```
Email headers contain a chain of `Received:` entries showing every server the email passed through. More than 8 hops is unusual and may indicate deliberate obfuscation.

Score added: **+10**.

### Scoring Pipeline (V1)

```
pre_score  =  sum of all signal scores above  (capped at 99)
final_score = max(ai_score, pre_score)         (AI cannot lower the score)

Hard floors:
  impersonates_brand OR has_homoglyph  →  final_score >= 70
  has_digit_substitution               →  final_score >= 75
```

### Risk Levels

| Score | Level | Meaning |
|---|---|---|
| 0–29 | 🟢 LOW | Appears safe |
| 30–59 | 🟡 MEDIUM | Proceed with caution |
| 60–94 | 🔴 HIGH | Do not click anything |
| 95–100 | ⛔ CRITICAL | Block immediately |

### Legit Domain Whitelist

Official domains are never flagged for impersonation regardless of signals:
`google.com`, `gmail.com`, `paypal.com`, `apple.com`, `microsoft.com`, `outlook.com`, `netflix.com`, `facebook.com`, `bca.co.id`, `mandiri.co.id`, `bri.co.id`, `bni.co.id`, `github.com`, and more.



## V2 — + Network Investigation

V2 runs everything V1 does, then adds three parallel network investigators that query external DNS and WHOIS services. The system also adding new category of scoring for network investigators.

### What V2 Adds

**1. WHOIS — Domain Age**
```python
# investigators/whois_check.py

data = whois.whois(intel.domain)

created = data.creation_date
if isinstance(created, list):
    created = created[0]          # some registrars return a list

if isinstance(created, datetime.datetime):
    intel.creation_date   = created.strftime("%Y-%m-%d")
    age = (datetime.datetime.now() - created).days
    intel.domain_age_days = age
    if age < config.YOUNG_DOMAIN_DAYS:    # default: 30 days
        intel.is_young_domain = True
        _note(intel,
              f"Very new domain: registered {age} days ago ({intel.creation_date})",
              score=25)
```
Newly registered domains are a strong phishing indicator. Attackers register domains specifically for campaigns and abandon them quickly.

- Domain age < 30 days → **+25**
- Domain age shown in report for context

**2. MX Record Check**
```python
# investigators/mx_check.py

try:
    resolver          = dns.resolver.Resolver()
    resolver.lifetime = config.DNS_TIMEOUT    # 5 seconds

    answers = resolver.resolve(intel.domain, "MX")
    intel.has_mx_records = True
    intel.mx_records = sorted([str(r.exchange).rstrip(".") for r in answers])
    _note(intel, f"MX records found: {', '.join(intel.mx_records[:2])}", score=0)

except dns.resolver.NXDOMAIN:
    # Domain does not exist at all in DNS
    intel.has_mx_records = False
    _note(intel, "Domain does not exist in DNS (NXDOMAIN) — likely a fake domain", score=30)

except dns.resolver.NoAnswer:
    # Domain exists but has no MX records
    intel.has_mx_records = False
    _note(intel, "Domain has no MX records — cannot receive email (one-way spammer?)", score=15)
```
A domain with no MX record cannot legitimately receive email, it only sends. This is a common characteristic of phishing domains.

- No MX records → **+15**
- Domain does not exist in DNS (NXDOMAIN) → **+30**

**3. SPF / DMARC Verification**

**SPF check:**
```python
# investigators/auth_check.py
 
def _process_spf(intel: DomainIntelligence, result: dict):
    spf = result.get("spf", {})
    if not spf:
        intel.spf_valid = False
        _note(intel, "SPF record not found", score=15)
        return
 
    valid = spf.get("valid", False)
    intel.spf_valid  = valid
    intel.spf_record = spf.get("record", "")
 
    if not valid:
        reason = spf.get("error", "unknown")
        _note(intel, f"SPF invalid: {reason}", score=15)
    else:
        _note(intel, "SPF valid ✓", score=0)
```
 
**DMARC check:**
```python
# investigators/auth_check.py
 
def _process_dmarc(intel: DomainIntelligence, result: dict):
    dmarc = result.get("dmarc", {})
    if not dmarc:
        intel.dmarc_valid = False
        _note(intel, "DMARC not found — domain is unprotected against email spoofing", score=20)
        return
 
    valid  = dmarc.get("valid", False)
    policy = dmarc.get("tags", {}).get("p", {}).get("value", "none")
    intel.dmarc_valid  = valid
    intel.dmarc_policy = policy
 
    if not valid:
        _note(intel, "DMARC invalid", score=15)
    elif policy == "none":
        _note(intel, "DMARC present but policy=none — email spoofing is NOT blocked", score=10)
    elif policy == "quarantine":
        _note(intel, "DMARC policy=quarantine ✓", score=0)
    elif policy == "reject":
        _note(intel, "DMARC policy=reject ✓✓", score=-10)   # trust boost
```

Email authentication records tell receiving servers how to handle messages that claim to come from a domain.

| Finding | Score | Meaning |
|---|---|---|
| SPF missing or invalid | +15 | Domain has no sending policy |
| DMARC missing | +20 | No protection against spoofing |
| DMARC policy = `none` | +10 | DMARC exists but not enforced |
| DMARC policy = `quarantine` | 0 | Moderately protected |
| DMARC policy = `reject` | −10 | Strongly protected (trust boost) |

### V2 Scoring Pipeline

```
base_score  =  pre_score (V1 signals) + net_score (network signals)
final_score =  max(ai_score, base_score)
```

Network score is capped at **+60** to prevent network signals from completely overriding local detection.

### When to Use V2

Use V2 when an email looks suspicious but local signals are ambiguous, for example, a domain with no obvious typosquatting but an unfamiliar sender. The network checks will reveal whether the domain was registered yesterday, has no mail infrastructure, or lacks basic email authentication.
