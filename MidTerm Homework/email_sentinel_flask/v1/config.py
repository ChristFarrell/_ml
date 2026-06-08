# config.py — Konfigurasi Email Sentinel V1

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "llama3.2:1b"   # sesuaikan dengan model yang kamu pull
OLLAMA_TIMEOUT  = 30               # detik

DB_PATH         = "data/sentinel.db"

# Threshold skor risiko
RISK_LOW        = 30
RISK_MEDIUM     = 60
RISK_HIGH       = 80

# Daftar TLD mencurigakan yang umum dipakai phishing
SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".click", ".pw", ".cc", ".tk",
    ".gq", ".ml", ".ga", ".cf", ".work", ".men",
}

# Domain email gratis (bukan berbahaya, tapi catat)
FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "protonmail.com", "tutanota.com", "guerrillamail.com",
    "mailinator.com", "tempmail.com", "10minutemail.com",
}

# Brand populer yang sering ditiru phishing
IMPERSONATION_TARGETS = [
    "paypal", "amazon", "apple", "microsoft", "google",
    "netflix", "facebook", "instagram", "whatsapp",
    "bank", "bca", "mandiri", "bri", "bni", "dhl", "fedex",
]

# Domain RESMI dari brand di atas — jangan pernah flagged sebagai spoofing
LEGIT_DOMAINS = {
    "google.com", "google.co.id", "gmail.com",
    "paypal.com", "paypal.co.id",
    "amazon.com", "amazon.co.id",
    "apple.com", "icloud.com",
    "microsoft.com", "outlook.com", "live.com", "hotmail.com",
    "netflix.com",
    "facebook.com", "fb.com",
    "instagram.com",
    "whatsapp.com",
    "bca.co.id", "klikbca.com",
    "bankmandiri.co.id", "mandiri.co.id",
    "bri.co.id",
    "bni.co.id",
    "dhl.com", "fedex.com",
}
