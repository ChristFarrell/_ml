#!/usr/bin/env bash
# Email Sentinel — Native Messaging Host installer
# Jalankan SEKALI saja: bash native_host/install.sh
# Setelah ini, klik "Start Agent" di popup Firefox — tidak perlu terminal lagi.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOST_PY="$SCRIPT_DIR/email_sentinel_host.py"
MANIFEST_TEMPLATE="$SCRIPT_DIR/com.emailsentinel.host.json"
WRAPPER="$SCRIPT_DIR/run_host.sh"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📧 Email Sentinel — Native Host Installer"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Cek Python
PYTHON=$(which python3 2>/dev/null || which python 2>/dev/null || echo "")
if [ -z "$PYTHON" ]; then
  echo "✗ Python tidak ditemukan. Install Python 3 dulu."
  exit 1
fi
echo "  ✓ Python   : $PYTHON ($($PYTHON --version 2>&1))"

# 2. Buat wrapper shell yang panggil python3
cat > "$WRAPPER" << WRAPPER_EOF
#!/usr/bin/env bash
exec "$PYTHON" -u "$HOST_PY" "\$@"
WRAPPER_EOF
chmod +x "$WRAPPER"
chmod +x "$HOST_PY"
echo "  ✓ Wrapper  : $WRAPPER"

# 3. Tulis manifest dengan path wrapper yang benar
MANIFEST_JSON=$(sed "s|__REPLACED_BY_INSTALL_SCRIPT__|$WRAPPER|g" "$MANIFEST_TEMPLATE")

# 4. Deteksi OS → taruh manifest di lokasi Firefox
if [[ "$OSTYPE" == "darwin"* ]]; then
    NM_DIR="$HOME/Library/Application Support/Mozilla/NativeMessagingHosts"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    echo ""
    echo "  Windows terdeteksi."
    echo "  Tambahkan registry key ini secara manual:"
    echo "  HKCU\\SOFTWARE\\Mozilla\\NativeMessagingHosts\\com.emailsentinel.host"
    echo "  Value: path ke com.emailsentinel.host.json"
    exit 0
else
    NM_DIR="$HOME/.mozilla/native-messaging-hosts"
fi

mkdir -p "$NM_DIR"
echo "$MANIFEST_JSON" > "$NM_DIR/com.emailsentinel.host.json"
echo "  ✓ Manifest : $NM_DIR/com.emailsentinel.host.json"

# 5. Verifikasi manifest bisa dibaca
if python3 -c "import json; json.load(open('$NM_DIR/com.emailsentinel.host.json'))" 2>/dev/null; then
    echo "  ✓ Manifest JSON valid"
else
    echo "  ✗ Manifest JSON invalid — cek output di atas"
    exit 1
fi

# 6. Buat folder data/ jika belum ada (untuk log & PID)
mkdir -p "$SCRIPT_DIR/../data"
echo "  ✓ data/    : siap untuk log & PID file"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Instalasi selesai!"
echo ""
echo "  Langkah berikutnya:"
echo "  1. Buka Firefox → about:debugging"
echo "  2. Load Temporary Add-on → pilih extension/manifest.json"
echo "  3. Buka Gmail → klik icon Email Sentinel → 'Start Agent'"
echo ""
echo "  Log agent akan tersimpan di:"
echo "  $(cd "$SCRIPT_DIR/.." && pwd)/data/sentinel_agent.log"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""