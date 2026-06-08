#!/usr/bin/env python3
# cli/analyze.py — Entry point CLI untuk Email Sentinel V1
#
# Penggunaan:
#   python -m cli.analyze --email "support@paypa1.com"
#   python -m cli.analyze --file /path/to/header.txt
#   python -m cli.analyze --raw "From: a@b.com\nReply-To: x@evil.ru"
#   python -m cli.analyze --email "x@y.com" --json
#   python -m cli.analyze --history paypa1.com

import argparse
import sys
import os

# Pastikan root project ada di path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .core.parser   import parse_input
from .core.signals  import extract_signals
from .core.analyzer import analyze
from .reports.formatter import format_plain, format_json
from .data import db


def main():
    parser = argparse.ArgumentParser(
        description="Email Sentinel V1 — Detect dangerous emails",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cli.analyze --email "noreply@paypa1.com"
  python -m cli.analyze --file header.txt
  python -m cli.analyze --email "x@evil.xyz" --json
  python -m cli.analyze --history paypa1.com
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--email",   help="Email address to analyze")
    group.add_argument("--raw",     help="Raw email header as string")
    group.add_argument("--file",    help="Path to file containing raw email header")
    group.add_argument("--history", metavar="DOMAIN", help="View analysis history for a domain")

    parser.add_argument("--json",   action="store_true", help="Output in JSON format")
    parser.add_argument("--save",   action="store_true", default=True,
                        help="Save result to database (default: on)")
    parser.add_argument("--no-save", action="store_true", help="Do not save to database")

    args = parser.parse_args()

    # ── Mode history ──────────────────────────────────────────────
    if args.history:
        _show_history(args.history)
        return

    # ── Baca input ────────────────────────────────────────────────
    raw_input = ""
    if args.email:
        raw_input = args.email
    elif args.raw:
        raw_input = args.raw
    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8", errors="replace") as f:
                raw_input = f.read()
        except FileNotFoundError:
            print(f"❌ File not found: {args.file}")
            sys.exit(1)

    if not raw_input.strip():
        print("❌ Empty input.")
        sys.exit(1)

    # ── Pipeline analisis ─────────────────────────────────────────
    print("⏳ Analyzing...", end="", flush=True)

    parsed  = parse_input(raw_input)
    signals = extract_signals(parsed)
    report  = analyze(parsed, signals)

    print("\r" + " " * 20 + "\r", end="")  # clear loading text

    # ── Output ────────────────────────────────────────────────────
    if args.json:
        print(format_json(report))
    else:
        print(format_plain(report))

    # ── Simpan ke DB ──────────────────────────────────────────────
    if args.save and not args.no_save:
        try:
            row_id = db.save_report(report)
            print(f"  💾 Saved to database (ID: {row_id})\n")
        except Exception as e:
            print(f"  ⚠️  Failed to save to DB: {e}\n")


def _show_history(domain: str):
    """Display analysis history for a domain."""
    history = db.get_domain_history(domain)
    if not history:
        print(f"\nNo analysis history found for domain: {domain}\n")
        return

    print(f"\n{'━'*50}")
    print(f"  History: {domain} ({len(history)} entries)")
    print(f"{'━'*50}")
    for h in history:
        emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "⛔"}.get(h["risk_level"], "⚪")
        print(f"  {emoji} {h['risk_level']:8s}  {h['risk_score']:3d}/100  {h['created_at'][:19]}  {h['email']}")
    print()


if __name__ == "__main__":
    main()