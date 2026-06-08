#!/usr/bin/env python3
# monitor/run.py — Email Sentinel V3 main entry point
#
# Usage:
#   python monitor/run.py                  # full agent (IMAP + API + Telegram)
#   python monitor/run.py --scan-now       # one scan, then exit
#   python monitor/run.py --api-only       # API server only (no IMAP)
#   python monitor/run.py --v2             # enable V2 network investigation

import argparse
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def main():
    parser = argparse.ArgumentParser(description="Email Sentinel V3 — The Monitor")
    parser.add_argument("--scan-now", action="store_true",
                        help="Run one scan immediately and exit")
    parser.add_argument("--api-only", action="store_true",
                        help="Start only the local API server (no IMAP scanning)")
    parser.add_argument("--v2", action="store_true",
                        help="Enable V2 network investigation per email")
    args = parser.parse_args()

    _check_config(args)

    from monitor import api_server, scheduler

    if args.scan_now:
        print("Running one-time scan...")
        results = scheduler.run_scan(use_v2=args.v2)
        _print_summary(results)
        return

    if args.api_only:
        print(f"[V3] Starting API server only at http://{config.API_HOST}:{config.API_PORT}")
        api_server.start()
        return

    # Full agent: API + scheduler
    print("=" * 54)
    print("  📧  Email Sentinel V3 — The Monitor")
    print("=" * 54)
    print(f"  IMAP      : {config.IMAP_USER}@{config.IMAP_HOST}")
    print(f"  Scan limit: {config.IMAP_SCAN_LIMIT} emails per cycle")
    print(f"  Telegram  : every {config.TELEGRAM_INTERVAL_HOURS} hours")
    print(f"  API       : http://{config.API_HOST}:{config.API_PORT}")
    print(f"  V2 mode   : always enabled")
    print("=" * 54)

    api_server.start_background()
    scheduler.start_background(use_v2=args.v2)

    print("\n[V3] Agent running. Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        scheduler.stop()
        print("\n[V3] Stopped.")


def _check_config(args):
    errors = []
    if not args.api_only:
        if not config.IMAP_HOST:
            errors.append("IMAP_HOST not set in config.py")
        if not config.IMAP_USER:
            errors.append("IMAP_USER not set in config.py")
        if not config.IMAP_PASSWORD:
            errors.append("IMAP_PASSWORD not set in config.py")
    if not config.TELEGRAM_BOT_TOKEN and not args.api_only:
        print("[Warning] TELEGRAM_BOT_TOKEN not set — Telegram alerts disabled.")
    if errors:
        print("\n[Config Error] Fix these before running:")
        for e in errors:
            print(f"  ✗ {e}")
        print("\nEdit config.py and fill in the missing values.")
        sys.exit(1)


def _print_summary(results: list[dict]):
    total = len(results)
    if not total:
        print("No results.")
        return
    counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for r in results:
        l = r.get("risk_level", "UNKNOWN")
        if l in counts:
            counts[l] += 1
    safe_pct = round(counts["LOW"] / total * 100)
    print(f"\n  Total scanned : {total}")
    print(f"  Safe (LOW)    : {counts['LOW']} ({safe_pct}%)")
    print(f"  Caution       : {counts['MEDIUM']}")
    print(f"  Dangerous     : {counts['HIGH']}")
    print(f"  Critical      : {counts['CRITICAL']}\n")


if __name__ == "__main__":
    main()
