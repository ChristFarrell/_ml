# monitor/scheduler.py — Background agent: scan inbox every N hours, send Telegram digest

import time
import threading
import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from monitor.imap_scanner import scan_inbox
from monitor.telegram_bot import send_summary, send_alert

# Shared state — also read by the API server
state = {
    "last_results"     : [],
    "last_scan_time"   : None,
    "scan_count"       : 0,
    "running"          : False,
    "scan_in_progress" : False,
    "next_scan_time"   : None,
}
_lock = threading.Lock()


def run_scan(use_v2: bool = False) -> list[dict]:
    """Run a single scan cycle and update shared state."""
    print(f"\n[Scheduler] Starting scan at {datetime.datetime.now().strftime('%H:%M:%S')}...")

    try:
        results = scan_inbox(use_v2=use_v2)
    except Exception as e:
        print(f"[Scheduler] Scan failed: {e}")
        return []

    with _lock:
        state["last_results"]   = results
        state["last_scan_time"] = datetime.datetime.now().isoformat()
        state["scan_count"]    += 1
        interval = config.TELEGRAM_INTERVAL_HOURS * 3600
        state["next_scan_time"] = (
            datetime.datetime.now() +
            datetime.timedelta(seconds=interval)
        ).isoformat()

    # Send immediate alerts for HIGH/CRITICAL
    for r in results:
        if r.get("risk_level") in ("HIGH", "CRITICAL"):
            send_alert(r)

    # Send digest with descriptions of first 10 emails
    send_summary(results, scan_number=state["scan_count"])

    return results


def start(use_v2: bool = False):
    """
    Start the background scheduler loop.
    Runs scan immediately, then every TELEGRAM_INTERVAL_HOURS hours.
    """
    with _lock:
        if state["running"]:
            print("[Scheduler] Already running.")
            return
        state["running"] = True

    interval_sec = config.TELEGRAM_INTERVAL_HOURS * 3600
    print(f"[Scheduler] Started — scanning every {config.TELEGRAM_INTERVAL_HOURS} hours.")

    while state["running"]:
        with _lock:
            state["scan_in_progress"] = True
        try:
            run_scan(use_v2=use_v2)
        finally:
            with _lock:
                state["scan_in_progress"] = False
        print(f"[Scheduler] Next scan in {config.TELEGRAM_INTERVAL_HOURS} hours.")
        time.sleep(interval_sec)


def start_background(use_v2: bool = False) -> threading.Thread:
    """Start the scheduler in a daemon thread and return it."""
    t = threading.Thread(target=start, args=(use_v2,), daemon=True, name="SentinelScheduler")
    t.start()
    return t


def stop():
    """Gracefully stop the scheduler."""
    with _lock:
        state["running"] = False
    print("[Scheduler] Stopped.")
