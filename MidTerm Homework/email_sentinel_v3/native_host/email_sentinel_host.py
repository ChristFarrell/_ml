#!/usr/bin/env python3
"""
Email Sentinel — Native Messaging Host
Firefox Extension <-> Python bridge.

Responsibilities:
  - Receive JSON messages from Firefox extension via stdin (Native Messaging protocol)
  - Spawn / check / stop  monitor/run.py
  - Reply with JSON status via stdout

Native Messaging wire format:
  Each message = 4-byte LE uint32 (payload length) + UTF-8 JSON payload
"""

import sys
import json
import struct
import subprocess
import os
import signal

# Paths
HOST_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(HOST_DIR)
RUN_PY      = os.path.join(PROJECT_DIR, "monitor", "run.py")
PID_FILE    = os.path.join(PROJECT_DIR, "data", "sentinel_agent.pid")
LOG_FILE    = os.path.join(PROJECT_DIR, "data", "sentinel_agent.log")

_agent_proc = None


# Native Messaging I/O

def read_message():
    raw_len = sys.stdin.buffer.read(4)
    if len(raw_len) < 4:
        return None
    msg_len = struct.unpack("<I", raw_len)[0]
    payload = sys.stdin.buffer.read(msg_len)
    return json.loads(payload.decode("utf-8"))


def send_message(obj):
    payload = json.dumps(obj).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("<I", len(payload)))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


# Agent lifecycle

def is_agent_running():
    global _agent_proc
    if _agent_proc and _agent_proc.poll() is None:
        return True
    if os.path.exists(PID_FILE):
        try:
            pid = int(open(PID_FILE).read().strip())
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, ValueError, PermissionError):
            pass
    return False


def spawn_agent():
    global _agent_proc

    if is_agent_running():
        return {"ok": True, "status": "already_running"}

    if not os.path.isfile(RUN_PY):
        return {"ok": False, "error": f"run.py not found at {RUN_PY}"}

    try:
        python_exec = sys.executable
        os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
        log_fd = open(LOG_FILE, "a", buffering=1)  # line-buffered → log muncul realtime

        _agent_proc = subprocess.Popen(
            [python_exec, "-u", RUN_PY, "--v2"],   # -u = unbuffered Python output
            cwd=PROJECT_DIR,
            stdout=log_fd,
            stderr=log_fd,
            start_new_session=True,
        )
        with open(PID_FILE, "w") as f:
            f.write(str(_agent_proc.pid))

        return {"ok": True, "status": "spawned", "pid": _agent_proc.pid, "log": LOG_FILE}

    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def stop_agent():
    global _agent_proc
    killed = False

    if _agent_proc and _agent_proc.poll() is None:
        _agent_proc.terminate()
        try:
            _agent_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _agent_proc.kill()
        _agent_proc = None
        killed = True

    if os.path.exists(PID_FILE):
        try:
            pid = int(open(PID_FILE).read().strip())
            os.kill(pid, signal.SIGTERM)
            killed = True
        except Exception:
            pass
        os.remove(PID_FILE)

    return {"ok": True, "status": "stopped" if killed else "was_not_running"}


# Message dispatch

def handle(msg):
    action = msg.get("action", "")

    if action == "SPAWN_AGENT":
        return spawn_agent()

    if action == "STOP_AGENT":
        return stop_agent()

    if action == "GET_AGENT_STATUS":
        running = is_agent_running()
        pid = None
        if running and os.path.exists(PID_FILE):
            try:
                pid = int(open(PID_FILE).read().strip())
            except Exception:
                pass
        return {"ok": True, "running": running, "pid": pid}

    return {"ok": False, "error": f"Unknown action: {action}"}


def main():
    while True:
        try:
            msg = read_message()
            if msg is None:
                break
            reply = handle(msg)
            send_message(reply)
        except Exception as exc:
            send_message({"ok": False, "error": str(exc)})
            break


if __name__ == "__main__":
    main()