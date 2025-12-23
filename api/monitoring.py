# api/monitoring.py
import os, json, time
from typing import Dict, Any, Generator
from threading import Lock

LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "execution.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

_lock = Lock()
_SSE_BUFFER = []
_SSE_MAX = 300

def emit_event(run_id: str, step: str, status: str = "start", details: Dict[str, Any] = None):
    details = details or {}
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "run_id": run_id,
        "step": step,
        "status": status,
        "details": details
    }
    line = json.dumps(entry, default=str)
    with _lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        _SSE_BUFFER.append(entry)
        if len(_SSE_BUFFER) > _SSE_MAX:
            _SSE_BUFFER.pop(0)

def read_latest(n: int = 200):
    with _lock:
        return list(_SSE_BUFFER)[-n:]

def sse_event_generator(last_event_id: int = 0) -> Generator[str, None, None]:
    seen = last_event_id
    hb = 0
    while True:
        with _lock:
            new = _SSE_BUFFER[seen:]
            seen = len(_SSE_BUFFER)
        for ev in new:
            yield f"data: {json.dumps(ev, default=str)}\n\n"
        hb += 1
        if hb >= 10:
            hb = 0
            yield "data: {\"heartbeat\": true}\n\n"
        time.sleep(0.8)
