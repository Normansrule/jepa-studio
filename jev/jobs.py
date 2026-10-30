"""Background jobs with an append-only event log: chat turns, science renders, model pulls/imports.

The UI polls `GET .../events?since=<n>` and gets `{events, next, status}`; every event is a dict
with `i` (its index), `t` (unix time) and `event` (see docs/jev/api.md for the shapes).
Cancellation is cooperative: `job.cancelled()` turns true and running tools / sandboxed code are
stopped at their next check (sandboxed processes are killed within ~0.1 s).
"""
from __future__ import annotations

import secrets
import threading
import time
import traceback
from typing import Callable

MAX_EVENTS = 20000
KEEP_FINISHED = 300
TERMINAL = {"done", "error", "cancelled"}


class Job:
    def __init__(self, kind: str, jid: str | None = None, meta: dict | None = None):
        self.id = jid or ("j-" + secrets.token_hex(6))
        self.kind = kind
        self.meta = meta or {}
        self.status = "queued"          # queued | running | waiting_approval | done | error | cancelled
        self.events: list[dict] = []
        self.created = time.time()
        self.finished: float | None = None
        self.cancel_event = threading.Event()
        self.lock = threading.Lock()
        self._dropped = 0

    def emit(self, event: str, **fields) -> dict:
        with self.lock:
            rec = {"i": len(self.events), "t": round(time.time(), 3), "event": event, **fields}
            if len(self.events) >= MAX_EVENTS and event == "token":
                self._dropped += 1          # keep the log bounded; final text is still in `done`
                return rec
            self.events.append(rec)
            return rec

    def since(self, n: int, limit: int = 1000) -> dict:
        with self.lock:
            n = max(0, int(n))
            ev = self.events[n:n + limit]
            return {"events": ev, "next": n + len(ev), "status": self.status}

    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def cancel(self) -> None:
        self.cancel_event.set()

    def to_json(self) -> dict:
        return {"id": self.id, "kind": self.kind, "status": self.status, "created": self.created,
                "finished": self.finished, "meta": self.meta, "events": len(self.events)}


class JobManager:
    """Runs each job in a daemon thread. Heavy jobs (renders, pulls) share a small semaphore so two
    renders never fight over the GPU; chat turns do not wait on it."""

    def __init__(self, heavy_slots: int = 1):
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.heavy = threading.Semaphore(heavy_slots)

    def start(self, kind: str, fn: Callable[[Job], None], meta: dict | None = None, jid: str | None = None,
              heavy: bool = False) -> Job:
        job = Job(kind, jid, meta)
        with self.lock:
            self.jobs[job.id] = job
            self._prune()

        def run():
            if heavy:
                while not self.heavy.acquire(timeout=0.2):
                    if job.cancelled():
                        job.status = "cancelled"
                        job.emit("cancelled")
                        job.finished = time.time()
                        return
            try:
                job.status = "running"
                fn(job)
                if job.status not in TERMINAL:
                    job.status = "cancelled" if job.cancelled() else "done"
                    if job.status == "cancelled" and not any(e["event"] == "cancelled" for e in job.events[-3:]):
                        job.emit("cancelled")
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                job.status = "error"
                job.emit("error", message=f"{type(e).__name__}: {str(e)[:500]}")
            finally:
                job.finished = time.time()
                if heavy:
                    self.heavy.release()
        threading.Thread(target=run, name=f"jev-{kind}-{job.id}", daemon=True).start()
        return job

    def get(self, jid: str) -> Job | None:
        with self.lock:
            return self.jobs.get(jid)

    def cancel(self, jid: str) -> bool:
        j = self.get(jid)
        if j is None:
            return False
        j.cancel()
        return True

    def _prune(self) -> None:
        done = sorted((j for j in self.jobs.values() if j.finished), key=lambda j: j.finished)
        for j in done[:-KEEP_FINISHED] if len(done) > KEEP_FINISHED else []:
            self.jobs.pop(j.id, None)

    def wait(self, jid: str, timeout: float = 30.0) -> Job:
        """Testing / CLI helper: block until the job finishes."""
        deadline = time.monotonic() + timeout
        j = self.get(jid)
        while j and j.finished is None and time.monotonic() < deadline:
            time.sleep(0.02)
        return j
