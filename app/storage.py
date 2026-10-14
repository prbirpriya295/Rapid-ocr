"""Where uploads, job records and OCR output live.

- MemoryStore: in-process (local runs). Jobs are stored in memory.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Callable, Protocol

log = logging.getLogger(__name__)

PENDING = ("queued", "processing")


class JobStore(Protocol):
    durable: bool

    def put_file(self, job_id: str, filename: str, data: bytes) -> str: ...
    def get_file(self, path: str) -> bytes: ...
    def put_ocr_pages(self, job_id: str, pages: list[dict]) -> None: ...
    def get_ocr_pages(self, job_id: str) -> list[dict] | None: ...
    def create(self, record: dict) -> None: ...
    def get(self, job_id: str) -> dict | None: ...
    def update(self, job_id: str, change: Callable[[dict], dict | None]) -> dict | None: ...
    def due(self, now: float, limit: int = 10) -> list[str]: ...
    def pending_count(self) -> int: ...
    def purge(self, older_than: float) -> None: ...


def now_epoch() -> float:
    return time.time()


class MemoryStore:
    durable = False

    def __init__(self) -> None:
        self._records: dict[str, dict] = {}
        self._files: dict[str, bytes] = {}
        self._ocr: dict[str, str] = {}
        self._lock = threading.Lock()

    def put_file(self, job_id: str, filename: str, data: bytes) -> str:
        path = f"{job_id}/upload"
        self._files[path] = data
        return path

    def get_file(self, path: str) -> bytes:
        return self._files[path]

    def put_ocr_pages(self, job_id: str, pages: list[dict]) -> None:
        self._ocr[job_id] = json.dumps(pages)
        self._files.pop(f"{job_id}/upload", None)  # free the upload once processed

    def get_ocr_pages(self, job_id: str) -> list[dict] | None:
        raw = self._ocr.get(job_id)
        return json.loads(raw) if raw else None

    def create(self, record: dict) -> None:
        with self._lock:
            self._records[record["id"]] = dict(record)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            rec = self._records.get(job_id)
            return dict(rec) if rec else None

    def update(self, job_id: str, change: Callable[[dict], dict | None]) -> dict | None:
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return None
            new = change(dict(rec))
            if new is None:  # precondition failed (e.g. already claimed)
                return None
            self._records[job_id] = new
            return dict(new)

    def due(self, now: float, limit: int = 10) -> list[str]:
        with self._lock:
            return [r["id"] for r in self._records.values()
                    if r["status"] == "queued" or (r["status"] == "processing" and r.get("lease_until", 0) < now)
                    ][:limit]

    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for r in self._records.values() if r["status"] in PENDING)

    def purge(self, older_than: float) -> None:
        with self._lock:
            old = [j for j, r in self._records.items()
                   if r["status"] in ("done", "failed") and r.get("finished_epoch", now_epoch()) < older_than]
            for jid in old:
                del self._records[jid]
                self._ocr.pop(jid, None)
                self._files.pop(f"{jid}/upload", None)


def build_store(settings=None) -> JobStore:
    log.info("Job store: in-memory")
    return MemoryStore()
