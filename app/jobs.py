"""Background jobs: upload -> stored -> OCR -> field mapping -> result fetched later by job id.

The request that uploads a document returns a job id at once, so no HTTP request ever waits for OCR.
A worker thread picks jobs up, reports page progress, saves the OCR pages and the mapped fields, and marks the job done.
"""
from __future__ import annotations

import hashlib
import logging
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Callable

from .messages import job_message
from .invoice import additional_fields, build_rows, extract_invoice, key_fields, missing_required, validation
from .schemas import InvoiceFields, JobView, OcrLine, OcrResponse, OcrTable, PageResult, Validation
from .storage import JobStore

log = logging.getLogger(__name__)

Runner = Callable[[bytes, Callable[[int, int], None], int], OcrResponse]


class TooManyJobs(RuntimeError):
    pass


class JobNotReady(RuntimeError):
    pass


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rect(bbox) -> list[float] | None:
    if not bbox:
        return None
    xs, ys = [p[0] for p in bbox], [p[1] for p in bbox]
    return [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]


def _poly(rect) -> list[list[float]] | None:
    if not rect:
        return None
    x0, y0, x1, y1 = rect
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def pages_for_store(result: OcrResponse) -> list[dict]:
    """One compact dict per page: page summary + its lines ([text, confidence, [x0, y0, x1, y1]]) + tables."""
    out = []
    for p in result.pages:
        out.append({
            **p.model_dump(),
            "lines": [[l.text, l.confidence, _rect(l.bbox)] for l in result.lines if l.page == p.page],
            "tables": [t.html for t in result.tables if t.page == p.page],
        })
    return out


def result_from_store(rec: dict, pages: list[dict]) -> OcrResponse:
    lines = [OcrLine(text=t, confidence=c, bbox=_poly(b), page=p["page"]) for p in pages for t, c, b in p["lines"]]
    tables = [OcrTable(page=p["page"], html=h) for p in pages for h in p.get("tables", [])]
    page_results = [PageResult(**{k: v for k, v in p.items() if k not in ("lines", "tables")}) for p in pages]
    return OcrResponse(
        fields=InvoiceFields(**(rec.get("fields") or {})),
        additional_fields=rec.get("additional_fields") or {},
        missing_required_fields=rec.get("missing_required_fields") or [],
        validation=Validation(**rec["validation"]) if rec.get("validation") else Validation(),
        text="\n".join(p.text for p in page_results if p.text), lines=lines, tables=tables, pages=page_results,
        average_confidence=rec.get("average_confidence"), page_count=len(page_results),
        pages_total=rec.get("pages_total", len(page_results)), truncated=rec.get("truncated", False),
        needs_review=rec.get("needs_review", False), engine=rec.get("engine") or "",
        processing_seconds=rec.get("processing_seconds"), invoice=rec.get("invoice"),
    )


class JobManager:
    def __init__(self, store: JobStore, runner: Runner, *, max_pending: int = 50, max_attempts: int = 3,
                 lease_seconds: int = 900, poll_seconds: int = 15, ttl_seconds: int = 3600):
        self.store = store
        self._runner = runner
        self._max_pending = max_pending
        self._max_attempts = max_attempts
        self._lease = lease_seconds
        self._poll = poll_seconds
        self._ttl = ttl_seconds
        self._wake: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._current: str | None = None  # job this instance is processing right now
        self._thread = threading.Thread(target=self._loop, name="ocr-job-worker", daemon=True)
        self._thread.start()

    # ---- API side
    def submit(self, data: bytes, filename: str | None, pages_total: int, pages_to_process: int) -> dict:
        self.store.purge(time.time() - self._ttl)
        pending = self.store.pending_count()
        if pending >= self._max_pending:
            raise TooManyJobs(f"{pending} jobs already pending; try again later")
        job_id = uuid.uuid4().hex
        path = self.store.put_file(job_id, filename or "upload", data)
        record = {
            "id": job_id, "status": "queued", "stage": "queued", "filename": filename, "file_path": path,
            "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
            "pages_total": pages_total, "pages_to_process": pages_to_process, "pages_done": 0,
            "attempts": 0, "lease_until": 0, "created_at": _iso(), "started_at": None, "finished_at": None,
            "error": None, "fields": None, "additional_fields": None, "missing_required_fields": None,
            "validation": None,
        }
        self.store.create(record)
        self._wake.put(job_id)
        log.info("Job %s queued (%s, %d pages)", job_id, filename, pages_total)
        return record

    def get(self, job_id: str) -> dict | None:
        self.store.purge(time.time() - self._ttl)
        return self.store.get(job_id)

    def view(self, rec: dict, include_ocr: bool = True, remapped: bool = False) -> JobView:
        result = None
        if include_ocr and rec["status"] == "done":
            pages = self.store.get_ocr_pages(rec["id"])
            if pages is not None:
                result = result_from_store(rec, pages)
                result.message = job_message(rec, remapped)
        return JobView(
            job_id=rec["id"], status=rec["status"], message=job_message(rec, remapped), stage=rec.get("stage"), filename=rec.get("filename"),
            pages_total=rec.get("pages_total", 0), pages_to_process=rec.get("pages_to_process", 0),
            pages_done=rec.get("pages_done", 0), attempts=rec.get("attempts", 0),
            created_at=rec["created_at"], started_at=rec.get("started_at"), finished_at=rec.get("finished_at"),
            processing_seconds=rec.get("processing_seconds"), error=rec.get("error"),
            fields=InvoiceFields(**rec["fields"]) if rec.get("fields") else None,
            additional_fields=rec.get("additional_fields"),
            missing_required_fields=rec.get("missing_required_fields"),
            validation=Validation(**rec["validation"]) if rec.get("validation") else None,
            storage="memory", result=result,
        )

    def remap(self, job_id: str) -> dict:
        """Re-run field mapping from the stored OCR pages (no OCR), e.g. after the mapping rules improve."""
        rec = self.store.get(job_id)
        if rec is None:
            raise KeyError(job_id)
        if rec["status"] != "done":
            raise JobNotReady(f"Job is {rec['status']}; it can be re-mapped once it is done")
        pages = self.store.get_ocr_pages(job_id)
        if pages is None:
            raise JobNotReady("OCR output for this job is no longer stored")
        result = result_from_store(rec, pages)
        inv = extract_invoice(build_rows(result.pages, result.lines))
        fields = key_fields(inv)
        missing = missing_required(fields)
        return self.store.update(job_id, lambda r: {
            **r, "fields": fields.model_dump(), "additional_fields": additional_fields(inv),
            "missing_required_fields": missing, "validation": validation(inv, missing).model_dump(),
            "invoice": inv.model_dump() if inv else None, "remapped_at": _iso()}) or rec

    def shutdown(self) -> None:
        """Graceful stop (App Service restart / deploy): hand the running job back to the queue now, so
        another instance (or this one after restart) resumes it at once instead of waiting for the lease."""
        self._stop.set()
        job_id = self._current
        if job_id:
            try:
                self.store.update(job_id, lambda r: {**r, "status": "queued", "stage": "queued", "lease_until": 0}
                                  if r["status"] == "processing" else None)
                log.info("Job %s handed back to the queue (shutting down)", job_id)
            except Exception:
                log.exception("Job %s: could not release on shutdown; it resumes when its lease expires", job_id)

    # ---- worker side
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                ids = [self._wake.get(timeout=self._poll)]
            except queue.Empty:
                try:
                    ids = self.store.due(time.time())  # queued or retried jobs
                except Exception:
                    log.exception("Could not look for pending jobs")
                    continue
            for job_id in ids:
                if self._stop.is_set():
                    break
                try:
                    self._process(job_id)
                except Exception:
                    log.exception("Job %s: unexpected worker error", job_id)

    def _claim(self, job_id: str) -> dict | None:
        now = time.time()

        def change(r: dict) -> dict | None:
            claimable = r["status"] == "queued" or (r["status"] == "processing" and r.get("lease_until", 0) < now)
            if not claimable:
                return None
            if r.get("attempts", 0) >= self._max_attempts:
                return {**r, "status": "failed", "stage": "failed", "finished_at": _iso(), "finished_epoch": now,
                        "error": r.get("error") or f"Gave up after {r['attempts']} attempts: the service stopped "
                                                    "while processing this document (often: not enough memory)"}
            return {**r, "status": "processing", "stage": "ocr", "attempts": r.get("attempts", 0) + 1,
                    "lease_until": now + self._lease, "started_at": r.get("started_at") or _iso()}

        rec = self.store.update(job_id, change)
        return rec if rec and rec["status"] == "processing" else None

    def _process(self, job_id: str) -> None:
        rec = self._claim(job_id)
        if rec is None:
            return
        log.info("Job %s: processing (attempt %d)", job_id, rec["attempts"])
        started = time.perf_counter()
        self._current = job_id
        try:
            data = self.store.get_file(rec["file_path"])

            def progress(done: int, to_process: int) -> None:
                lease = time.time() + self._lease  # every finished page renews the lease
                self.store.update(job_id, lambda r: {**r, "pages_done": done, "pages_to_process": to_process,
                                                     "lease_until": lease})

            result = self._runner(data, progress, rec["pages_total"])
            self.store.update(job_id, lambda r: {**r, "stage": "saving"})
            self.store.put_ocr_pages(job_id, pages_for_store(result))
            seconds = round(time.perf_counter() - started, 2)
            self.store.update(job_id, lambda r: {
                **r, "status": "done", "stage": "done", "finished_at": _iso(), "finished_epoch": time.time(),
                "error": None, "processing_seconds": seconds,
                "fields": result.fields.model_dump(), "additional_fields": result.additional_fields,
                "missing_required_fields": result.missing_required_fields,
                "validation": result.validation.model_dump() if result.validation else None,
                "invoice": result.invoice.model_dump() if result.invoice else None,
                "average_confidence": result.average_confidence, "needs_review": result.needs_review,
                "page_count": result.page_count, "truncated": result.truncated, "engine": result.engine,
                "pages": [{"page": p.page, "source": p.source, "average_confidence": p.average_confidence,
                           "needs_review": p.needs_review, "seconds": p.seconds} for p in result.pages],
            })
            log.info("Job %s done in %.1fs (%d pages)", job_id, seconds, result.page_count)
        except Exception as exc:
            log.exception("Job %s failed", job_id)
            message = f"{type(exc).__name__}: {exc}"

            def fail(r: dict) -> dict:
                if r.get("attempts", 0) >= self._max_attempts:
                    return {**r, "status": "failed", "stage": "failed", "error": message,
                            "finished_at": _iso(), "finished_epoch": time.time()}
                return {**r, "status": "queued", "stage": "queued", "error": message, "lease_until": 0}

            rec = self.store.update(job_id, fail)
            if rec and rec["status"] == "queued":
                self._wake.put(job_id)  # retry
        finally:
            self._current = None

 