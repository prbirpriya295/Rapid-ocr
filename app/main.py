"""Document Processing Service: PP-OCRv6 via RapidOCR/OpenVINO, PDF text-layer extraction, background jobs."""
from __future__ import annotations
 
import logging
from contextlib import asynccontextmanager
from pathlib import Path
 
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
 
from .config import get_settings
from .engine import OcrEngine, build_engine
from .jobs import JobManager, JobNotReady, TooManyJobs
from .pipeline import process_document
from .preprocess import UnsupportedDocument, count_pages
from .storage import build_store
from .schemas import JobView, OcrResponse
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # model downloads log every HTTP request
log = logging.getLogger("ocr-service")
 
state: dict = {}
WEB_UI = Path(__file__).parent / "static" / "index.html"
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Load models once at startup (they are baked into the image).
    engine: OcrEngine = build_engine(settings)
    state["engine"] = engine
    state["jobs"] = JobManager(
        build_store(settings),
        runner=lambda data, on_page, total: process_document(data, engine, get_settings(), on_page, total),
        max_pending=settings.max_pending_jobs, max_attempts=settings.job_max_attempts,
        lease_seconds=settings.job_lease_seconds, poll_seconds=settings.job_poll_seconds,
        ttl_seconds=settings.job_ttl_seconds,
    )
    yield
    state["jobs"].shutdown()
    state.clear()
 
 
app = FastAPI(title="Document Processing Service", version="2.0.0", lifespan=lifespan)
# Lets the web UI call the API when opened from another origin (e.g. index.html opened as a file).
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins, allow_methods=["GET", "POST"],
                   allow_headers=["*"])
 
 
@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    """Errors carry the same status/message pair as successful responses ("detail" kept for older clients)."""
    message = exc.detail if isinstance(exc.detail, str) else "Request failed"
    return JSONResponse({"status": "error", "message": message, "detail": exc.detail},
                        status_code=exc.status_code, headers=getattr(exc, "headers", None))
 
 
@app.exception_handler(RequestValidationError)
async def bad_request(request: Request, exc: RequestValidationError):
    missing_file = any(e.get("loc", [None])[-1] == "file" for e in exc.errors())
    message = ("No file uploaded: send the document as multipart form field 'file'." if missing_file
               else "Invalid request: " + "; ".join(e.get("msg", "") for e in exc.errors()))
    return JSONResponse({"status": "error", "message": message, "detail": exc.errors()}, status_code=422)
 
 
@app.get("/", include_in_schema=False)
def web_ui():
    return FileResponse(WEB_UI)
 
 
@app.get("/health")
def health():
    engine = state.get("engine")
    jobs = state.get("jobs")
    return {"status": "ok" if engine else "starting",
            "message": "Service is ready." if engine else "OCR engine is loading; try again in a minute.",
            "engine": getattr(engine, "name", None),
            "storage": "memory" if jobs else None}
 
 
@app.post("/v1/ocr", response_model=OcrResponse)
async def ocr(file: UploadFile = File(...)) -> OcrResponse:
    """Synchronous OCR for short documents (up to OCR_SYNC_MAX_PAGES pages)."""
    settings = get_settings()
    engine = _engine()
    data = await _read_upload(file)
    pages_total = await _count_pages(data)
    if pages_total > settings.sync_max_pages:
        raise HTTPException(
            413, f"Document has {pages_total} pages; /v1/ocr handles up to {settings.sync_max_pages}. "
                 "Submit it to POST /v1/jobs and poll GET /v1/jobs/{job_id}.")
    return await run_in_threadpool(process_document, data, engine, settings, None, pages_total)
 
 
@app.post("/v1/jobs", response_model=JobView, status_code=202)
async def submit_job(file: UploadFile = File(...)) -> JobView:
    """Upload a document of any length. Returns at once with a job_id (the process id); the document is
    stored, OCR'd and mapped in the background. Fetch the result with GET /v1/jobs/{job_id}."""
    settings = get_settings()
    _engine()
    data = await _read_upload(file)
    pages_total = await _count_pages(data)
    jobs = state["jobs"]
    try:
        record = await run_in_threadpool(jobs.submit, data, file.filename, pages_total,
                                         min(pages_total, settings.max_pages))
    except TooManyJobs as exc:
        raise HTTPException(429, str(exc)) from exc
    return jobs.view(record, include_ocr=False)
 
 
@app.get("/v1/jobs/{job_id}", response_model=JobView)  # nulls kept: every key field is always present
def get_job(job_id: str, include_ocr: bool = Query(
        True, description="false: status + mapped fields only (small, fast); true: also the full OCR detail")) -> JobView:
    jobs = state.get("jobs")
    rec = jobs.get(job_id) if jobs else None
    if rec is None:
        raise HTTPException(404, "Job not found (unknown id, or it expired / the service restarted)")
    return jobs.view(rec, include_ocr=include_ocr)
 
 
@app.post("/v1/jobs/{job_id}/remap", response_model=JobView)
def remap_job(job_id: str) -> JobView:
    """Re-run field mapping from the stored OCR (no new OCR), e.g. after the mapping rules improve."""
    jobs = state.get("jobs")
    try:
        rec = jobs.remap(job_id)
    except KeyError:
        raise HTTPException(404, "Job not found")
    except JobNotReady as exc:
        raise HTTPException(409, str(exc)) from exc
    return jobs.view(rec, include_ocr=False, remapped=True)
 
 
def _engine() -> OcrEngine:
    engine = state.get("engine")
    if engine is None:
        raise HTTPException(503, "OCR engine is still loading")
    return engine
 
 
async def _read_upload(file: UploadFile) -> bytes:
    limit = get_settings().max_upload_mb * 1024 * 1024
    chunks, size = [], 0
    while chunk := await file.read(1024 * 1024):
        size += len(chunk)
        if size > limit:
            raise HTTPException(413, f"File exceeds {get_settings().max_upload_mb} MB")
        chunks.append(chunk)
    if not size:
        raise HTTPException(400, "Empty file")
    return b"".join(chunks)
 
 
async def _count_pages(data: bytes) -> int:
    try:
        return await run_in_threadpool(count_pages, data)
    except UnsupportedDocument as exc:
        raise HTTPException(415, str(exc)) from exc