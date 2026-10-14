"""Run a whole document through text-layer extraction / OCR, page by page."""
from __future__ import annotations

import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
from typing import Callable

from .config import Settings
from .engine import OcrEngine, PageOcr, join_rows, order_lines, page_confidence
from .messages import extracted
from .invoice import additional_fields, build_rows, extract_invoice, key_fields, missing_required, validation
from .preprocess import count_pages, iter_pages
from .schemas import OcrResponse, PageResult

log = logging.getLogger(__name__)

TEXT_LAYER = "pdf-text-layer"
TEXT_LAYER_OCR = "pdf-text-layer+ocr"


def process_document(
    data: bytes, engine: OcrEngine, settings: Settings,
    on_page: Callable[[int, int], None] | None = None, pages_total: int | None = None,
) -> OcrResponse:
    """OCR every page (up to settings.max_pages). `on_page(done, to_process)` reports progress."""
    started = time.perf_counter()
    if pages_total is None:
        pages_total = count_pages(data)
    to_process = min(pages_total, settings.max_pages)
    if pages_total > to_process:
        log.warning("Document has %d pages; processing the first %d (OCR_MAX_PAGES)", pages_total, to_process)

    pages_iter = iter_pages(
        data, max_pages=to_process, max_side=settings.max_image_side, dpi=settings.pdf_render_dpi,
        use_text_layer=settings.pdf_text_layer, min_text_chars=settings.min_text_layer_chars,
    )
    done: dict[int, tuple[PageOcr, PageResult]] = {}

    def finished(index: int, item: tuple[PageOcr, PageResult]) -> None:
        done[index] = item
        log.info("Page %d/%d done in %.1fs (%s)", index + 1, to_process, item[1].seconds, item[0].engine)
        if on_page:
            on_page(len(done), to_process)

    parallel = max(1, getattr(engine, "parallel", 1))
    with closing(pages_iter):
        if parallel == 1:
            for page in pages_iter:
                finished(page.index, _read_page(page, engine, settings))
        else:
            # Pages are rendered in order; at most `parallel` are being read at once (bounded memory).
            with ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="page") as pool:
                running = {}
                for page in pages_iter:
                    while len(running) >= parallel:
                        for fut in wait(running, return_when=FIRST_COMPLETED).done:
                            finished(running.pop(fut), fut.result())
                    running[pool.submit(_read_page, page, engine, settings)] = page.index
                for fut in wait(running).done:
                    finished(running[fut], fut.result())

    lines, tables, pages = [], [], []
    for index in sorted(done):  # back in page order
        ocr, result = done[index]
        lines.extend(ocr.lines)
        tables.extend(ocr.tables)
        pages.append(result)

    avg = page_confidence(lines)
    invoice = None
    if settings.extract_invoice:
        try:
            invoice = extract_invoice(build_rows(pages, lines))
        except Exception:  # field extraction is best-effort; never lose the OCR result over it
            log.exception("Invoice field extraction failed")
    fields = key_fields(invoice)
    extra = additional_fields(invoice)
    missing = missing_required(fields)
    if missing:
        log.warning("Required field(s) not found: %s", ", ".join(missing))
    log.info("OCR done: pages=%d/%d lines=%d avg_conf=%s sources=%s in %.1fs",
             len(pages), pages_total, len(lines), f"{avg:.3f}" if avg is not None else "n/a",
             _source_counts(pages), time.perf_counter() - started)
    checks = validation(invoice, missing)
    return OcrResponse(
        message=extracted(len(pages), fields, extra, missing, checks),
        fields=fields, additional_fields=extra, missing_required_fields=missing,
        validation=checks,
        text="\n".join(p.text for p in pages if p.text),
        lines=lines, tables=tables, pages=pages,
        average_confidence=round(avg, 4) if avg is not None else None,
        page_count=len(pages), pages_total=pages_total, truncated=pages_total > len(pages),
        needs_review=any(p.needs_review for p in pages), engine=engine.name, invoice=invoice,
        processing_seconds=round(time.perf_counter() - started, 2),
    )


def _read_page(page, engine: OcrEngine, settings: Settings) -> tuple[PageOcr, PageResult]:
    page_started = time.perf_counter()
    if page.text_lines is not None and page.image is not None:
        # Digital page with a big image (letterhead/logo block): keep the exact text layer and add
        # whatever OCR reads outside it, e.g. a vendor name or GSTIN printed inside the image.
        extra = [l for l in engine.process(page.image, page.index).lines
                 if not any(_overlaps(l, t) for t in page.text_lines)]
        merged = order_lines(page.text_lines + extra)
        ocr = PageOcr(lines=merged, engine=TEXT_LAYER_OCR)
        text = join_rows(merged)
    elif page.text_lines is not None:
        ocr = PageOcr(lines=page.text_lines, engine=TEXT_LAYER)
        text = join_rows(page.text_lines)
    else:
        ocr = engine.process(page.image, page.index)
        text = ocr.text
    result = _page_result(page.index, ocr, text, settings)
    result.seconds = round(time.perf_counter() - page_started, 2)
    return ocr, result


def _page_result(index: int, ocr: PageOcr, text: str, settings: Settings) -> PageResult:
    conf = page_confidence(ocr.lines)
    return PageResult(
        page=index, source=ocr.engine, text=text,
        average_confidence=round(conf, 4) if conf is not None else None,
        needs_review=conf is not None and conf < settings.review_below_confidence,
    )


def _overlaps(a, b) -> bool:
    """True when OCR box a mostly lies on text-layer box b (same text read twice)."""
    if not a.bbox or not b.bbox:
        return False
    ax0, ay0 = min(p[0] for p in a.bbox), min(p[1] for p in a.bbox)
    ax1, ay1 = max(p[0] for p in a.bbox), max(p[1] for p in a.bbox)
    bx0, by0 = min(p[0] for p in b.bbox), min(p[1] for p in b.bbox)
    bx1, by1 = max(p[0] for p in b.bbox), max(p[1] for p in b.bbox)
    w, h = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
    area = max((ax1 - ax0) * (ay1 - ay0), 1.0)
    return w > 0 and h > 0 and w * h / area > 0.3


def _source_counts(pages: list[PageResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in pages:
        counts[p.source] = counts.get(p.source, 0) + 1
    return counts

 