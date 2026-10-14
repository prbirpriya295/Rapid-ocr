"""Turn an uploaded file (image or PDF) into pages for OCR, one page at a time.

PDF pages are rendered lazily so long documents never sit in memory all at once, and pages with a
usable embedded text layer (digital invoices, exported reports) skip OCR entirely.
"""
from __future__ import annotations

import io
import re
import threading
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from PIL import Image, ImageOps

from .schemas import OcrLine

PDF_MAGIC = b"%PDF"
# PDFium is not thread-safe, and the sync endpoint and the job worker can both use it.
_PDFIUM_LOCK = threading.Lock()
# A page whose largest image covers this much of it is treated as a scan, even if it has a
# (hidden, often poor) OCR text layer: re-OCR it ourselves.
_SCAN_IMAGE_COVERAGE = 0.8
# A digital page that also carries a big image (letterhead, logo block, stamp) is OCR'd as well, and the
# OCR text outside the text layer's boxes is added: the image can hold the vendor name or GSTIN.
_MIXED_IMAGE_COVERAGE = 0.3


class UnsupportedDocument(ValueError):
    pass


@dataclass
class PageInput:
    index: int
    image: np.ndarray | None = None  # RGB; set when the page needs OCR
    text_lines: list[OcrLine] | None = None  # set when the PDF text layer was used (both set: merge the two)


def is_pdf(data: bytes) -> bool:
    return data[:4] == PDF_MAGIC


def count_pages(data: bytes) -> int:
    """Validate the document and return its page count (cheap: nothing is rendered)."""
    if not is_pdf(data):
        _open_image(data)
        return 1
    with _PDFIUM_LOCK:
        pdf = _open_pdf(data)
        n = len(pdf)
        pdf.close()
    if n == 0:
        raise UnsupportedDocument("PDF has no pages.")
    return n


def iter_pages(
    data: bytes, *, max_pages: int, max_side: int, dpi: int,
    use_text_layer: bool = True, min_text_chars: int = 30,
) -> Iterator[PageInput]:
    """Yield pages lazily; at most one rendered page is held in memory at a time."""
    if not is_pdf(data):
        yield PageInput(0, image=_normalise(_open_image(data), max_side))
        return

    with _PDFIUM_LOCK:
        pdf = _open_pdf(data)
        n = min(len(pdf), max_pages)
    try:
        for i in range(n):
            with _PDFIUM_LOCK:
                page = pdf[i]
                try:
                    w, h = page.get_size()
                    scale = dpi / 72 * _resize_ratio(w * dpi / 72, h * dpi / 72, max_side)
                    lines = _text_layer(page, i, scale, min_text_chars) if use_text_layer else None
                    mixed = bool(lines) and _largest_image_coverage(page) >= _MIXED_IMAGE_COVERAGE
                    img = None if lines and not mixed else page.render(scale=dpi / 72).to_pil()
                finally:
                    page.close()
            if lines:
                yield PageInput(i, text_lines=lines, image=_normalise(img, max_side) if img else None)
            else:
                yield PageInput(i, image=_normalise(img, max_side))
    finally:
        with _PDFIUM_LOCK:
            pdf.close()


def _open_pdf(data: bytes):
    import pypdfium2 as pdfium  # imported lazily: only needed for PDFs

    try:
        return pdfium.PdfDocument(data)
    except Exception as exc:
        raise UnsupportedDocument("PDF could not be opened.") from exc


def _open_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # PIL raises several exception types
        raise UnsupportedDocument("File is not a readable image or PDF.") from exc
    # Respect camera EXIF orientation (phone photos are often rotated).
    return ImageOps.exif_transpose(img)


def _text_layer(page, index: int, scale: float, min_chars: int) -> list[OcrLine] | None:
    """Lines from the page's embedded text, in the same pixel space as the rendered page; None if unusable."""
    if page.get_rotation() % 360:
        return None  # char boxes are in unrotated space; let OCR handle rotated pages
    textpage = page.get_textpage()
    try:
        if not _usable_text(textpage.get_text_bounded(), min_chars):
            return None
        if _largest_image_coverage(page) >= _SCAN_IMAGE_COVERAGE:
            return None
        left, _bottom, _right, top = page.get_cropbox()
        lines = []
        for text, (l, b, r, t) in _segments(textpage):
            x1, x2 = (l - left) * scale, (r - left) * scale
            y1, y2 = (top - t) * scale, (top - b) * scale
            lines.append(OcrLine(text=text, confidence=1.0,
                                 bbox=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]], page=index))
        return lines or None
    finally:
        textpage.close()


def _segments(textpage) -> Iterator[tuple[str, tuple[float, float, float, float]]]:
    """Group characters into runs on one baseline, split at line breaks and wide horizontal gaps
    (so table columns become separate boxes, like OCR output)."""
    chars: list[str] = []
    box: list[float] | None = None  # [left, bottom, right, top] in PDF points

    def flush():
        nonlocal chars, box
        text = " ".join("".join(chars).split())
        # Some PDFs emit U+FFFE/U+FFFD for glyphs they cannot map: a hyphen between letters, else drop it.
        text = re.sub("(?<=[A-Za-z])[\ufffe\ufffd](?=[A-Za-z])", "-", text).replace("\ufffe", "").replace("\ufffd", "")
        if text and box:
            yield text, tuple(box)
        chars, box = [], None

    for i in range(textpage.count_chars()):
        ch = textpage.get_text_range(i, 1)
        if not ch or ch in "\r\n":
            yield from flush()
            continue
        if ch.isspace():
            if chars:
                chars.append(" ")
            continue
        l, b, r, t = textpage.get_charbox(i, loose=True)  # font-height boxes: ',' stays on its line
        if box is not None:
            height = max(box[3] - box[1], t - b, 1.0)
            overlap = min(box[3], t) - max(box[1], b)
            if overlap < 0.5 * min(box[3] - box[1], t - b) or l - box[2] > 2.5 * height:
                yield from flush()
        if box is None:
            box = [l, b, r, t]
        else:
            box = [min(box[0], l), min(box[1], b), max(box[2], r), max(box[3], t)]
        chars.append(ch)
    yield from flush()


def _usable_text(text: str, min_chars: int) -> bool:
    visible = [c for c in text if not c.isspace()]
    if len(visible) < min_chars:
        return False
    # Broken font encodings show up as replacement / private-use / control characters.
    good = sum(1 for c in visible if c.isprintable() and c != "\ufffd" and not "" <= c <= "")
    return good / len(visible) >= 0.9


def _largest_image_coverage(page) -> float:
    import pypdfium2.raw as pdfium_c

    w, h = page.get_size()
    largest = 0.0
    for obj in page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,), max_depth=3):
        l, b, r, t = obj.get_pos()
        largest = max(largest, (r - l) * (t - b))
    return largest / (w * h) if w and h else 0.0


def _resize_ratio(w: float, h: float, max_side: int) -> float:
    longest = max(w, h)
    if longest > max_side:
        return max_side / longest
    if longest < 1000:
        # Small thumbnails OCR poorly; upscale modestly.
        return 1000 / longest
    return 1.0


def _normalise(img: Image.Image, max_side: int) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    ratio = _resize_ratio(w, h, max_side)
    if ratio != 1.0:
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS if ratio < 1 else Image.BICUBIC)
    return np.array(img)
