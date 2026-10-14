"""OCR engine: PP-OCRv6 medium (detection + recognition) run through RapidOCR on OpenVINO.

No PaddlePaddle at runtime. Settings mirror PaddleOCR's own OCR pipeline (same models, same thresholds),
and pages are straightened first with PaddleOCR's page-orientation model (PP-LCNet_x1_0_doc_ori),
converted to ONNX at image build time. On the 18-sample test set this gives the same field accuracy as
PaddleOCR (117/118) and is faster on CPU.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .config import Settings
from .schemas import OcrLine, OcrTable

log = logging.getLogger(__name__)


@dataclass
class PageOcr:
    lines: list[OcrLine]  # in reading order
    engine: str
    tables: list[OcrTable] = field(default_factory=list)

    @property
    def text(self) -> str:
        return join_rows(self.lines)


class OcrEngine(Protocol):
    name: str

    def process(self, page: np.ndarray, page_index: int) -> PageOcr: ...


def available_cpus() -> int:
    """CPU threads available on this machine."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def page_confidence(lines: list[OcrLine]) -> float | None:
    """Length-weighted mean so a confident long line counts more than a stray character."""
    scored = [l for l in lines if l.confidence is not None]
    chars = sum(len(l.text) for l in scored)
    if not chars:
        return None
    return sum(l.confidence * len(l.text) for l in scored) / chars  # type: ignore[operator]


class PageOrienter:
    """Page orientation (0 / 90 / 180 / 270) with PaddleOCR's PP-LCNet_x1_0_doc_ori model (ONNX, on OpenVINO).
    Pre-processing and rotation replicate PaddleOCR's document pre-processor exactly (verified: same angle as
    PaddleOCR on every test rotation)."""

    LABELS = (0, 90, 180, 270)
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, model_path: str, threads: int):
        import openvino as ov

        core = ov.Core()
        self._model = core.compile_model(model_path, "CPU", {"INFERENCE_NUM_THREADS": threads,
                                                             "PERFORMANCE_HINT": "LATENCY"})

    def angle(self, bgr: np.ndarray) -> int:
        import cv2

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # the classifier reads RGB
        h, w = rgb.shape[:2]
        s = 256 / min(h, w)  # resize short side to 256, centre-crop 224
        rgb = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_LINEAR)
        h, w = rgb.shape[:2]
        y, x = (h - 224) // 2, (w - 224) // 2
        crop = (rgb[y:y + 224, x:x + 224].astype(np.float32) / 255.0 - self.MEAN) / self.STD
        scores = self._model(crop.transpose(2, 0, 1)[None])[0]
        return self.LABELS[int(np.argmax(scores))]

    @staticmethod
    def rotate(image: np.ndarray, angle: int) -> np.ndarray:
        """Same rotation as PaddleOCR (counter-clockwise by `angle`, canvas grown to fit, cubic interpolation)."""
        import cv2

        if angle == 0:
            return image
        h, w = image.shape[:2]
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        cos, sin = abs(mat[0, 0]), abs(mat[0, 1])
        new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
        mat[0, 2] += (new_w - w) / 2
        mat[1, 2] += (new_h - h) / 2
        return cv2.warpAffine(image, mat, (new_w, new_h), flags=cv2.INTER_CUBIC)


class RapidEngine:
    """PP-OCRv6 medium via RapidOCR. Settings mirror PaddleOCR's OCR pipeline so the text matches it."""

    name = "rapidocr"

    def __init__(self, settings: Settings):
        from rapidocr import RapidOCR
        from rapidocr.utils.typings import EngineType, LangCls, LangDet, LangRec, ModelType, OCRVersion

        threads = settings.cpu_threads or available_cpus()
        backend = EngineType(settings.rapid_backend)
        thread_key = ("EngineConfig.openvino.inference_num_threads" if backend == EngineType.OPENVINO
                      else "EngineConfig.onnxruntime.intra_op_num_threads")
        params = {
            # PaddleOCR keeps every recognised line (score_thresh 0) and only shrinks images above 4000 px.
            "Global.text_score": 0.0, "Global.max_side_len": 4000, "Global.min_side_len": 30,
            "Global.use_cls": settings.use_angle_cls, "Global.log_level": "warning",
            "Global.model_root_dir": settings.rapid_model_dir, thread_key: threads,
            # PP-OCRv6 medium detection with PaddleOCR's pipeline thresholds.
            "Det.engine_type": backend, "Det.lang_type": LangDet.EN, "Det.ocr_version": OCRVersion.PPOCRV6,
            "Det.model_type": ModelType.MEDIUM, "Det.limit_side_len": 64, "Det.limit_type": "min",
            "Det.thresh": 0.3, "Det.box_thresh": 0.6, "Det.unclip_ratio": 1.5, "Det.use_dilation": False,
            # Text-line orientation: the same PP-LCNet x1_0 textline_ori model PaddleOCR uses (argmax decision).
            "Cls.engine_type": backend, "Cls.lang_type": LangCls.CH, "Cls.ocr_version": OCRVersion.PPOCRV5,
            "Cls.model_type": ModelType.SERVER, "Cls.cls_thresh": 0.5,
            # PP-OCRv6 medium recognition (multilingual, as PaddleOCR loads for lang=en).
            "Rec.engine_type": backend, "Rec.lang_type": LangRec.EN, "Rec.ocr_version": OCRVersion.PPOCRV6,
            "Rec.model_type": ModelType.MEDIUM,
        }
        if backend == EngineType.OPENVINO:
            # One document at a time: a single inference stream per model ("latency" mode) instead of
            # OpenVINO's default throughput mode, which keeps several model copies in memory. Same outputs.
            params.update({"EngineConfig.openvino.performance_hint": "LATENCY",
                           "EngineConfig.openvino.num_streams": 1,
                           "EngineConfig.openvino.performance_num_requests": 1})
        self._ocr = RapidOCR(params=params)
        self._orient = None
        if settings.use_angle_cls:
            if os.path.exists(settings.orientation_model):
                self._orient = PageOrienter(settings.orientation_model, threads)
            else:
                log.warning("Page orientation model %s not found; rotated scans will not be straightened",
                            settings.orientation_model)
        self._lock = threading.Lock()
        log.info("RapidOCR ready (PP-OCRv6 medium, backend=%s, cpu_threads=%s, page orientation=%s)",
                 settings.rapid_backend, threads, bool(self._orient))

    def process(self, page: np.ndarray, page_index: int) -> PageOcr:
        bgr = page[:, :, ::-1].copy()  # models were trained on BGR arrays (OpenCV order)
        with self._lock:
            if self._orient is not None:
                angle = self._orient.angle(bgr)
                if angle:
                    bgr = np.ascontiguousarray(self._orient.rotate(bgr, angle))
                    log.info("Page %d rotated by %d degrees before OCR", page_index + 1, angle)
            out = self._ocr(bgr)
        lines: list[OcrLine] = []
        boxes = out.boxes if out.boxes is not None else []
        for box, text, score in zip(boxes, out.txts or (), out.scores or ()):
            if text and text.strip():
                lines.append(OcrLine(text=text.strip(), confidence=float(score),
                                     bbox=[[float(x), float(y)] for x, y in box], page=page_index))
        return PageOcr(lines=order_lines(lines), engine=self.name)


class FakeEngine:
    """Deterministic engine for tests and running the stack without OCR models."""

    name = "fake"

    SAMPLE = [
        "TAMIL TRADERS PVT LTD",
        "12, Anna Salai, Chennai 600002",
        "GSTIN: 33AABCT1332L1ZL",
        "TAX INVOICE",
        "Invoice No: TT/2026/0457",
        "Date: 28/09/2026",
        "Bill To GSTIN: 33AAGCA4567H1ZP",
        "A4 Paper Ream x 10 2,500.00",
        "Toner Cartridge x 2 3,000.00",
        "Taxable Value 5,500.00",
        "CGST @9% 495.00",
        "SGST @9% 495.00",
        "Grand Total 6,490.00",
    ]

    def process(self, page: np.ndarray, page_index: int) -> PageOcr:
        lines = [
            OcrLine(text=t, confidence=0.97 - (i % 4) * 0.02,
                    bbox=[[20, 40 * i + 10], [600, 40 * i + 10], [600, 40 * i + 40], [20, 40 * i + 40]], page=page_index)
            for i, t in enumerate(self.SAMPLE)
        ]
        return PageOcr(lines=lines, engine=self.name)


def build_engine(settings: Settings) -> OcrEngine:
    kind = settings.engine.lower()
    if kind == "fake":
        log.warning("Using FakeEngine: OCR output is canned sample text")
        return FakeEngine()
    if kind == "rapid":
        return RapidEngine(settings)
    raise ValueError(f"Unknown OCR_ENGINE {settings.engine!r} (expected rapid or fake)")


def _mid_y(line: OcrLine) -> float:
    ys = [p[1] for p in line.bbox] if line.bbox else [0.0]
    return sum(ys) / len(ys)


def _min_x(line: OcrLine) -> float:
    return min(p[0] for p in line.bbox) if line.bbox else 0.0


def group_rows(lines: list[OcrLine]) -> list[list[OcrLine]]:
    """Cluster boxes into visual rows (tolerates slight skew), each row sorted left-to-right."""
    rows: list[list[OcrLine]] = []
    for line in sorted(lines, key=lambda l: (l.page, _mid_y(l))):
        row = rows[-1] if rows else None
        if row and line.bbox and row[0].bbox and line.page == row[0].page and _same_row(row[0], line):
            row.append(line)
        else:
            rows.append([line])
    return [sorted(r, key=_min_x) for r in rows]


def order_lines(lines: list[OcrLine]) -> list[OcrLine]:
    """Reading order: page, then row top-to-bottom, then left-to-right."""
    return [l for row in group_rows(lines) for l in row]


def join_rows(lines: list[OcrLine]) -> str:
    """Join boxes on the same visual row with spaces so 'Total' and '1,180.00' land on one line."""
    return "\n".join(" ".join(l.text for l in row) for row in group_rows(lines))


def _same_row(a: OcrLine, b: OcrLine) -> bool:
    ay = [p[1] for p in a.bbox]  # type: ignore[union-attr]
    by = [p[1] for p in b.bbox]  # type: ignore[union-attr]
    height = max(max(ay) - min(ay), max(by) - min(by), 1)
    return abs(_mid_y(a) - _mid_y(b)) < height * 0.5
