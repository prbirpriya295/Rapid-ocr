from functools import lru_cache

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OCR_", env_file=".env", extra="ignore")

    # "rapid": PP-OCRv6 medium via RapidOCR on OpenVINO (the OCR engine)
    # "fake":  canned text (local dev / tests without OCR models)
    engine: str = "rapid"
    # CPU threads for OCR. 0 = auto (uses all available CPU cores).
    cpu_threads: int = 0
    # Straighten pages (0/90/180/270) and text lines (180) before reading them
    use_angle_cls: bool = True
    rapid_backend: str = "openvino"  # openvino (fastest on CPU)
    rapid_model_dir: str = str(Path.home() / ".rapidocr")  # PP-OCRv6 detection / recognition / line-orientation models
    orientation_model: str = "models/doc_ori.onnx"
    # Pages whose OCR confidence is below this are flagged needs_review. OCR can stay confident on bad
    # photos (a blurry invoice scored 0.896 with a wrong GSTIN and tax amount), so keep this high.
    review_below_confidence: float = 0.93

    # Documents
    max_pages: int = 300  # longer documents are processed up to this page and flagged truncated
    sync_max_pages: int = 5  # POST /v1/ocr rejects longer documents; use POST /v1/jobs instead
    max_image_side: int = 2400
    pdf_render_dpi: int = 200
    # Use the PDF's embedded text instead of OCR when a page has a usable text layer
    pdf_text_layer: bool = True
    min_text_layer_chars: int = 30
    max_upload_mb: int = 50

    # Pull structured invoice fields (vendor, buyer, items, taxes, approval ...) out of the text
    extract_invoice: bool = True
    # PANs of the buying group: GSTINs containing them are the buyer's. Defaults: Amara Raja Energy &
    # Mobility, Amara Raja Infra, Amara Raja Electronics. JSON list, e.g. '["AABCA9264E"]'
    own_pans: list[str] = ["AABCA9264E", "AAHCA1764F", "AAECA2774J"]

    # Origins allowed to call the API from a browser, as a JSON list (e.g. '["https://ocr.example.com"]')
    cors_origins: list[str] = ["*"]

    # Background jobs (in-memory store for local execution)
    job_store: str = "memory"
    job_ttl_seconds: int = 3600  # in-memory store retention
    max_pending_jobs: int = 50
    job_max_attempts: int = 3
    job_lease_seconds: int = 300
    job_poll_seconds: int = 15

    @model_validator(mode="before")
    @classmethod
    def _blank_means_default(cls, values: dict) -> dict:
        """An app setting left empty means 'use the default', not an empty string.
        Also trims stray whitespace / Windows line endings."""
        if not isinstance(values, dict):
            return values
        out = {k: v.strip() if isinstance(v, str) else v for k, v in values.items()}
        return {k: v for k, v in out.items() if not (isinstance(v, str) and not v)}


@lru_cache
def get_settings() -> Settings:
    return Settings()
