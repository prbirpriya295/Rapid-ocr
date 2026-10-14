from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class OcrLine(BaseModel):
    text: str
    # OCR confidence 0-1; 1.0 for text taken from a PDF text layer
    confidence: float | None = Field(default=None, ge=0, le=1)
    bbox: list[list[float]] | None = None  # 4 points [[x,y],...] in page pixels (at the render DPI for PDFs)
    page: int = 0


class OcrTable(BaseModel):
    page: int
    html: str | None = None


class PageResult(BaseModel):
    page: int
    source: str  # "pdf-text-layer", "pdf-text-layer+ocr", "rapidocr" or "fake"
    text: str
    average_confidence: float | None
    needs_review: bool
    seconds: float | None = None  # time spent on this page


class Party(BaseModel):
    name: str | None = None
    address: str | None = None
    gstin: str | None = None
    pan: str | None = None
    cin: str | None = None
    state: str | None = None
    state_code: str | None = None
    email: str | None = None


class LineItem(BaseModel):
    sl_no: int | None = None
    description: str | None = None
    details: str | None = None  # sub-description lines under the item
    hsn_sac: str | None = None
    quantity: str | None = None
    rate: float | None = None
    per: str | None = None
    amount: float | None = None


class TaxLine(BaseModel):
    name: str  # CGST, SGST, UTGST, IGST, CESS, TCS
    rate: float | None = None  # percent
    amount: float


class EInvoice(BaseModel):
    irn: str | None = None
    ack_no: str | None = None
    ack_date: str | None = None


class Approval(BaseModel):
    status: str  # approved | requested
    approved_by: str | None = None
    approved_by_email: str | None = None
    approved_on: str | None = None
    requested_by: str | None = None
    subject: str | None = None


class CostSheetLine(BaseModel):
    sl_no: int | None = None
    account_code: str | None = None
    description: str
    amount: float | None = None


class CostSheet(BaseModel):
    title: str | None = None
    location: str | None = None
    month: str | None = None
    date: str | None = None
    lines: list[CostSheetLine] = []
    grand_total: float | None = None


class BillEntry(BaseModel):
    """One vendor bill listed on a bill submission form (several bills can be submitted together)."""
    invoice_number: str
    invoice_date: str | None = None
    amount: float | None = None


class SubmissionForm(BaseModel):
    """Bill submission cover form (e.g. Amara Raja FSSC 'NON PO' format): numbered label -> value rows."""
    title: str | None = None
    fields: dict[str, str] = {}  # every numbered row, e.g. {"vendor_code": "10007235", "location": "..."}
    prepared_by: str | None = None
    verified_by: str | None = None
    approved_by: list[str] = []


class Check(BaseModel):
    name: str
    status: str  # pass | warn | fail
    message: str


class InvoiceData(BaseModel):
    """Structured fields read from the document. Values are as printed; verify anything `checks` flags."""
    status: str = "pass"  # worst check result: pass | warn | fail
    document_type: str | None = None
    invoice_number: str | None = None  # several numbers are joined with ", "
    invoice_numbers: list[str] = []  # every invoice number found
    invoice_date: str | None = None  # as printed
    invoice_date_iso: str | None = None
    e_invoice: EInvoice | None = None
    vendor: Party | None = None
    buyer: Party | None = None
    consignee: Party | None = None
    references: dict[str, str] = {}  # delivery note, buyer's order no., dispatch details ...
    line_items: list[LineItem] = []
    taxes: list[TaxLine] = []
    round_off: float | None = None
    taxable_value: float | None = None
    total_tax: float | None = None
    grand_total: float | None = None
    currency: str = "INR"
    amount_in_words: str | None = None
    tax_amount_in_words: str | None = None
    remarks: str | None = None
    jurisdiction: str | None = None
    approval: Approval | None = None
    cost_sheet: CostSheet | None = None
    submission_form: SubmissionForm | None = None
    bills: list[BillEntry] = []  # per-bill breakdown when one submission covers several bills
    checks: list[Check] = []


class InvoiceFields(BaseModel):
    """The 5 core fields: always present in every response (null when not found). vendor_name,
    invoice_number, invoice_date and total_amount are required; any missing are listed in
    missing_required_fields. Everything else found in a document goes to additional_fields."""
    vendor_name: str | None = None
    invoice_number: str | None = None  # several invoice numbers are joined with ", "
    invoice_date: str | None = None  # YYYY-MM-DD when the printed date can be parsed, else as printed
    total_amount: float | None = None  # amount payable
    currency: str | None = None


class Validation(BaseModel):
    """Plain-language result of the automatic checks (GSTIN check digit, totals add up, amount in words ...)."""
    status: str = "pass"  # pass | warn | fail  (fail: something is wrong, warn: please look)
    issues: list[str] = []  # one sentence per failed / warning check; empty when everything passed
    checks_passed: int = 0
    checks_total: int = 0


class OcrResponse(BaseModel):
    status: str = "done"
    message: str = ""  # plain-language summary, e.g. "Document extracted successfully: ..."
    fields: InvoiceFields = InvoiceFields()  # the 5 core fields, always present
    # every other field found in THIS document (no fixed list, no nulls): GSTINs, taxes, codes, line items ...
    additional_fields: dict[str, Any] = {}
    missing_required_fields: list[str] = []  # required core fields not found
    validation: Validation = Validation()
    text: str
    lines: list[OcrLine]
    tables: list[OcrTable] = []
    pages: list[PageResult] = []
    average_confidence: float | None  # over scored lines only; None when no line carries a score
    page_count: int  # pages processed
    pages_total: int  # pages in the document
    truncated: bool  # True when pages_total > page_count (max_pages limit)
    needs_review: bool  # True when any page needs a human check
    engine: str
    processing_seconds: float | None = None  # whole document, OCR + field extraction
    invoice: InvoiceData | None = None  # structured invoice / approval / cost-sheet fields, when found


class JobView(BaseModel):
    job_id: str  # the process id: keep it and fetch the result with GET /v1/jobs/{job_id}
    status: str  # queued | processing | done | failed
    message: str = ""  # plain-language status, e.g. "Reading the document: 2 of 6 pages done."
    stage: str | None = None  # queued | ocr | saving | done | failed
    filename: str | None = None
    pages_total: int
    pages_to_process: int  # pages_total capped at OCR_MAX_PAGES
    pages_done: int
    attempts: int = 0  # >1 when the job was retried after the service stopped mid-run
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    processing_seconds: float | None = None
    error: str | None = None
    fields: InvoiceFields | None = None  # the 5 core fields, once done
    additional_fields: dict[str, Any] | None = None  # every other field found in this document
    missing_required_fields: list[str] | None = None
    validation: Validation | None = None
    storage: str = "memory"
    result: OcrResponse | None = None  # full OCR + invoice detail (omitted with ?include_ocr=false)
 