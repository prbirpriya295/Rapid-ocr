"""Plain-language status messages returned by every endpoint."""
from __future__ import annotations

from .schemas import InvoiceFields, Validation

FIELD_LABELS = {"vendor_name": "vendor name", "invoice_number": "invoice number",
                "invoice_date": "invoice date", "total_amount": "total amount"}


def _pages(n: int) -> str:
    return f"{n} page" + ("" if n == 1 else "s")


def _join(names: list[str]) -> str:
    names = [FIELD_LABELS.get(n, n.replace("_", " ")) for n in names]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def extracted(pages: int, fields: InvoiceFields | None, additional: dict | None, missing: list[str] | None,
              validation: Validation | None, remapped: bool = False) -> str:
    """Message for a finished document."""
    core = sum(1 for v in (fields.model_dump().values() if fields else ()) if v not in (None, ""))
    found = core + len(additional or {})
    head = ("Fields re-mapped from the stored OCR text" if remapped
            else f"Document extracted successfully: {_pages(pages)} read")
    msg = f"{head}, {found} field{'' if found == 1 else 's'} extracted."
    if missing:
        msg += f" Not found: {_join(missing)}; please enter manually."
    elif validation and validation.status == "fail":
        msg += " Some checks failed; please review the issues listed under validation."
    elif validation and validation.status == "warn":
        msg += " Some checks need a look; see validation."
    else:
        msg += " All required fields found."
    return msg


def job_message(rec: dict, remapped: bool = False) -> str:
    status, stage = rec.get("status"), rec.get("stage")
    total = rec.get("pages_to_process") or rec.get("pages_total") or 0
    if status == "queued":
        if rec.get("attempts"):
            return (f"Processing was interrupted (service restart); the document is queued again "
                    f"(attempt {rec['attempts'] + 1}).")
        return f"Document received ({_pages(rec.get('pages_total', 0))}) and queued for processing."
    if status == "processing":
        if stage == "saving":
            return f"All {_pages(total)} read; saving the results."
        return f"Reading the document: {rec.get('pages_done', 0)} of {_pages(total)} done."
    if status == "done":
        v = rec.get("validation")
        return extracted(rec.get("page_count") or total, InvoiceFields(**rec["fields"]) if rec.get("fields") else None,
                         rec.get("additional_fields"), rec.get("missing_required_fields"),
                         Validation(**v) if v else None, remapped)
    if status == "failed":
        return f"Processing failed: {rec.get('error') or 'unknown error'}."
    return str(status)
