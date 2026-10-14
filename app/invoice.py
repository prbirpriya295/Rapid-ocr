"""Structured fields from Indian GST invoices (Tally-style layout) plus the approval e-mail and cost
sheet that usually travel with them.

Works on OCR boxes when they have positions (PP-OCR, PDF text layer) and falls back to plain text rows
(pasted text). Every value is "as read"; `checks` cross-validates them (GSTIN check
digit, totals, amount in words ...) so misreads surface instead of flowing silently into accounting.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from datetime import date

from .engine import group_rows
from .schemas import (
    Approval, BillEntry, Check, CostSheet, CostSheetLine, EInvoice, InvoiceData, InvoiceFields, LineItem, OcrLine,
    PageResult, Party, SubmissionForm, TaxLine, Validation,
)

# ---------------------------------------------------------------------------------------------- rows

GEOMETRY_SOURCES = {"rapidocr", "paddleocr", "pdf-text-layer", "pdf-text-layer+ocr", "fake"}  # paddleocr: older jobs


@dataclass
class Cell:
    text: str
    x0: float
    x1: float
    y0: float
    y1: float


@dataclass
class Row:
    page: int
    cells: list[Cell] = field(default_factory=list)  # empty for text-only rows
    plain: str = ""

    @property
    def text(self) -> str:
        return " ".join(c.text for c in self.cells) if self.cells else self.plain


def build_rows(pages: list[PageResult], lines: list[OcrLine]) -> list[Row]:
    rows: list[Row] = []
    for page in pages:
        page_lines = [l for l in lines if l.page == page.page]
        if page.source in GEOMETRY_SOURCES and page_lines and all(l.bbox for l in page_lines):
            for group in group_rows(page_lines):
                cells = []
                for l in group:
                    xs = [p[0] for p in l.bbox]  # type: ignore[union-attr]
                    ys = [p[1] for p in l.bbox]  # type: ignore[union-attr]
                    cells.append(Cell(l.text, min(xs), max(xs), min(ys), max(ys)))
                rows.append(Row(page.page, cells))
        else:
            rows.extend(Row(page.page, plain=t) for t in page.text.splitlines() if t.strip())
    return rows


def rows_from_text(text: str) -> list[Row]:
    return [Row(0, plain=t) for t in text.splitlines() if t.strip()]


# ------------------------------------------------------------------------------------------- parsing

GSTIN_RE = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][0-9A-Z]Z[0-9A-Z]\b")
PAN_RE = re.compile(r"\bPAN\s*(?:No\.?)?\s*[:\-]?\s*([A-Z]{5}\d{4}[A-Z])\b", re.I)
CIN_RE = re.compile(r"\bCIN\s*(?:No\.?)?\s*[:\-]?\s*([LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})\b", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
STATE_RE = re.compile(r"state\s*name\s*:?\s*([A-Za-z][A-Za-z &.]*?)\s*,\s*code\s*:?\s*(\d{1,2})\b", re.I)
# Money with 2 decimals; tolerates OCR dots as thousands separators ("1.446.57") and "(-)0.14".
AMOUNT_RE = re.compile(r"(\(-\)|-)?\s*₹?\s*((?:\d{1,3}(?:[.,]\d{2,3})+|\d+)\.\d{2})\b")
PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
# 27-Aug-24, 27/08/2024, 03.08.2024, 12 August 2024. Same separator throughout, and not the tail of a
# longer token, so "SL-PU-476-24-25 27-Aug-24" yields only 27-Aug-24.
DATE_RE = re.compile(
    r"(?<![\w-])(\d{1,2}([-/.])(?:[A-Za-z]{3,9}|\d{1,2})\2\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9},?\s+\d{2,4})(?![\w-])")
WORDS_RE = re.compile(r"((?:INR|Rs\.?|Rupees)\b.*?\bonly)\b", re.I)


def to_number(s: str) -> float:
    s = s.replace("₹", "").replace(" ", "")
    return float(re.sub(r"[.,](?=.*[.])", "", s).replace(",", ""))  # keep only the last "." as decimal


def amounts(text: str) -> list[float]:
    return [-to_number(m.group(2)) if m.group(1) else to_number(m.group(2)) for m in AMOUNT_RE.finditer(text)]


# Written dates: "5th of October, 2026", "January 25, 2025" (returned normalised as "5 October 2026").
_MONTH_NAMES = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
_DATE_DMY_WORDS = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_NAMES}),?\s+(\d{{4}})\b", re.I)
_DATE_MDY_WORDS = re.compile(rf"\b({_MONTH_NAMES})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)


def find_dates(text: str) -> list[str]:
    found = [(m.start(), m.group(1)) for m in DATE_RE.finditer(text)]
    found += [(m.start(), f"{m.group(1)} {m.group(2)} {m.group(3)}") for m in _DATE_DMY_WORDS.finditer(text)]
    found += [(m.start(), f"{m.group(2)} {m.group(1)} {m.group(3)}") for m in _DATE_MDY_WORDS.finditer(text)]
    seen, out = set(), []
    for _, d in sorted(found):
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def last_amount(text: str) -> float | None:
    found = amounts(text)
    return found[-1] if found else None


MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def iso_date(s: str | None) -> str | None:
    if not s:
        return None
    parts = re.split(r"[-/. ,]+", s.strip())
    if len(parts) != 3:
        return None
    d, m, y = parts
    try:
        month = int(m) if m.isdigit() else MONTHS[m[:3].lower()]
        year = int(y) + (2000 if len(y) == 2 else 0)
        return date(year, month, int(d)).isoformat()
    except (KeyError, ValueError):
        return None


_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i > 1}
_SCALES = {"thousand": 1_000, "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000,
           "crore": 10_000_000, "crores": 10_000_000, "million": 1_000_000}
_NUM_WORD_RE = re.compile("|".join(sorted([*_UNITS, *_TENS, *_SCALES, "hundred"], key=len, reverse=True)))


def _words_value(s: str) -> int | None:
    # Finds number words even when OCR glues them together ("EighteenThousand", "SixtySix").
    tokens = _NUM_WORD_RE.findall(s.lower())
    if not tokens:
        return None
    total = current = 0
    for t in tokens:
        if t in _UNITS:
            current += _UNITS[t]
        elif t in _TENS:
            current += _TENS[t]
        elif t == "hundred":
            current = (current or 1) * 100
        else:
            total += (current or 1) * _SCALES[t]
            current = 0
    return total + current


def words_to_amount(s: str | None) -> float | None:
    """'INR Two Thousand Eight Hundred Ninety Three and Fourteen paise Only' -> 2893.14"""
    if not s:
        return None
    low = s.lower()
    paise = 0
    if "paise" in low:
        cut = low.rfind(" and ", 0, low.find("paise"))
        if cut >= 0:
            paise = _words_value(low[cut + 5:low.find("paise")]) or 0
            low = low[:cut]
    rupees = _words_value(low)
    return None if rupees is None else round(rupees + paise / 100, 2)


GST_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def gstin_valid(g: str) -> bool:
    total = 0
    for i, ch in enumerate(g[:14]):
        v = GST_CHARS.index(ch) * (2 if i % 2 else 1)
        total += v // 36 + v % 36
    return GST_CHARS[(36 - total % 36) % 36] == g[14]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


# ------------------------------------------------------------------------------- header (right column)

# Tally prints these labels in the right-hand column with the value directly below each one.
HEADER_LABELS = {
    "Invoice No.": "invoice_number", "Dated": "date", "Delivery Note": "delivery_note",
    "Mode/Terms of Payment": "payment_terms", "Reference No. & Date.": "reference",
    "Other References": "other_references", "Buyer's Order No.": "buyer_order_number",
    "Dispatch Doc No.": "dispatch_doc_number", "Delivery Note Date": "delivery_note_date",
    "Dispatched through": "dispatched_through", "Destination": "destination",
    "Terms of Delivery": "terms_of_delivery", "Bill of Lading/LR-RR No.": "lr_number",
    "Motor Vehicle No.": "vehicle_number", "e-Way Bill No.": "eway_bill_number",
    "Supplier's Ref.": "supplier_reference",
}
_LABEL_NORMS = {_norm(k): v for k, v in HEADER_LABELS.items()}


def _label_key(text: str) -> str | None:
    """Fuzzy label match: OCR reads 'Dated' as 'Daled', 'Date' as 'Dale' ..."""
    n = _norm(text)
    if not n or re.search(r"\d", text):  # labels carry no digits; "Invoice No: X/123" is a value row
        return None
    best, score = None, 0.0
    for norm, key in _LABEL_NORMS.items():
        r = difflib.SequenceMatcher(None, n, norm).ratio()
        if r > score:
            best, score = key, r
    return best if score >= 0.78 else None


def header_fields(rows: list[Row]) -> tuple[dict[str, str], dict[int, float]]:
    """Values under the right-column labels, plus that column's left edge per page (to split the page)."""
    found: dict[str, str] = {}
    label_x0: dict[int, list[float]] = {}
    for i, row in enumerate(rows):
        prev_key = None
        for cell in row.cells:
            key = _label_key(cell.text)
            if not key:
                prev_key = None
                continue
            label_x0.setdefault(row.page, []).append(cell.x0)
            if key == "date":  # "Dated" belongs to the label before it on the same row
                key = {"invoice_number": "invoice_date", "buyer_order_number": "buyer_order_date"}.get(
                    prev_key or "", f"{prev_key}_date" if prev_key else "date")
            prev_key = key if key != "invoice_date" else "invoice_number"
            value = _value_below(rows, i, cell)
            if key in ("invoice_number", "buyer_order_number") and value and not re.search(r"\d", value):
                value = None  # numbers carry digits; a word here is a misread label
            if value and key not in found:
                found[key] = value
    if "invoice_number" not in found:
        _header_from_text(rows, found)
    # Column split per page: labels on other pages (supporting documents) must not move it.
    return found, {page: min(xs) for page, xs in label_x0.items() if len(xs) >= 3}


def _value_below(rows: list[Row], i: int, label: Cell) -> str | None:
    height = max(label.y1 - label.y0, 8)
    for row in rows[i + 1:i + 3]:
        if row.page != rows[i].page:
            break
        for c in row.cells:
            if c.y0 - label.y1 > 1.8 * height:
                return None
            if c.x0 < label.x1 + 30 and c.x1 > label.x0 - 10:
                head = c.text.split(":")[0]
                if _label_key(c.text) or _label_key(head) or (
                        re.fullmatch(r"[A-Za-z .&/'-]+", head.strip()) and ":" in c.text
                        or re.fullmatch(r"[A-Za-z .&/'-]*\b(name|date|no|number|code|order|transporter|vehicle)\b\.?",
                                        c.text.strip(" :"), re.I)):
                    return None  # the next label, not a value ("Invoice Date", "Transporter Name :")
                return c.text.strip(" :")
    return None


def _header_from_text(rows: list[Row], found: dict[str, str]) -> None:
    """No positions: value sits on the same row ('Invoice No: X') or merged into the next row."""
    for i, row in enumerate(rows):
        m = re.search(r"invoice\s*no\.?\s*[:\-]?\s*(.*)", row.text, re.I)
        if not m:
            # "Invoice 241037002567": bare label with a long number right after it on the same line
            m2 = re.search(r"\binvoice\s*[:\-]?\s+([A-Z0-9/\-]*\d{5,}[A-Z0-9/\-]*)\b", row.text, re.I)
            if m2 and m2.group(1) not in find_dates(row.text):
                found.setdefault("invoice_number", m2.group(1))
                break
            continue
        candidates = [m.group(1)] + ([rows[i + 1].text] if i + 1 < len(rows) else [])
        for text in candidates:
            dates = find_dates(text)
            if "invoice_number" not in found:
                for tok in text.split():
                    if re.search(r"\d", tok) and re.search(r"[-/]", tok) and tok not in dates \
                            and not EMAIL_RE.search(tok) and len(tok) >= 4:
                        found["invoice_number"] = tok.strip(".,:")
                        break
            if dates:
                found.setdefault("invoice_date", dates[-1])
        if "invoice_number" in found:
            break
    if "invoice_date" not in found:  # "Date: 28/09/2026" anywhere
        m = re.search(r"\b(?:invoice\s*)?date[d]?\s*[:\-]\s*(.{6,20})", "\n".join(r.text for r in rows), re.I)
        if m and find_dates(m.group(1)):
            found["invoice_date"] = find_dates(m.group(1))[0]


# ---------------------------------------------------------------------------- parties (left column)

_PARTY_LABEL_RE = re.compile(
    r"^\s*(gst|gstin|pan|state|cin|e-?mail|ph|phone|tel|mob|mobile|contact|code|fssai|udyam|msme|website)\b", re.I)
# Header lines that are never a company name, OCR typos included ("Inveice No", "Dale:").
_NOT_NAME_RE = re.compile(r"\binv\w{0,5}\s*no\b|\b(date[d]?|dale)\s*[:.]|\bpage\s+\d", re.I)
_SKIP_RE = re.compile(r"^\s*(irn|ack|tax invoice|e-?invoice|original|duplicate|triplicate)\b|^[0-9a-f-]{16,}$", re.I)
_TEXT_LABEL_STRIP = re.compile(
    "|".join(re.escape(k).replace(r"\ ", r"\s*") for k in sorted(HEADER_LABELS, key=len, reverse=True))
    + r"|\bDa[lt]ed?\b|\bOlher References\b|\bDale\b", re.I)


def left_column(rows: list[Row], split_x: dict[int, float], header: dict[str, str]) -> list[str]:
    """Text of the left (party) column. With no right-hand label column the layout is single-column,
    so whole rows are used, minus any header labels and values."""
    out = []
    for row in rows:
        if row.cells and split_x.get(row.page):
            text = " ".join(c.text for c in row.cells if c.x1 <= split_x[row.page] - 10)
        else:
            text = _TEXT_LABEL_STRIP.sub(" ", row.text)
            for v in (header.get("invoice_number"), header.get("invoice_date")):
                if v:
                    text = text.replace(v, " ")
        text = " ".join(text.split())
        if text:
            out.append(text)
    return out


def parties(left: list[str]) -> tuple[Party | None, Party | None, Party | None]:
    """Split the left column into seller / consignee / buyer blocks (Tally order)."""
    sections: dict[str, list[str]] = {"vendor": [], "consignee": [], "buyer": []}
    current = "vendor"
    for text in left:
        if _TABLE_HEADER_RE.search(text) or AMOUNT_RE.search(text):
            break  # item table reached (party blocks never hold money amounts)
        header = (re.match(r"^consignee\s*(\(?\s*ship\s*to\s*\)?)?\s*:?", text, re.I) and "consignee") or \
                 (re.match(r"^(buyer\s*\(?\s*bill\s*to\s*\)?|bill\s*to)\s*:?", text, re.I) and "buyer")
        if header:
            current = header
            text = text[re.match(r"^\S+(\s*\(?\s*(ship|bill)\s*to\s*\)?)?\s*(to\b)?\s*:?", text, re.I).end():].strip()
            if not text:
                continue  # "Bill To GSTIN: X" keeps the GSTIN in the buyer block
        sections[current].append(text)
    return tuple(_party(sections[k]) for k in ("vendor", "consignee", "buyer"))  # type: ignore[return-value]


_COMPANY_SUFFIX_RE = re.compile(r"\b(pvt|private|ltd|limited|llp|inc|corporation)\b\.?", re.I)


def _party(lines: list[str]) -> Party | None:
    lines = [l for l in lines if not _SKIP_RE.search(l)]
    if not lines:
        return None
    blob = "\n".join(lines)
    name, address = None, []
    for l in lines:
        if _PARTY_LABEL_RE.match(l) or EMAIL_RE.search(l):
            if name:
                break
            continue
        if name is None:
            if _NOT_NAME_RE.search(l):
                continue
            name = l.strip(" ,.")
        else:
            address.append(l.strip(" ,"))
    # A logo read as text ("BMBHOOMI") often sits above the real name: prefer a nearby line with a
    # company suffix ("BHOOMI MODULAR SYSTEMS PRIVATE LIMITED").
    if name and not _COMPANY_SUFFIX_RE.search(name):
        nearby = [l.strip(" ,.") for l in lines[:4] if not _PARTY_LABEL_RE.match(l) and not _NOT_NAME_RE.search(l)]
        named = next((l for l in nearby if _COMPANY_SUFFIX_RE.search(l) and not re.search(r"\d{3}", l)), None)
        if named and named != name:
            if name in address:
                address.remove(name)
            address = [a for a in address if a.strip(" ,.") != named]
            name = named
    gstins = GSTIN_RE.findall(blob.upper())
    state = STATE_RE.search(blob)
    pan, cin, email = PAN_RE.search(blob), CIN_RE.search(blob), EMAIL_RE.search(blob)
    party = Party(
        name=name, address=", ".join(address).rstrip(".,") or None,
        gstin=gstins[0] if gstins else None,
        pan=pan.group(1).upper() if pan else None, cin=cin.group(1).upper() if cin else None,
        state=state.group(1).strip() if state else None, state_code=state.group(2).zfill(2) if state else None,
        email=email.group(0) if email else None,
    )
    return party if (party.name or party.gstin) else None


# -------------------------------------------------------------------------------------- line items

_CHARGE_RE = re.compile(r"\b(cgst|sgst|utgst|igst|cess|round\s*off|tcs)\b", re.I)
# Item table header: "Sl No ... Particulars ... Amount" (Tally) or "Item / Description ... Qty / Amount".
_TABLE_HEADER_RE = re.compile(
    r"^\s*(sl|si|s\.?\s*no)\b|(particulars|description|\bitems?\b|goods|services?)\b.*\b(amount|qty|quantity|rate|value)\b",
    re.I)
# Row with the amount payable. Plain "Total" counts too (Tally prints "Total ₹18,966.00").
GRAND_TOTAL_RE = re.compile(
    r"^\s*(grand\s*total|invoice\s*total|total\s*amount|amount\s*payable|net\s*payable|net\s*amount|total)\b", re.I)
_SUBTOTAL_RE = re.compile(r"^\s*(sub\s*-?\s*total|taxable\s*(value|amount))\b", re.I)
_COLUMNS = {"hsn": r"hsn|sac", "quantity": r"quantity|qty", "rate": r"rate", "per": r"\bper\b", "amount": r"amount"}


def _column_centres(header: Row) -> dict[str, float]:
    """x-centre of each column; a box holding several headers ('HSN/SAC Quantity') is split by characters."""
    centres: dict[str, float] = {}
    for c in header.cells:
        for key, pat in _COLUMNS.items():
            m = re.search(pat, c.text, re.I)
            if m and key not in centres:
                n = max(len(c.text), 1)
                mid_char = (m.start() + m.end()) / 2
                centres[key] = c.x0 + (c.x1 - c.x0) * mid_char / n
    return centres


def line_items(rows: list[Row]) -> tuple[list[LineItem], list[TaxLine], float | None, int]:
    """Items, tax/round-off lines, round-off, and the index of the row after the table."""
    start = next((i for i, r in enumerate(rows) if _TABLE_HEADER_RE.search(r.text)
                  and re.search(r"amount|qty|quantity|value", r.text, re.I)), None)
    if start is None:
        return [], [], None, 0
    centres = _column_centres(rows[start]) if rows[start].cells else {}
    items: list[LineItem] = []
    taxes: list[TaxLine] = []
    round_off = None
    charges_started = False
    items_open = True  # closed by a "Sub Total" / "Taxable Value" row; taxes may still follow
    end = len(rows)
    for i in range(start + 1, len(rows)):
        row, text = rows[i], rows[i].text
        if GRAND_TOTAL_RE.match(text) or re.match(r"^\s*amount\s*chargeable\b", text, re.I):
            end = i
            break
        if _SUBTOTAL_RE.match(text):
            items_open = False
            continue
        if text.strip(" .").lower() == "no":
            continue
        charge = _CHARGE_RE.search(text)
        amount = last_amount(text)
        if charge and amount is not None:
            charges_started = True
            name = charge.group(1).upper() if "round" not in charge.group(1).lower() else "Round Off"
            if name == "Round Off":
                round_off = -abs(amount) if re.search(r"\(-\)|\bless\b|^-", text, re.I) else amount
                continue
            pct = PERCENT_RE.search(text)
            taxes.append(TaxLine(name=name, rate=float(pct.group(1)) if pct else None, amount=amount))
            continue
        if not items_open:
            continue
        item = _item(row, centres) if amount is not None else None
        if item:
            items.append(item)
        elif items and not charges_started and amount is None and not re.search(r"\d{6,}", text):
            prev = items[-1]  # continuation line, e.g. the italic sub-description under an item
            prev.details = f"{prev.details} {text.strip()}" if prev.details else text.strip()
    return items, taxes, round_off, end


def _item(row: Row, centres: dict[str, float]) -> LineItem | None:
    text = row.text
    m = re.match(r"^\s*(\d{1,3})[.)]?\s+(.*)$", text)
    sl = int(m.group(1)) if m else None
    hsn = re.search(r"\b(\d{4}|\d{6}|\d{8})\b(?![.,]\d)", m.group(2) if m else text)
    if sl is None and not hsn and not re.match(r"^\s*[A-Za-z]", text):
        return None  # needs a serial no., an HSN code, or at least a description first
    item = LineItem(sl_no=sl, amount=last_amount(text), hsn_sac=hsn.group(1) if hsn else None)
    if row.cells and centres:
        desc = []
        cols = {k: v for k, v in centres.items() if k != "amount"}  # the last box is the amount
        for c in row.cells[:-1] if len(row.cells) > 1 else row.cells:
            mid = (c.x0 + c.x1) / 2
            col = min(cols, key=lambda k: abs(cols[k] - mid)) if cols else "amount"
            if c.x1 < min(centres.values()) - 15:  # left of the first numeric column: Sl No / description
                if not (sl is not None and c.text.strip(" .)") == str(sl)):
                    desc.append(c.text)
            elif col == "quantity":
                item.quantity = c.text
            elif col == "rate":
                item.rate = last_amount(c.text) or _plain_number(c.text)
            elif col == "per":
                item.per = c.text
        item.description = " ".join(desc).strip() or None
    if not item.description:  # no positions, or OCR merged the whole row into one box
        body = m.group(2) if m else text
        first_number = re.search(r"\s\d", body)
        cut = hsn.start() if hsn else (first_number.start() if first_number else len(body))
        item.description = body[:cut].strip() or None
    return item


def _plain_number(s: str) -> float | None:
    m = re.search(r"\d[\d,]*(?:\.\d+)?", s)
    return to_number(m.group(0)) if m else None


# ---------------------------------------------------------------------------------- totals & footer

def totals(rows: list[Row], end: int) -> dict:
    out: dict = {}
    words_row = None
    for i in range(end, len(rows)):
        t = rows[i].text
        if "grand_total" not in out and GRAND_TOTAL_RE.match(t) and last_amount(t) is not None:
            out["grand_total"] = last_amount(t)
        if re.search(r"amount\s*chargeable", t, re.I):
            words_row = i
            break
    if words_row is None:  # no "Amount Chargeable" heading: any "Rupees/INR ... Only" line
        for t in (r.text for r in rows[end:]):
            m = WORDS_RE.search(t)
            if m:
                out["amount_in_words"] = m.group(1)
                break
    if words_row is not None:
        for t in (r.text for r in rows[words_row:words_row + 3]):
            m = WORDS_RE.search(t)
            if m:
                out["amount_in_words"] = m.group(1)
                break
    # HSN/SAC tax summary: "Total <taxable> <tax...> <total tax>"
    start = words_row if words_row is not None else end
    summary = next((i for i in range(start, len(rows)) if re.search(r"taxable", rows[i].text, re.I)), None)
    if summary is not None:
        rates = []
        for r in rows[summary:summary + 12]:
            if re.match(r"^\s*total\b", r.text, re.I):
                vals = amounts(r.text)
                if len(vals) >= 2:
                    out["taxable_value"], out["total_tax"] = vals[0], vals[-1]
                break
            rates += [float(p) for p in PERCENT_RE.findall(r.text)]
        out["summary_rates"] = rates
    if "taxable_value" not in out:  # simple layouts print "Taxable Value 5,500.00" / "Sub Total ..."
        row = next((r for r in rows if _SUBTOTAL_RE.match(r.text) and last_amount(r.text) is not None), None)
        if row:
            out["taxable_value"] = last_amount(row.text)
    doc = "\n".join(r.text for r in rows)
    m = re.search(r"tax\s*am\w*nt\s*\(?in\s*words\)?\s*:?\s*((?:INR|Rs\.?|Rupees)\b.*?\bonly)\b", doc, re.I)
    if m:
        out["tax_amount_in_words"] = m.group(1)
    return out


def footer(rows: list[Row], split_x: dict[int, float]) -> dict:
    out: dict = {}
    for i, row in enumerate(rows):
        if re.match(r"^\s*remarks\b", row.text, re.I):
            for r in rows[i:i + 2]:
                if r.cells and split_x.get(r.page):
                    text = " ".join(c.text for c in r.cells if c.x1 <= split_x[r.page] - 10)
                else:
                    text = re.sub(r"\s+for\s+\S.*$", "", r.plain) if r is rows[i] else r.plain
                text = re.sub(r"^\s*remarks\s*:?", "", text, flags=re.I).strip(" .")
                if text:
                    out["remarks"] = text
                    break
            break
    doc = "\n".join(r.text for r in rows)
    m = re.search(r"subject\s+to\s+([a-z ]+?)\s+jurisdiction", doc, re.I)
    if m:
        out["jurisdiction"] = m.group(1).title()
    m = re.search(r"company'?s\s*pan\s*:?\s*([A-Z]{5}\d{4}[A-Z])", doc, re.I)
    if m:
        out["company_pan"] = m.group(1).upper()
    for r in rows:
        if re.search(r"invoice\s*no", r.text, re.I):
            continue
        m = re.search(r"\b(tax invoice|bill of supply|credit note|debit note|proforma invoice)\b", r.text, re.I)
        if m:
            out["document_type"] = m.group(1).title()
            break
    return out


def e_invoice(rows: list[Row]) -> EInvoice | None:
    irn = ack_no = ack_date = None
    for i, row in enumerate(rows):
        t = row.text
        m = re.search(r"\bIRN\b\s*:?\s*([0-9a-f]{16,})", t, re.I)
        if m and not irn:
            irn = m.group(1)
            if len(irn) < 64 and i + 1 < len(rows):  # printed IRN wraps onto a second line
                nxt = re.match(r"^\s*-?([0-9a-f]{8,})\b", rows[i + 1].text, re.I)
                if nxt:
                    irn += nxt.group(1)
        m = re.search(r"ack\.?\s*no\.?\s*:?\s*(\d{10,})", t, re.I)
        if m and not ack_no:
            ack_no = m.group(1)
        m = re.search(r"ack\.?\s*date\s*:?\s*(.*)", t, re.I)
        if m and not ack_date and find_dates(m.group(1)):
            ack_date = find_dates(m.group(1))[0]
    if not (irn or ack_no):
        return None
    return EInvoice(irn=irn.lower() if irn else None, ack_no=ack_no, ack_date=ack_date)


# ------------------------------------------------------------------------- approval e-mail & sheet

def approval(text: str) -> Approval | None:
    starts = [m.start() for m in re.finditer(r"(?mi)^\s*From:\s*", text)]
    if not starts:
        return None
    messages = []
    for a, b in zip(starts, starts[1:] + [len(text)]):
        chunk = text[a:b]
        frm = re.search(r"From:\s*(.+)", chunk)
        sent = re.search(r"Sent:\s*(.+)", chunk)
        subject = re.search(r"Subject:\s*(.+)", chunk)
        body = [l for l in chunk.splitlines()
                if l.strip() and not re.match(r"^\s*(from|sent|to|cc|bcc|subject|date):", l, re.I)]
        sender = (frm.group(1).strip() if frm else "")
        em = re.search(r"<([^>]+)>", sender)
        messages.append({
            "name": re.sub(r"\s*<[^>]*>.*", "", sender).strip() or None, "email": em.group(1) if em else None,
            "sent": sent.group(1).strip() if sent else None,
            "subject": re.sub(r"^\s*(re|fw|fwd):\s*", "", subject.group(1).strip(), flags=re.I) if subject else None,
            "body": body,
        })
    approved = next((m for m in messages if any(
        re.match(r"^\s*(approved|ok[, ]+approved|i approve)\b", l, re.I) and not re.search(r"\bby\b", l, re.I)
        for l in m["body"])), None)
    requested = next((m for m in messages if any(
        re.search(r"(provide|need|request|seek)\w*\s+(your\s+)?approval|for\s+approval", l, re.I)
        for l in m["body"])), None)
    if not (approved or requested):
        return None
    ref = approved or requested
    return Approval(
        status="approved" if approved else "requested",
        approved_by=approved["name"] if approved else None,
        approved_by_email=approved["email"] if approved else None,
        approved_on=approved["sent"] if approved else None,
        requested_by=requested["name"] if requested else None,
        subject=ref["subject"] or (requested or {}).get("subject"),
    )


def cost_sheet(lines: list[str]) -> CostSheet | None:
    anchor = next((i for i, l in enumerate(lines)
                   if re.search(r"bill\s+for\s+the\s+month|grand\s+total\s*\(rs", l, re.I)), None)
    if anchor is None:
        return None
    window = lines[max(0, anchor - 5):]
    joined = "\n".join(window)
    sheet = CostSheet(
        title=next((l.strip() for l in window if re.search(r"bill\s+for\s+the\s+month", l, re.I)), None))
    if m := re.search(r"(?mi)^\s*month\s+(.+)$", joined):
        sheet.month = m.group(1).strip()
    if m := re.search(r"(?mi)^\s*date\s+(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", joined):
        sheet.date = m.group(1)
    if m := re.search(r"(?mi)^\s*DC\s+([A-Za-z]+)", joined):
        sheet.location = m.group(1)
    for i, l in enumerate(window):
        m = re.match(r"^\s*(\d{1,3})\s+(\d{5,8})\s+([A-Za-z][A-Za-z &/.'-]*?)\s*(\d[\d,]*(?:\.\d+)?)?\s*$", l)
        if m:
            sheet.lines.append(CostSheetLine(
                sl_no=int(m.group(1)), account_code=m.group(2), description=m.group(3).strip(),
                amount=to_number(m.group(4)) if m.group(4) else None))
        if re.search(r"grand\s+total", l, re.I):
            nums = [to_number(n) for t in window[i:i + 2] for n in re.findall(r"\d[\d,]*(?:\.\d+)?", t)]
            sheet.grand_total = max(nums) if nums else None
            break
    return sheet if (sheet.lines or sheet.grand_total) else None


# ---------------------------------------------------------------------------------------- checks

def _close(a: float | None, b: float | None, tol: float) -> bool | None:
    return None if a is None or b is None else abs(a - b) <= tol


def _money(v: float) -> str:
    return f"{v:,.2f}"


def run_checks(inv: InvoiceData, doc_after_form: str = "") -> list[Check]:
    checks: list[Check] = []

    def add(name: str, ok: bool | None, good: str, bad: str, severity: str = "fail") -> None:
        if ok is not None:
            checks.append(Check(name=name, status="pass" if ok else severity, message=good if ok else bad))

    missing = missing_required(key_fields(inv))
    add("Required fields", not missing, "Vendor name, invoice number, invoice date and total amount were all found",
        f"Required field(s) not found: {', '.join(missing)}. Enter them manually")
    if inv.submission_form:
        _form_checks(inv, doc_after_form, add)

    for role, p in (("Vendor", inv.vendor), ("Buyer", inv.buyer)):
        if p and p.gstin:
            add(f"{role} GSTIN", gstin_valid(p.gstin), f"{p.gstin}: check digit valid",
                f"{p.gstin}: check digit INVALID, likely misread")
            if p.state_code:
                add(f"{role} state code", p.gstin[:2] == p.state_code,
                    f"State code {p.state_code} matches GSTIN",
                    f"State code {p.state_code} does not match GSTIN prefix {p.gstin[:2]}", "warn")
    v = inv.vendor
    if v and v.gstin and v.pan:
        add("Vendor PAN", v.gstin[2:12] == v.pan, f"PAN {v.pan} matches GSTIN",
            f"PAN {v.pan} does not match GSTIN {v.gstin}")

    tax = {t.name: t for t in inv.taxes}
    if inv.vendor and inv.buyer and inv.vendor.gstin and inv.buyer.gstin and tax:
        intra = inv.vendor.gstin[:2] == inv.buyer.gstin[:2]
        add("GST type", ("IGST" not in tax) if intra else ("IGST" in tax),
            "CGST + SGST for a same-state supply" if intra else "IGST for an inter-state supply",
            "Same-state supply should charge CGST + SGST, not IGST" if intra
            else "Inter-state supply should charge IGST")
    if "CGST" in tax and ("SGST" in tax or "UTGST" in tax):
        other = tax.get("SGST") or tax["UTGST"]
        add("CGST = SGST", _close(tax["CGST"].amount, other.amount, 0.01),
            f"CGST and {other.name} are equal ({_money(other.amount)})",
            f"CGST {_money(tax['CGST'].amount)} != {other.name} {_money(other.amount)}")
    if inv.taxable_value:
        for t in inv.taxes:
            if t.rate:
                expected = inv.taxable_value * t.rate / 100
                add(f"{t.name} rate", _close(expected, t.amount, 1.0),
                    f"{t.name} {_money(t.amount)} = {t.rate:g}% of {_money(inv.taxable_value)}",
                    f"{t.name} {_money(t.amount)} but {t.rate:g}% of {_money(inv.taxable_value)} "
                    f"is {_money(expected)}", "warn")
    if inv.grand_total is not None and inv.taxable_value is not None and inv.taxes:
        computed = inv.taxable_value + sum(t.amount for t in inv.taxes) + (inv.round_off or 0)
        add("Totals add up", _close(computed, inv.grand_total, 0.05),
            f"Taxable + tax + round-off = {_money(computed)} matches the grand total",
            f"Taxable + tax + round-off = {_money(computed)} but the grand total reads "
            f"{_money(inv.grand_total)}: an amount was likely misread")
    items_sum = sum(i.amount for i in inv.line_items if i.amount is not None)
    if inv.line_items and inv.taxable_value is not None:
        add("Line items sum", _close(items_sum, inv.taxable_value, 0.05),
            f"Line items add up to the taxable value ({_money(items_sum)})",
            f"Line items add up to {_money(items_sum)} but the taxable value is {_money(inv.taxable_value)}")
    add("Amount in words", _close(words_to_amount(inv.amount_in_words), inv.grand_total, 0.01),
        "Amount in words matches the grand total",
        f"Amount in words reads {_money(words_to_amount(inv.amount_in_words) or 0)} "
        f"but the grand total is {_money(inv.grand_total or 0)}")
    add("Tax in words", _close(words_to_amount(inv.tax_amount_in_words), inv.total_tax, 0.01),
        "Tax amount in words matches the total tax",
        f"Tax in words reads {_money(words_to_amount(inv.tax_amount_in_words) or 0)} "
        f"but the total tax is {_money(inv.total_tax or 0)}")
    if inv.e_invoice and inv.e_invoice.irn:
        add("IRN format", bool(re.fullmatch(r"[0-9a-f]{64}", inv.e_invoice.irn)), "IRN is 64 hex characters",
            f"IRN has {len(inv.e_invoice.irn)} characters (expected 64): check the printed IRN", "warn")
    if inv.approval:
        add("Approval", inv.approval.status == "approved",
            f"Approved by {inv.approval.approved_by or 'unknown'} on {inv.approval.approved_on or 'unknown date'}",
            "Approval was requested but no approval reply was found", "warn")
    if inv.cost_sheet:
        cs = inv.cost_sheet
        known = [l.amount for l in cs.lines if l.amount is not None]
        if known and cs.grand_total:
            add("Cost sheet total", _close(sum(known), cs.grand_total, 1.0),
                f"Cost sheet lines add up to its grand total ({_money(cs.grand_total)})",
                f"Cost sheet lines add up to {_money(sum(known))} but its grand total reads {_money(cs.grand_total)}",
                "warn")
        match = _matching_sheet_line(inv)
        if match and match.amount is not None and inv.taxable_value is not None:
            diff = abs(match.amount - inv.taxable_value)
            add("Invoice vs cost sheet", diff <= 0.01,
                f"Cost sheet '{match.description}' matches the invoice taxable value ({_money(inv.taxable_value)})",
                f"Cost sheet '{match.description}' is {_money(match.amount)} but the invoice taxable value is "
                f"{_money(inv.taxable_value)} (differs by {_money(diff)})", "warn")
    return checks


_STOP = {"charges", "charge", "services", "service", "invoice", "bill", "expenses", "with", "from", "month"}


def _matching_sheet_line(inv: InvoiceData) -> CostSheetLine | None:
    words = set()
    for text in [inv.remarks or ""] + [f"{i.description or ''} {i.details or ''}" for i in inv.line_items]:
        words |= {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOP}
    best, score = None, 0
    for line in inv.cost_sheet.lines if inv.cost_sheet else []:
        s = len(words & {w for w in re.findall(r"[a-z]{4,}", line.description.lower()) if w not in _STOP})
        if s > score:
            best, score = line, s
    return best


# ------------------------------------------------------------------------- bill submission form

# Numbered "label value" rows of a bill submission cover form (Amara Raja FSSC NON PO format and alike).
FORM_LABELS = {
    "account code": "account_code", "gl account code": "account_code", "cost centre code": "cost_centre_code",
    "cost center code": "cost_centre_code", "archival code": "archival_code", "company name": "company_name",
    "name of the company": "company_name", "company code": "company_code",
    "vendor invoice number & date": "invoice_number_date", "vendor invoice no & date": "invoice_number_date",
    "payment to be released": "payment_to_release", "advance request": "advance_request",
    "account description": "account_description", "gl account code description": "account_description",
    "cost centre / wbs code": "cost_centre_code", "account number": "bank_account_number", "bank name": "bank_name",
    "ifsc code": "ifsc_code", "branch name": "bank_branch", "spoc mail id": "spoc_contact",
    "location": "location", "vendor code": "vendor_code", "vendor name": "vendor_name",
    "item/service code": "item_service_code", "item/service description": "item_service_description",
    "sac/hsn code": "sac_hsn_code", "sac/hsn description": "sac_hsn_description",
    "invoice amount": "invoice_amount", "currency": "currency", "invoice creation date": "invoice_date",
    "invoice date": "invoice_date", "vendor invoice number": "invoice_number", "invoice number": "invoice_number",
    "po number": "po_number", "gstin": "gstin",
    "spoc employee name": "spoc_name", "spoc employee id": "spoc_id",
    "spoc mail id / contact number": "spoc_contact", "spoc department": "spoc_department",
    "spoc hod name": "spoc_hod_name", "spoc hod id": "spoc_hod_id", "spoc hod mail id": "spoc_hod_mail",
    "spoc lof name": "spoc_lof_name", "spoc lof id": "spoc_lof_id", "spoc lof mail id": "spoc_lof_mail",
    "towards": "towards",
}
_FORM_LABEL_RES = sorted(
    ((re.compile(r"^" + r"[\s/:.]*".join(re.escape(w) for w in re.split(r"[\s/]+", label)) + r"\b\s*[:\-]?\s*", re.I), key)
     for label, key in FORM_LABELS.items()),
    key=lambda p: -len(p[0].pattern))
_FORM_STOP_RE = re.compile(r"^\s*(prepared\s*by|for\s+finance|approved\s*by|verified\s*by)\b", re.I)
_BARCODE_RE = re.compile(r"\b\d{4}[A-Z]{3}\d{4}-\d{7}\b")  # scan barcode text that OCR merges into a row


@dataclass
class FormRead:
    form: SubmissionForm
    end: int  # index of the last row belonging to the form


_FORM_LABEL_NORMS = {_norm(k): v for k, v in FORM_LABELS.items()}
_MONEY_RE = re.compile(r"(?:\brs\.?|₹|\binr)\s*[\d,]+|\d{1,3}(?:,\d{2,3})+(?:\.\d{2})?\b|\d+\.\d{2}\b|\d\s*/-", re.I)
_AMOUNT_KEYS = {"invoice_amount", "payment_to_release"}
_CODE_KEYS = {"account_code", "cost_centre_code", "company_code", "vendor_code", "spoc_id", "spoc_hod_id",
              "spoc_lof_id", "bank_account_number"}
_TEXT_KEYS = {"vendor_name", "company_name", "location", "spoc_name", "spoc_hod_name", "spoc_lof_name",
              "spoc_department", "item_service_description", "towards", "bank_name", "account_description"}


def _form_label(text: str) -> tuple[str, str] | None:
    """(field key, text after the label) when a box starts with a form label; OCR typos tolerated."""
    t = re.sub(r"^\s*\d{1,2}[.)]?\s+", "", text).replace("*", " ").strip()
    for pat, key in _FORM_LABEL_RES:
        m = pat.match(t)
        if m:
            return key, t[m.end():]
    n = _norm(t)
    if 6 <= len(n) <= 40:
        best = max(_FORM_LABEL_NORMS, key=lambda k: difflib.SequenceMatcher(None, n, k).ratio())
        if difflib.SequenceMatcher(None, n, best).ratio() >= 0.86:
            return _FORM_LABEL_NORMS[best], ""
    return None


def _fits(key: str, text: str) -> bool:
    """Does this value look like what the field holds? Used to break ties between neighbouring rows."""
    if key in _AMOUNT_KEYS:
        return bool(re.search(r"\d", text)) or bool(re.search(r"rupee|lakh|thousand|hundred", text, re.I))
    if _MONEY_RE.search(text):
        return False  # "RS. 94,625/-", "2,01,337.50" only belong to an amount row
    if key == "currency":
        return bool(re.search(r"\bINR\b|rupee|indian|\bUSD\b|\bEUR\b|[$₹]", text, re.I))
    if key == "invoice_date":
        return bool(_LOOSE_DATE_RE.search(text))
    if key in ("invoice_number", "invoice_number_date"):
        return bool(re.search(r"\d", text)) or bool(re.match(r"^\s*nil\b", text, re.I))
    if key in _CODE_KEYS:
        return bool(re.search(r"\d", text)) or text.strip().lower() in ("attached", "na", "no")
    if key.endswith("_mail") or key == "spoc_contact":
        return "@" in text or bool(re.search(r"\d{6,}", text))
    if key in _TEXT_KEYS:
        return bool(re.search(r"[A-Za-z]{3}", text)) and not _LOOSE_DATE_RE.fullmatch(text.strip())
    return True


def _form_from_boxes(rows: list[Row]) -> FormRead | None:
    """Form read from OCR boxes. Values often sit at the top of their table cell or wrap onto two lines,
    so each value box goes to the label that best fits by position (values sit level with or above
    their label) and by type (an amount row wants digits, the currency row wants 'INR' ...)."""
    pages: dict[int, list[tuple[int, Cell]]] = {}
    for i, r in enumerate(rows):
        for c in r.cells:
            pages.setdefault(r.page, []).append((i, c))
    best_page, best_anchors = None, []
    for page, boxes in pages.items():
        anchors = [(c, *lab) for _, c in boxes if (lab := _form_label(c.text))]
        titled = any(re.search(r"bill\s*submission|\bfssc\b", c.text, re.I) for _, c in boxes)
        keys = len({a[1] for a in anchors})
        # Invoices also print "Invoice No", "Location" ... so a form needs its title or many form labels.
        if (keys >= 5 and titled or keys >= 8) and keys > len({a[1] for a in best_anchors}):
            best_page, best_anchors = page, anchors
    if not best_anchors:
        return None
    anchors = sorted(best_anchors, key=lambda a: a[0].y0)
    mids = [(a[0].y0 + a[0].y1) / 2 for a in anchors]
    gaps = sorted(b - a for a, b in zip(mids, mids[1:]) if b - a > 2)
    gap = gaps[len(gaps) // 2] if gaps else 20.0
    split = sorted(a[0].x1 for a in anchors)[len(anchors) // 2] - 5
    top, bottom = mids[0] - 1.5 * gap, mids[-1] + 1.5 * gap
    page_rows = [r for r in rows if r.page == best_page]
    stop_y = next(((min(c.y0 for c in r.cells)) for r in page_rows
                   if r.cells and re.search(r"for\s+finance", r.text, re.I) and min(c.y0 for c in r.cells) > mids[0]),
                  None)

    values: dict[int, list[tuple[float, float, str]]] = {i: [] for i in range(len(anchors))}
    for i, (c, key, rest) in enumerate(anchors):  # value text OCR merged into the label box
        if rest.strip(" :-"):
            values[i].append((mids[i], c.x0, rest))
    anchor_ids = {id(a[0]) for a in anchors}
    candidates = []
    for _, c in pages[best_page]:
        mid = (c.y0 + c.y1) / 2
        if id(c) in anchor_ids or c.x0 < split or not (top <= mid <= bottom) or (stop_y and mid >= stop_y):
            continue
        text = _BARCODE_RE.sub(" ", c.text).strip()
        if text and not _FORM_STOP_RE.match(text):
            candidates.append((mid, c.x0, text))

    def costs(mid: float, text: str, offset: float) -> list[float]:
        return [abs(mid - m - offset) / gap + (0 if _fits(anchors[i][1], text) else 1.5) for i, m in enumerate(mids)]

    # Each form prints values at a consistent height relative to its labels (above in some scans, below in
    # others). Measure that offset from unambiguous matches, then assign every value relative to it.
    shifts = []
    for mid, _, text in candidates:
        c0 = costs(mid, text, 0.0)
        i = min(range(len(anchors)), key=c0.__getitem__)
        if c0[i] < 0.6:
            shifts.append(mid - mids[i])
    offset = sorted(shifts)[len(shifts) // 2] if shifts else 0.0
    for mid, x0, text in candidates:
        cost = costs(mid, text, offset)
        i = min(range(len(anchors)), key=cost.__getitem__)
        if cost[i] <= 2:
            values[i].append((mid, x0, text))
    fields: dict[str, str] = {}
    for i, (_, key, _) in enumerate(anchors):
        text = " ".join(t for _, _, t in sorted(values[i]))
        if text and key not in fields:
            fields[key] = text
    form = SubmissionForm(
        title=next((r.text.strip() for r in page_rows if re.search(r"bill\s+submission", r.text, re.I)), None),
        fields=fields)
    _sign_off_from_boxes(form, page_rows)
    end = max(i for i, r in enumerate(rows) if r.page == best_page)
    return FormRead(form, end)


def _sign_off_from_boxes(form: SubmissionForm, page_rows: list[Row]) -> None:
    """'Approved by / Verified by / Prepared by' labels with the name printed under each, e.g. '(K.Srinivasulu)'."""
    roles = []
    for r in page_rows:
        for c in r.cells:
            m = re.search(r"(approved|verified|prepared)\s*(&\s*approved\s*)?b\w*", c.text, re.I)
            if m and not re.search(r"f\s*\.?\s*c\b|finance", c.text, re.I):
                roles.append((m.group(1).lower(), c))
    for r in page_rows:
        for c in r.cells:
            m = re.fullmatch(r"\(?\s*([A-Z][A-Za-z. ]{2,40}?)\s*\)?", c.text.strip())
            if not m or not c.text.strip().startswith("(") or re.search(r"approved|verified|prepared", c.text, re.I):
                continue
            below = [(abs((rc.x0 + rc.x1) / 2 - (c.x0 + c.x1) / 2), role) for role, rc in roles
                     if 0 < c.y0 - rc.y0 < 120]
            if not below:
                continue
            role = min(below)[1]
            name = m.group(1).strip()
            if role == "prepared" and not form.prepared_by:
                form.prepared_by = name
            elif role == "verified" and not form.verified_by:
                form.verified_by = name
            elif role == "approved" and name not in form.approved_by:
                form.approved_by.append(name)


def submission_form(rows: list[Row]) -> FormRead | None:
    if any(r.cells for r in rows):
        boxed = _form_from_boxes(rows)
        if boxed:
            return boxed
    lines = [r.text for r in rows]
    fields: dict[str, str] = {}
    current, numbered, end, cont = None, 0, -1, 0
    for i, line in enumerate(lines):
        if numbered and re.match(r"^\s*for\s+finance", line, re.I):
            break  # the finance-use box below is blank stamps, not part of the submission
        if _FORM_STOP_RE.match(line):
            current = None
            if numbered:
                end = max(end, i)
            continue
        m = re.match(r"^\s*(\d{1,2})[.)]?\s+(.+)$", line)
        key = rest = None
        if m:
            for pat, k in _FORM_LABEL_RES:
                lm = pat.match(m.group(2))
                if lm:
                    key, rest = k, m.group(2)[lm.end():]
                    break
        if key:
            numbered += 1
            current, cont, end = key, 0, i
            value = " ".join(_BARCODE_RE.sub(" ", rest).split()).strip(" :-")
            if value and key not in fields:
                fields[key] = value
            elif key not in fields:
                fields[key] = ""
        elif current and numbered and cont < 2 and not re.match(r"^\s*\d{1,2}\s+[A-Za-z]", line):
            extra = " ".join(_BARCODE_RE.sub(" ", line).split())
            if extra and len(extra) < 80:  # wrapped value: "...Month of Aug" + "2024"
                fields[current] = f"{fields[current]} {extra}".strip()
                cont += 1
                end = i
            else:
                current = None
    if numbered < 5:
        return None
    form = SubmissionForm(
        title=next((l.strip() for l in lines[:end + 1] if re.search(r"bill\s+submission", l, re.I)), None),
        fields={k: v for k, v in fields.items() if v})
    sign = "\n".join(lines[:end + 1])
    if m := re.search(r"prepared\s*by\s*:?\s*(.+?)(?=\s+verified\s*by|$)", sign, re.I | re.M):
        form.prepared_by = m.group(1).strip(" :-")
    if m := re.search(r"verified\s*by\s*:?\s*(.+?)(?=\s+approved\s*by|$)", sign, re.I | re.M):
        form.verified_by = m.group(1).strip(" :-")
    # "Approved by (AREML) - Mr. X" -> "Mr. X (AREML)"
    form.approved_by = [n.strip(" :-") + (f" {r}" if r else "") for r, n in re.findall(
        r"approved\s*by\s*(\([^)]*\))?\s*[-:]\s*(.+?)(?=\s+approved\s*by|$)", sign, re.I | re.M)]
    return FormRead(form, end)


def _form_numbers(value: str | None) -> list[str]:
    # Short numbers are real too ("39", "02"); this is the invoice-number field, so any token with a digit counts.
    return [t.strip(".:") for t in re.split(r"[,\s;]+", value or "") if re.search(r"\d", t)]


_LOOSE_DATE_RE = re.compile(r"(\d{1,2}([./-])(?:\d{1,2}|[A-Za-z]{3,9})\2\d{2,4})")


def _split_number_date(value: str) -> tuple[str | None, str | None]:
    """'F25RTM019690 & 29-08-2024', 'SAPR25000303043(17.08.2024)', '3993-21.12.2024', 'Nil/30-11-2024'."""
    dates = _LOOSE_DATE_RE.findall(value)
    date_text = dates[-1][0] if dates else None
    number = value.replace(date_text, " ") if date_text else value
    number = re.sub(r"(?i)\bdate\b|[&()/,]", " ", number).strip(" -.:")
    number = number.split()[0] if number.split() else None
    if number and number.lower() in ("nil", "na", "n/a", "-"):
        number = None
    return number, date_text


def _clean_form_value(v: str) -> str:
    return " ".join(v.replace("*", " ").split()).strip(" :-")


def _apply_form(inv: InvoiceData, read: FormRead, rows: list[Row]) -> None:
    """The cover form is what finance submitted, so its vendor / numbers / amount take priority."""
    f = read.form.fields = {k: _clean_form_value(v) for k, v in read.form.fields.items() if _clean_form_value(v)}
    if f.get("invoice_number_date"):
        num, dt = _split_number_date(f["invoice_number_date"])
        if num and not f.get("invoice_number"):
            f["invoice_number"] = num
        if dt and not f.get("invoice_date"):
            f["invoice_date"] = dt
    inv.submission_form = read.form
    inv.document_type = inv.document_type or "Bill Submission"
    if f.get("vendor_name"):
        inv.vendor = inv.vendor or Party()
        inv.vendor.name = f["vendor_name"]
    if f.get("company_name"):
        inv.buyer = inv.buyer or Party()
        inv.buyer.name = inv.buyer.name or f["company_name"]
    if inv.buyer and f.get("gstin") and not inv.buyer.gstin:
        gst = GSTIN_RE.search(f["gstin"].upper())
        inv.buyer.gstin = gst.group(0) if gst else None
    numbers = _form_numbers(f.get("invoice_number"))
    printed = inv.invoice_number  # from the attached tax invoice, if one was read
    if len(numbers) == 1 and printed and printed != numbers[0] and len(printed) == len(numbers[0]) \
            and sum(a != b for a, b in zip(printed, numbers[0])) <= 2:
        # Nearly identical: the form's digits are often handwritten corrections, the invoice is typed.
        read.form.fields["invoice_number_note"] = f"form reads {numbers[0]}, attached invoice reads {printed}"
        numbers = [printed]
    if numbers:
        inv.invoice_numbers = numbers
        inv.invoice_number = ", ".join(numbers)
    dates = find_dates(f.get("invoice_date", ""))
    if dates:
        unique = list(dict.fromkeys(dates))
        inv.invoice_date = ", ".join(unique)
        inv.invoice_date_iso = iso_date(unique[0]) if len(unique) == 1 else None
    amount = f.get("invoice_amount", "")
    nums = [to_number(n) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", amount.split("=")[0])]
    after_eq = re.findall(r"\d[\d,]*(?:\.\d+)?", amount.split("=")[1]) if "=" in amount else []
    if after_eq:
        inv.grand_total = to_number(after_eq[-1])
    elif len(nums) == 1:
        inv.grand_total = nums[0]
    elif nums:
        inv.grand_total = round(sum(nums), 2)
    if numbers and (len(nums) == len(numbers) or len(numbers) == 1):
        parts = nums if len(nums) == len(numbers) else [inv.grand_total]
        inv.bills = [BillEntry(invoice_number=n, amount=a, invoice_date=dates[i] if i < len(dates) else
                               (dates[0] if dates else None)) for i, (n, a) in enumerate(zip(numbers, parts))]
    cur = f.get("currency", "")
    if re.search(r"\bINR\b|rupee|indian", cur, re.I):
        inv.currency = "INR"
    elif re.match(r"^[A-Z]{3}\b", cur):
        inv.currency = cur[:3]
    inv.remarks = inv.remarks or f.get("towards") or f.get("item_service_description")


def _form_checks(inv: InvoiceData, doc_after_form: str, add) -> None:
    f = inv.submission_form.fields if inv.submission_form else {}
    amount = f.get("invoice_amount", "")
    if "=" in amount:
        parts = [to_number(n) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", amount.split("=")[0])]
        if len(parts) > 1 and inv.grand_total is not None:
            add("Invoice amount sum", _close(sum(parts), inv.grand_total, 0.01),
                f"{len(parts)} bill amounts add up to {_money(inv.grand_total)}",
                f"Bill amounts add up to {_money(sum(parts))} but the form total reads {_money(inv.grand_total)}")
    if len(inv.invoice_numbers) > 1:
        add("Bills listed", len(inv.bills) == len(inv.invoice_numbers),
            f"{len(inv.invoice_numbers)} invoice numbers, each with an amount",
            f"{len(inv.invoice_numbers)} invoice numbers but the amounts could not be matched one-to-one", "warn")
    if inv.invoice_numbers and len(doc_after_form.strip()) > 300:  # only when bill pages are attached
        missing = [n for n in inv.invoice_numbers if n not in doc_after_form]
        found = len(inv.invoice_numbers) - len(missing)
        add("Attached bills", not missing,
            f"All {found} bill numbers appear in the attached pages",
            f"{found} of {len(inv.invoice_numbers)} bill numbers found in the attached pages; not found: "
            f"{', '.join(missing)}. Check the attachments", "warn")


# ------------------------------------------------------------------------ generic (any document)

# Numbers printed as "<kind> No: X" / "<kind> #: X" / "REF: X", strongest kinds first.
_NUMBER_LABELS = [
    r"(?:tax\s+)?invoice", r"bill", r"receipt", r"quotation|quote", r"proforma(?:\s+invoice)?|\bpi", r"voucher",
    r"ref(?:erence)?",
]
_GENERIC_TOTALS = [
    r"grand\s*total", r"total\s*bill\s*amount", r"total\s*invoice\s*(?:value|amount)", r"(?:total\s*)?amount\s*payable",
    r"net\s*payable", r"gross\s*premium", r"total\s*amount", r"net\s*amount", r"total",
]
_LOOSE_MONEY_RE = re.compile(
    r"(?:₹|\brs\.?|\binr|\$|£|€)\s*(\d[\d,]*(?:\.\d{1,2})?)|(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+\.\d{2})(?![\d,])", re.I)
_VENDOR_HINT_RE = re.compile(
    r"\b(pvt|private|ltd|limited|llp|industries|services|systems|emporium|traders|enterprises|insurance|hotel|"
    r"stores?|company|corporation|agencies|solutions|associates|motors|automobiles)\b", re.I)


def _money_values(text: str) -> list[float]:
    out = []
    for m in _LOOSE_MONEY_RE.finditer(text):
        try:
            out.append(to_number(m.group(1) or m.group(2)))
        except ValueError:
            pass
    return out


def _generic_number(lines: list[str]) -> str | None:
    for kind in _NUMBER_LABELS:
        pat = re.compile(rf"\b(?:{kind})\s*(?:no|number|num|#)?\s*\.?\s*[:#]\s*#?\s*([A-Z0-9][A-Z0-9/\-]*\d[A-Z0-9/\-]*)",
                         re.I)
        for line in lines:
            m = pat.search(line)
            if m:
                return m.group(1).strip("/-")
    return None


def _generic_date(lines: list[str]) -> str | None:
    for line in lines:
        m = re.search(r"\b(?:invoice\s*|bill\s*|receipt\s*)?(?:date|dated|dt)\b\.?\s*[:\-]?\s*(.{6,30})", line, re.I)
        if m and find_dates(m.group(1)):
            return find_dates(m.group(1))[0]
    return None


def _generic_total(lines: list[str]) -> float | None:
    for label in _GENERIC_TOTALS:
        pat = re.compile(rf"\b{label}\b", re.I)
        hits = [l for l in lines if pat.search(l) and not re.search(r"sub\s*-?\s*total|total\s*(tax|qty|quantity|items?)",
                                                                       l, re.I)]
        for line in reversed(hits):  # the payable total is printed last
            vals = _money_values(line[pat.search(line).end():])
            if vals:
                return vals[-1]
    return None


def _generic_vendor(lines: list[str]) -> str | None:
    pans = _own_pans()
    m = re.search(r"\bfor\s*:?\s*([A-Z][A-Za-z0-9&.,()' -]{2,}?\b(?:Pvt\.?\s*Ltd|Private\s+Limited|Limited|Ltd|LLP)\b\.?)",
                  "\n".join(lines), re.I)
    if m and "amara raja" not in m.group(1).lower() and m.group(1)[0].isupper():
        return m.group(1).strip(" .,")
    for line in lines[:8]:  # letterhead: first lines naming a business
        t = line.strip()
        if re.match(r"^(name|address|to|bill|ship|customer|buyer|consignee|proposer|insured|principal|contractor|"
                    r"kind\s+att)", t, re.I):
            continue  # a labelled field ("Name of Principal ..."), not the letterhead
        if _VENDOR_HINT_RE.search(t) and "amara raja" not in t.lower() and not re.search(r"[:@]|\d{4}", t) \
                and not re.search(r"\b(quote|quotation|invoice|receipt|bill|proposal|estimate|statement)\b", t, re.I) \
                and not any(p in t for p in pans):
            return t.strip(" .,")
    return None


def _read_generic(inv: InvoiceData, rows: list[Row]) -> None:
    """Last resort for receipts, quotations, proformas and other layouts: fill what is still missing."""
    lines = [r.text for r in rows]
    if not inv.invoice_number:
        inv.invoice_number = _generic_number(lines)
    if not inv.invoice_date:
        inv.invoice_date = _generic_date(lines)
        inv.invoice_date_iso = iso_date(inv.invoice_date)
    if inv.grand_total is None:
        inv.grand_total = _generic_total(lines)
    if not (inv.vendor and inv.vendor.name) or _NOT_NAME_RE.search(inv.vendor.name or "") \
            or re.match(r"^\s*(ref|to|sub|date|gst)\b", inv.vendor.name or "", re.I):
        name = _generic_vendor(lines)
        if name:
            inv.vendor = inv.vendor or Party()
            inv.vendor.name = name
    if not inv.taxes:
        for line in lines:
            m = re.search(r"\b(CGST|SGST|UTGST|IGST)\b", line, re.I)
            vals = _money_values(line[m.end():]) if m else []
            if vals:
                pct = PERCENT_RE.search(line)
                name = m.group(1).upper()
                if not any(t.name == name and t.amount == vals[-1] for t in inv.taxes):
                    inv.taxes.append(TaxLine(name=name, rate=float(pct.group(1)) if pct else None, amount=vals[-1]))
    if inv.taxable_value is None:
        for line in lines:
            m = re.search(r"\b(sub\s*-?\s*total|total\s*assessable\s*value|taxable\s*(?:value|amount))\b", line, re.I)
            vals = _money_values(line[m.end():]) if m else []
            if vals:
                inv.taxable_value = vals[-1]
    if inv.currency == "INR" and re.search(r"\$\s*\d", "\n".join(lines)) and not re.search(r"₹|\brs\.?\s*\d|\binr\b",
                                                                                         "\n".join(lines), re.I):
        inv.currency = "USD"


def _solve_from_rates(inv: InvoiceData, doc: str) -> None:
    """When taxable + taxes != total but every tax has a rate, the taxable value must be
    (total - round off) / (1 + sum of rates). Accept it, and each tax = taxable x rate, only when those exact
    figures are printed in the document, so a line item misread (e.g. a rate taken as the amount) cannot win."""
    if inv.grand_total is None or not inv.taxes or any(t.rate is None for t in inv.taxes):
        return
    ro = inv.round_off or 0
    current = (inv.taxable_value or 0) + sum(t.amount for t in inv.taxes) + ro
    if inv.taxable_value is not None and abs(current - inv.grand_total) <= 1:
        return  # already consistent
    printed = _money_values(doc)
    target = (inv.grand_total - ro) / (1 + sum(t.rate for t in inv.taxes) / 100)
    taxable = next((x for x in printed if abs(x - target) <= 1), None)
    if taxable is None:
        return
    amounts = []
    for t in inv.taxes:
        expected = taxable * t.rate / 100
        found = next((x for x in printed if abs(x - expected) <= 0.5), None)
        if found is None:
            return
        amounts.append(found)
    inv.taxable_value = taxable
    for t, amount in zip(inv.taxes, amounts):
        t.amount = amount
    inv.total_tax = round(sum(amounts), 2)


def _reconcile_amounts(inv: InvoiceData, doc: str) -> None:
    """Make taxable value, taxes and total agree when the layout misread which number is which."""
    unique: list[TaxLine] = []
    for t in inv.taxes:  # the same tax often repeats in the item row and the total row
        if not any(u.name == t.name and abs(u.amount - t.amount) < 0.01 for u in unique):
            unique.append(t)
    inv.taxes = unique
    v = inv.taxable_value
    for t in inv.taxes:  # "IGST@18% 4500.00 / 810.00": tax and taxable value swapped
        if v and t.rate and abs(v - t.amount * t.rate / 100) <= 1 and abs(t.amount - v * t.rate / 100) > 1:
            t.amount, v = v, t.amount
    inv.taxable_value = v
    _solve_from_rates(inv, doc)
    if inv.grand_total is not None and inv.taxes:
        tax = sum(t.amount for t in inv.taxes)
        computed = (inv.taxable_value or 0) + tax + (inv.round_off or 0)
        if inv.taxable_value is None or abs(computed - inv.grand_total) > 1:
            target = inv.grand_total - tax - (inv.round_off or 0)
            # only accept a value the document actually prints
            if any(abs(x - target) <= 1 for x in _money_values(doc)):
                inv.taxable_value = round(target, 2)
                inv.total_tax = round(tax, 2)
    if inv.grand_total is not None and inv.taxable_value is not None and inv.taxable_value > inv.grand_total + 1:
        inv.taxable_value = None  # a subtotal above the total is a misread; leave it blank, not wrong


# --------------------------------------------------------------------------------- key fields

REQUIRED_FIELDS = ("vendor_name", "invoice_number", "invoice_date", "total_amount")
_COMPANY_RE = re.compile(
    r"\bfor\s+([A-Z][A-Za-z0-9&.,()' -]{2,}?\b(?:Pvt\.?\s*Ltd|Private\s+Limited|Limited|Ltd|LLP|Industries|Traders|"
    r"Enterprises|Corporation|Company)\b\.?)", re.I)


def _fill_required(inv: InvoiceData, rows: list[Row], doc: str) -> None:
    """Second chances for the required fields when the layout-based reading missed them."""
    if not (inv.vendor and inv.vendor.name):
        m = _COMPANY_RE.search(doc)  # Tally signs off with "for <Company Name>"
        if m:
            inv.vendor = inv.vendor or Party()
            inv.vendor.name = m.group(1).strip(" .,")
    if inv.grand_total is None:
        from_words = words_to_amount(inv.amount_in_words)
        rupee = [to_number(m.group(1)) for m in re.finditer(r"₹\s*((?:\d{1,3}(?:[.,]\d{2,3})+|\d+)\.\d{2})", doc)]
        inv.grand_total = from_words if from_words else (max(rupee) if rupee else None)


# PANs of the buying group (set with OCR_OWN_PANS). A GSTIN carrying one belongs to the buyer side;
# any other valid GSTIN in the document is the vendor's.
def _own_pans() -> set[str]:
    from .config import get_settings
    return {p.upper() for p in get_settings().own_pans}


_GSTIN_LOOSE = re.compile(r"\b[0-9OILSBZG]{2}[A-Z0-9]{5}[0-9OILSBZG]{4}[A-Z0-9]{2}[Z2][A-Z0-9]\b")
_TO_DIGIT = {"O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"}
_TO_ALPHA = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}


def repair_gstin(token: str) -> str | None:
    """Fix OCR letter/digit confusions using the GSTIN layout (2 digits, 5 letters, 4 digits, letter,
    entity code, 'Z', check char); accepted only if the check digit then validates."""
    t = list(token.upper())
    if len(t) != 15:
        return None
    for i in (0, 1, 7, 8, 9, 10):
        t[i] = _TO_DIGIT.get(t[i], t[i])
    for i in (2, 3, 4, 5, 6, 11):
        t[i] = _TO_ALPHA.get(t[i], t[i])
    t[13] = "Z"
    candidates = ["".join(t)]
    for i in (12, 14):  # entity code / check char can be either a digit or a letter
        alt = _TO_DIGIT.get(t[i]) or _TO_ALPHA.get(t[i])
        if alt:
            candidates.append("".join(t[:i] + [alt] + t[i + 1:]))
    return next((c for c in candidates if GSTIN_RE.fullmatch(c) and gstin_valid(c)), None)


def find_gstins(doc: str) -> list[str]:
    """Valid GSTINs in document order, including ones repaired from OCR confusions."""
    out = []
    for tok in _GSTIN_LOOSE.findall(doc.upper()):
        g = tok if (GSTIN_RE.fullmatch(tok) and gstin_valid(tok)) else repair_gstin(tok)
        if g and g not in out:
            out.append(g)
    return out


def assign_gstins(inv: InvoiceData, doc: str) -> None:
    pans = _own_pans()
    found = find_gstins(doc)
    own = [g for g in found if g[2:12] in pans]
    other = [g for g in found if g[2:12] not in pans]
    inv.vendor = inv.vendor or Party()
    inv.buyer = inv.buyer or Party()
    v, b = inv.vendor, inv.buyer
    for p in (v, b):  # an invalid GSTIN read by layout -> its repaired form if one was found
        if p.gstin and not gstin_valid(p.gstin):
            p.gstin = next((g for g in found if g[2:12] == p.gstin[2:12]), p.gstin)
    if v.gstin and v.gstin[2:12] in pans and (other or len(own) == 1):
        if not b.gstin:
            b.gstin = v.gstin  # a lone group-company GSTIN is the buyer (vendor may be a person)
        v.gstin = None
    if b.gstin and b.gstin[2:12] not in pans and own:
        b.gstin = None  # a non-group GSTIN was read as the buyer's
    if not v.gstin:
        v.gstin = other[0] if other else (own[0] if len(own) >= 2 else None)
    if not b.gstin:
        b.gstin = next((g for g in own if g != v.gstin), None)
    if not (v.name or v.gstin):
        inv.vendor = None
    if not (b.name or b.gstin):
        inv.buyer = None


def key_fields(inv: InvoiceData | None) -> InvoiceFields:
    """The 5 core fields every response carries; everything else found goes to additional_fields()."""
    if inv is None:
        return InvoiceFields()
    vendor = inv.vendor.name if inv.vendor else None
    date_value = inv.invoice_date_iso or inv.invoice_date
    found_any = any(v not in (None, "") for v in (vendor, inv.invoice_number, date_value, inv.grand_total))
    return InvoiceFields(
        vendor_name=vendor, invoice_number=inv.invoice_number, invoice_date=date_value,
        total_amount=inv.grand_total, currency=inv.currency if found_any else None,
    )


def _drop_empty(value):
    """Remove None / "" / [] / {} at every level, so only what was really found remains."""
    if isinstance(value, dict):
        cleaned = {k: _drop_empty(v) for k, v in value.items()}
        return {k: v for k, v in cleaned.items() if v not in (None, "", [], {})}
    if isinstance(value, list):
        cleaned = [_drop_empty(v) for v in value]
        return [v for v in cleaned if v not in (None, "", [], {})]
    return value


# Form rows already reported as core fields (or folded into them).
_FORM_ROWS_IN_CORE = {"vendor_name", "invoice_number", "invoice_date", "invoice_amount", "currency",
                      "invoice_number_date"}


def additional_fields(inv: InvoiceData | None) -> dict:
    """Every field found in THIS document apart from the 5 core ones: no fixed list, nothing empty.
    Keys are snake_case; lists (line items, bills ...) and small objects (approval, cost sheet) stay nested."""
    if inv is None:
        return {}
    out: dict = {}

    def put(key: str, value) -> None:
        value = _drop_empty(value)
        if value not in (None, "", [], {}) and key not in out:
            out[key] = value

    put("document_type", inv.document_type)
    if len(inv.invoice_numbers) > 1:
        put("invoice_numbers", inv.invoice_numbers)
    if inv.invoice_date and inv.invoice_date_iso and inv.invoice_date != inv.invoice_date_iso:
        put("invoice_date_as_printed", inv.invoice_date)
    for role, party in (("vendor", inv.vendor), ("buyer", inv.buyer), ("consignee", inv.consignee)):
        if party is None:
            continue
        if role == "consignee" and inv.buyer and (party.name, party.gstin) == (inv.buyer.name, inv.buyer.gstin):
            continue  # same as the buyer
        for key, value in party.model_dump().items():
            if not (role == "vendor" and key == "name"):  # vendor name is a core field
                put(f"{role}_{key}", value)
    if inv.e_invoice:
        put("irn", inv.e_invoice.irn)
        put("ack_no", inv.e_invoice.ack_no)
        put("ack_date", inv.e_invoice.ack_date)
    for key, value in inv.references.items():
        put(key, value)
    put("subtotal", inv.taxable_value)
    by_tax: dict[str, list[TaxLine]] = {}
    for t in inv.taxes:
        by_tax.setdefault(t.name.lower(), []).append(t)
    for name, lines in by_tax.items():
        put(name, round(sum(t.amount for t in lines), 2))
        rates = {t.rate for t in lines if t.rate is not None}
        if len(rates) == 1:
            put(f"{name}_rate", rates.pop())
    put("total_tax", inv.total_tax)
    put("round_off", inv.round_off)
    put("amount_in_words", inv.amount_in_words)
    put("tax_amount_in_words", inv.tax_amount_in_words)
    put("line_items", [i.model_dump() for i in inv.line_items])
    put("bills", [b.model_dump() for b in inv.bills])
    if inv.submission_form:
        sf = inv.submission_form
        put("form_title", sf.title)
        for key, value in sf.fields.items():
            if key not in _FORM_ROWS_IN_CORE:
                put(key, value)
        put("prepared_by", sf.prepared_by)
        put("verified_by", sf.verified_by)
        put("approved_by", sf.approved_by)
    if inv.approval:
        put("approval", inv.approval.model_dump())
    if inv.cost_sheet:
        put("cost_sheet", inv.cost_sheet.model_dump())
    put("remarks", inv.remarks)
    put("jurisdiction", inv.jurisdiction)
    return out


def validation(inv: InvoiceData | None, missing: list[str]) -> Validation:
    """The automatic checks in plain sentences: status + what to look at."""
    if inv is None:
        return Validation(status="fail", issues=[
            "No invoice details were found in this document. Required: " + ", ".join(missing) + "."])
    issues = [f"{c.name}: {c.message}" for c in inv.checks if c.status != "pass"]
    order = {"fail": 0, "warn": 1}
    issues.sort(key=lambda m: order.get(next((c.status for c in inv.checks if m.startswith(c.name + ":")), "warn"), 2))
    return Validation(status=inv.status, issues=issues,
                      checks_passed=sum(1 for c in inv.checks if c.status == "pass"), checks_total=len(inv.checks))


def missing_required(fields: InvoiceFields) -> list[str]:
    return [name for name in REQUIRED_FIELDS if getattr(fields, name) in (None, "")]


# ------------------------------------------------------------------------------------------- entry

def extract_invoice(rows: list[Row]) -> InvoiceData | None:
    if not rows:
        return None
    text_lines = [r.text for r in rows]
    doc = "\n".join(text_lines)
    form = submission_form(rows)
    # With a cover form, the GST invoice (if attached) is read from the pages after the form only.
    inv_rows = rows[form.end + 1:] if form else rows
    inv_doc = "\n".join(r.text for r in inv_rows)
    if form:
        has_invoice = bool(re.search(r"invoice", inv_doc, re.I) and GSTIN_RE.search(inv_doc.upper()))
    else:
        has_invoice = bool(re.search(r"invoice", doc, re.I) and (GSTIN_RE.search(doc.upper()) or "HSN" in doc.upper()))
    appr, sheet = approval(doc), cost_sheet(text_lines)
    # Receipts, quotations, proformas ...: anything with a labelled number or total is worth reading.
    generic = _generic_total(text_lines) is not None or _generic_number(text_lines) is not None
    if not (has_invoice or appr or sheet or form or generic):
        return None

    inv = InvoiceData(approval=appr, cost_sheet=sheet)
    if has_invoice:
        _read_tax_invoice(inv, inv_rows, inv_doc)
    if form:
        _apply_form(inv, form, rows)
    _read_generic(inv, inv_rows if form else rows)
    _reconcile_amounts(inv, inv_doc if form else doc)
    if not inv.invoice_numbers and inv.invoice_number:
        inv.invoice_numbers = [inv.invoice_number]
    assign_gstins(inv, doc)
    inv.checks = run_checks(inv, inv_doc if form else "")
    statuses = {c.status for c in inv.checks}
    inv.status = "fail" if "fail" in statuses else "warn" if "warn" in statuses else "pass"
    return inv


def _read_tax_invoice(inv: InvoiceData, rows: list[Row], doc: str) -> None:
    """GST tax invoice (Tally and simple layouts)."""
    header, split_x = header_fields(rows)
    vendor, consignee, buyer = parties(left_column(rows, split_x, header))
    items, taxes, round_off, end = line_items(rows)
    tot = totals(rows, end)
    foot = footer(rows, split_x)
    inv.document_type = foot.get("document_type")
    inv.invoice_number = header.pop("invoice_number", None)
    inv.invoice_date = header.pop("invoice_date", None)
    inv.invoice_date_iso = iso_date(inv.invoice_date)
    inv.references = header
    inv.e_invoice = e_invoice(rows)
    inv.vendor, inv.consignee, inv.buyer = vendor, consignee, buyer or consignee
    if inv.vendor and not inv.vendor.pan and foot.get("company_pan"):
        inv.vendor.pan = foot["company_pan"]
    inv.line_items, inv.taxes, inv.round_off = items, taxes, round_off
    rates = tot.get("summary_rates") or []
    for t, r in zip([t for t in taxes if t.rate is None], rates):
        t.rate = r  # rates printed in the HSN summary, in column order
    inv.taxable_value = tot.get("taxable_value")
    if inv.taxable_value is None and items:
        inv.taxable_value = round(sum(i.amount for i in items if i.amount is not None), 2)
    inv.total_tax = tot.get("total_tax")
    if inv.total_tax is None and taxes:
        inv.total_tax = round(sum(t.amount for t in taxes), 2)
    inv.grand_total = tot.get("grand_total")
    inv.amount_in_words = tot.get("amount_in_words")
    _fill_required(inv, rows, doc)
    inv.tax_amount_in_words = tot.get("tax_amount_in_words")
    inv.remarks = foot.get("remarks")
    inv.jurisdiction = foot.get("jurisdiction")
