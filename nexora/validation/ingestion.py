"""Safe dataset ingestion (CSV/TSV and optional .xlsx).

Everything is parsed in memory from bytes: no uploaded file is ever written to
disk, so there is nothing to clean up and nothing left behind. All values are
kept as the exact original text (dtype=str, no NA conversion), so "007" stays
"007" and blank cells stay blank.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import PurePath

import pandas as pd

ALLOWED_EXTENSIONS = (".csv", ".tsv", ".txt", ".xlsx")
CANDIDATE_DELIMITERS = [",", ";", "\t", "|"]
ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]


class IngestError(ValueError):
    """A file could not be accepted. The message is safe to show to the user."""


@dataclass
class IngestResult:
    dataframe: pd.DataFrame
    filename: str
    sha256: str
    size_bytes: int
    encoding: str | None
    delimiter: str | None
    warnings: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.dataframe)

    @property
    def n_cols(self) -> int:
        return self.dataframe.shape[1]


def safe_filename(name: str) -> str:
    base = PurePath(str(name).replace("\\", "/")).name
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    return (base or "upload")[:120]


def _decode(data: bytes) -> tuple[str, str, list[str]]:
    warnings: list[str] = []
    for enc in ENCODINGS:
        try:
            text = data.decode(enc)
        except UnicodeDecodeError:
            continue
        if enc in ("cp1252", "latin-1"):
            warnings.append(f"The file is not valid UTF-8; it was read as {enc}. Check that accented characters look right.")
        return text, enc, warnings
    raise IngestError("The file could not be decoded as text. Save it as UTF-8 CSV and try again.")


def _detect_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:20])
    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(CANDIDATE_DELIMITERS)).delimiter
    except csv.Error:
        counts = {d: sample.count(d) for d in CANDIDATE_DELIMITERS}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ","


def _clean_header(header: list[str], warnings: list[str]) -> list[str]:
    names: list[str] = []
    seen: dict[str, int] = {}
    for i, raw in enumerate(header, 1):
        name = (raw or "").strip()
        if not name:
            name = f"unnamed_{i}"
            warnings.append(f"Column {i} had no name and was called '{name}'.")
        if name in seen:
            seen[name] += 1
            new = f"{name}__{seen[name]}"
            warnings.append(f"Duplicate column name '{name}' was renamed to '{new}'.")
            name = new
        else:
            seen[name] = 1
        names.append(name)
    return names


def _check_limits(df: pd.DataFrame, max_rows: int, max_cols: int) -> None:
    if df.shape[1] > max_cols:
        raise IngestError(f"The file has {df.shape[1]} columns; the limit is {max_cols}.")
    if len(df) > max_rows:
        raise IngestError(f"The file has {len(df):,} rows; the limit is {max_rows:,}. Split the file or raise DATASENTINEL_MAX_ROWS.")


def _parse_csv(data: bytes, max_rows: int, max_cols: int) -> tuple[pd.DataFrame, str, str, list[str]]:
    if b"\x00" in data[:8192]:
        raise IngestError("This looks like a binary file, not a CSV. Please upload a text CSV file.")
    text, enc, warnings = _decode(data)
    if not text.strip():
        raise IngestError("The file is empty.")
    delim = _detect_delimiter(text)
    try:
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=delim, strict=False)
        rows = []
        blank_lines = 0
        for r in reader:
            if not r or all(not c.strip() for c in r):
                blank_lines += 1
                continue
            rows.append(r)
            if len(rows) > max_rows + 1:
                raise IngestError(f"The file has more than {max_rows:,} rows, which is the limit.")
    except csv.Error as exc:
        raise IngestError(f"The file could not be parsed as CSV: {exc}") from exc
    if not rows:
        raise IngestError("The file has no content.")
    header = _clean_header(rows[0], warnings)
    body = rows[1:]
    if not body:
        raise IngestError("The file has a header row but no data rows.")
    width = len(header)
    too_long = [i + 2 for i, r in enumerate(body) if len(r) > width]
    if too_long:
        raise IngestError(
            f"{len(too_long)} row(s) have more fields than the header ({width}); first problem near row {too_long[0]}. "
            "This usually means an unquoted delimiter inside a value.")
    short = sum(len(r) < width for r in body)
    if short:
        warnings.append(f"{short} row(s) had fewer fields than the header and were padded with blanks.")
    if blank_lines:
        warnings.append(f"{blank_lines} blank line(s) were ignored; row numbers count only non-blank records.")
    if width == 1 and delim == ",":
        warnings.append("Only one column was found. If your file uses another separator, check the file format.")
    padded = [r + [""] * (width - len(r)) for r in body]
    df = pd.DataFrame(padded, columns=header, dtype=object).astype(str)
    _check_limits(df, max_rows, max_cols)
    return df, enc, delim, warnings


def _parse_xlsx(data: bytes, max_rows: int, max_cols: int, sheet: str | int = 0) -> tuple[pd.DataFrame, list[str]]:
    if not data.startswith(b"PK"):
        raise IngestError("The file does not look like a valid .xlsx workbook.")
    try:
        raw = pd.read_excel(io.BytesIO(data), sheet_name=sheet, dtype=str, keep_default_na=False, na_filter=False,
                            engine="openpyxl", header=None)
    except Exception as exc:  # openpyxl raises many different error types
        raise IngestError(f"The Excel file could not be read ({type(exc).__name__}). Is it password-protected or corrupted?") from exc
    warnings = ["Excel cells were converted to text; dates may appear as 'YYYY-MM-DD HH:MM:SS'. Check formats."]
    raw = raw.replace(r"^\s*$", "", regex=True)
    raw = raw.loc[~(raw == "").all(axis=1)]
    if raw.empty:
        raise IngestError("The selected worksheet is empty.")
    header = _clean_header([str(x) for x in raw.iloc[0].tolist()], warnings)
    body = raw.iloc[1:].reset_index(drop=True)
    if body.empty:
        raise IngestError("The worksheet has a header row but no data rows.")
    body.columns = header
    df = body.astype(str)
    _check_limits(df, max_rows, max_cols)
    return df, warnings


def list_excel_sheets(data: bytes) -> list[str]:
    """Sheet names of an .xlsx file (empty list if it cannot be read)."""
    import io
    try:
        with pd.ExcelFile(io.BytesIO(data), engine="openpyxl") as xl:
            return list(xl.sheet_names)
    except Exception:
        return []


def load_upload(data: bytes, filename: str, *, max_bytes: int = 20 * 1024 * 1024, max_rows: int = 200_000,
                max_cols: int = 200, sheet: str | int = 0) -> IngestResult:
    """Validate and parse an uploaded file. Raises IngestError with a clear message on any problem."""
    name = safe_filename(filename)
    ext = PurePath(name.lower()).suffix
    if ext not in ALLOWED_EXTENSIONS:
        raise IngestError(f"Files of type '{ext or 'unknown'}' are not accepted. Upload a .csv, .tsv or .xlsx file.")
    if not data:
        raise IngestError("The file is empty.")
    if len(data) > max_bytes:
        raise IngestError(f"The file is {len(data) / 1024 / 1024:.1f} MB; the limit is {max_bytes / 1024 / 1024:.0f} MB.")
    sha = hashlib.sha256(data).hexdigest()
    if ext == ".xlsx":
        df, warnings = _parse_xlsx(data, max_rows, max_cols, sheet)
        return IngestResult(df, name, sha, len(data), None, None, warnings)
    df, enc, delim, warnings = _parse_csv(data, max_rows, max_cols)
    return IngestResult(df, name, sha, len(data), enc, delim, warnings)


def schema_hash(columns: list[str]) -> str:
    return hashlib.sha256("\x1f".join(columns).encode("utf-8")).hexdigest()[:16]
