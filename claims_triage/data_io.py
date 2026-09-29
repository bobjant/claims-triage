"""Reading input files: claim batches (JSON or CSV) and the policy register.

WHERE THIS FITS
    Used by the `ingest_claims` skill (Intake agent) and at startup to load policies.

DESIGN: "tolerant ingestion"
    Real input files are messy. We separate two kinds of problems:
      * The whole file is unusable (missing, wrong format, broken JSON): raise IngestError, and the
        routine aborts with a clear message.
      * One row is bad (too many/too few CSV cells, a JSON entry that isn't an object): skip that
        row, record it in `parse_errors`, and carry on with the rest.
    Bad *values* (e.g. amount "12k", date "2026/08/40") are NOT handled here. They're kept as-is so
    the validation step can report them with a proper issue code.
"""
from __future__ import annotations

import csv
import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import ROOT

# The fields every claim record is expected to carry (description is optional).
CLAIM_FIELDS = [
    "claim_id",
    "policy_number",
    "claim_type",
    "loss_date",
    "report_date",
    "claim_amount",
    "description",
]


class IngestError(Exception):
    """The whole file is unreadable (as opposed to individual bad rows)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code        # machine-readable, e.g. "file_not_found", "malformed_json"
        self.message = message  # human-readable explanation


# ---------------------------------------------------------------------------------------------
# Normalisation helpers: accept the sloppy formats people actually type.
# ---------------------------------------------------------------------------------------------
def normalize_claim_id(raw: Any) -> str:
    """'c2031', 'C-2031', ' C 2031 ' -> 'C-2031'. Unrecognised values are returned stripped."""
    text = str(raw or "").strip()
    m = re.fullmatch(r"[cC][\s\-_]?(\d+)", text)
    return f"C-{m.group(1)}" if m else text


def normalize_policy_number(raw: Any) -> str:
    """'pol5521', 'POL 5521', 'POL-5521' -> 'POL-5521'."""
    text = str(raw or "").strip()
    m = re.fullmatch(r"[pP][oO][lL][\s\-_]?(\d+)", text)
    return f"POL-{m.group(1)}" if m else text


def parse_date(raw: Any) -> Optional[date]:
    """Parse an ISO date (YYYY-MM-DD). Returns None for empty or invalid input instead of raising."""
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw).strip())
    except ValueError:
        return None


def resolve_path(path: str) -> Path:
    """Relative paths are resolved against the project root, so 'data/claims.json' works from anywhere."""
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        return p
    return ROOT / p


def _coerce_amount(raw: Any) -> Any:
    """Turn '8,500' or '8500' into a number. Non-numeric text is left as a string for validation to flag."""
    if isinstance(raw, (int, float)) or raw is None:
        return raw
    text = str(raw).strip().replace(",", "")
    if text == "":
        return None
    try:
        num = float(text)
        return int(num) if num.is_integer() else num
    except ValueError:
        return str(raw).strip()  # left as-is; validation flags it as INVALID_AMOUNT


def _normalise_record(rec: Dict[str, Any], row: int) -> Dict[str, Any]:
    """Clean one raw record into our standard shape and tag it with its row number (`_row`)."""
    out: Dict[str, Any] = {}
    for key in CLAIM_FIELDS:
        val = rec.get(key)
        out[key] = val.strip() if isinstance(val, str) else val
    out["claim_id"] = normalize_claim_id(out.get("claim_id")) if out.get("claim_id") else ""
    if out.get("policy_number"):
        out["policy_number"] = normalize_policy_number(out["policy_number"])
    if isinstance(out.get("claim_type"), str):
        out["claim_type"] = out["claim_type"].lower().replace(" ", "_")   # "Water Damage" -> "water_damage"
    out["claim_amount"] = _coerce_amount(out.get("claim_amount"))
    out["_row"] = row  # fields starting with "_" are internal bookkeeping, never shown to agents as data
    return out


# ---------------------------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------------------------
def load_claims_file(path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (claims, parse_errors). Raises IngestError if the file itself can't be read."""
    p = resolve_path(path)
    if not p.exists():
        raise IngestError("file_not_found", f"No claims file at {p}")
    suffix = p.suffix.lower()
    if suffix == ".json":
        return _load_json(p)
    if suffix == ".csv":
        return _load_csv(p)
    raise IngestError("unsupported_format", f"Unsupported file type '{suffix}' (expected .json or .csv)")


def _load_json(p: Path):
    """JSON: either a top-level array of claims, or {"claims": [...]}."""
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IngestError("malformed_json", f"{p.name}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}")
    if isinstance(payload, dict) and isinstance(payload.get("claims"), list):
        payload = payload["claims"]
    if not isinstance(payload, list):
        raise IngestError("malformed_json", f"{p.name}: expected a JSON array of claim objects")
    claims, errors = [], []
    for i, rec in enumerate(payload, start=1):
        if not isinstance(rec, dict):   # e.g. a stray string or number in the array: skip that row only
            errors.append({"row": i, "code": "NOT_AN_OBJECT", "message": f"Row {i} is {type(rec).__name__}, expected an object"})
            continue
        claims.append(_normalise_record(rec, i))
    return claims, errors


def _load_csv(p: Path):
    """CSV: header row must contain all CLAIM_FIELDS. Rows with the wrong number of cells are skipped."""
    claims, errors = [], []
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing_cols = [c for c in CLAIM_FIELDS if c not in (reader.fieldnames or [])]
        if missing_cols:
            raise IngestError("malformed_csv", f"{p.name}: header is missing columns {missing_cols}")
        for i, rec in enumerate(reader, start=1):
            # csv.DictReader puts surplus cells under the key None...
            if None in rec:  # more cells than header columns
                errors.append({"row": i, "code": "EXTRA_COLUMNS",
                               "message": f"Row {i} has {len(rec[None])} unexpected extra cell(s); row skipped",
                               "raw_claim_id": rec.get("claim_id")})
                continue
            # ...and fills missing cells with the value None.
            if any(v is None for v in rec.values()):  # fewer cells than header columns
                errors.append({"row": i, "code": "MISSING_COLUMNS",
                               "message": f"Row {i} is truncated; row skipped",
                               "raw_claim_id": rec.get("claim_id")})
                continue
            claims.append(_normalise_record(rec, i))
    return claims, errors


def load_policies(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load the policy register into a dict keyed by normalised policy number."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return {normalize_policy_number(r["policy_number"]): r for r in records}
