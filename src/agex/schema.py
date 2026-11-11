"""Target schema and deterministic validation rules.

Validation is intentionally independent of any model: a field that fails these
checks is never auto-accepted, no matter how confident the model claims to be.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

SIGNAL_WORDS = {"Danger", "Warning", "None"}


def cas_checksum_ok(cas: str) -> bool:
    """CAS registry numbers carry a check digit: weighted sum of digits mod 10."""
    m = re.fullmatch(r"(\d{2,7})-(\d{2})-(\d)", cas or "")
    if not m:
        return False
    digits = (m.group(1) + m.group(2))[::-1]
    total = sum(int(d) * (i + 1) for i, d in enumerate(digits))
    return total % 10 == int(m.group(3))


@dataclass(frozen=True)
class Field:
    name: str
    kind: str  # "text" | "number" | "enum"
    required: bool = True
    check: Callable[[Any, dict], bool] | None = None
    tolerance: float = 0.0  # absolute tolerance for numeric comparison against gold

    def is_valid(self, value: Any, record: dict) -> bool:
        if value is None:
            return not self.required
        if self.kind == "number" and not isinstance(value, (int, float)):
            return False
        return True if self.check is None else bool(self.check(value, record))


def _fp_below_bp(v: float, rec: dict) -> bool:
    bp = rec.get("boiling_point_c")
    return -100 <= v <= 400 and (bp is None or v < bp)


FIELDS: list[Field] = [
    Field("product_name", "text", check=lambda v, r: 2 <= len(v) <= 80),
    Field("manufacturer", "text", check=lambda v, r: 2 <= len(v) <= 80),
    Field("cas_number", "text", check=lambda v, r: cas_checksum_ok(v)),
    Field("ghs_classification", "text", check=lambda v, r: len(v) >= 5),
    Field("signal_word", "enum", check=lambda v, r: v in SIGNAL_WORDS),
    Field("flash_point_c", "number", required=False, check=_fp_below_bp, tolerance=1.0),
    Field("boiling_point_c", "number", check=lambda v, r: -100 <= v <= 500, tolerance=1.0),
    Field("ph", "number", required=False, check=lambda v, r: 0 <= v <= 14, tolerance=0.1),
]
FIELD_NAMES = [f.name for f in FIELDS]
FIELD_BY_NAME = {f.name: f for f in FIELDS}


def validate(record: dict) -> dict[str, bool]:
    """Return {field: passes_rules} for a normalized record."""
    return {f.name: f.is_valid(record.get(f.name), record) for f in FIELDS}


def matches_gold(field: str, pred: Any, gold: Any) -> bool:
    f = FIELD_BY_NAME[field]
    if pred is None or gold is None:
        return pred is None and gold is None
    if f.kind == "number":
        try:
            return abs(float(pred) - float(gold)) <= f.tolerance
        except (TypeError, ValueError):
            return False
    return str(pred).strip().lower() == str(gold).strip().lower()
